from pathlib import Path
from types import SimpleNamespace
import warnings
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List, Callable

try:
    from flash_attn.modules.mha import FlashCrossAttention
except ModuleNotFoundError:
    FlashCrossAttention = None

if FlashCrossAttention or hasattr(F, 'scaled_dot_product_attention'):
    FLASH_AVAILABLE = True
else:
    FLASH_AVAILABLE = False

torch.backends.cudnn.deterministic = True


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(
        kpts: torch.Tensor,
        size: torch.Tensor) -> torch.Tensor:
    if not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(
        freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None,
                 gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ encode position vector """
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """ get confidence tokens """
        return (
            self.token(desc0.detach().float()).squeeze(-1),
            self.token(desc1.detach().float()).squeeze(-1))


class Attention(nn.Module):
    def __init__(self, allow_flash: bool) -> None:
        super().__init__()
        if allow_flash and not FLASH_AVAILABLE:
            warnings.warn(
                'FlashAttention is not available. For optimal speed, '
                'consider installing torch >= 2.0 or flash-attn.',
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE
        if allow_flash and FlashCrossAttention:
            self.flash_ = FlashCrossAttention()

    def forward(self, q, k, v) -> torch.Tensor:
        if self.enable_flash and q.device.type == 'cuda':
            if FlashCrossAttention:
                q, k, v = [x.transpose(-2, -3) for x in [q, k, v]]
                m = self.flash_(q.half(), torch.stack([k, v], 2).half())
                return m.transpose(-2, -3).to(q.dtype)
            else:  # use torch 2.0 scaled_dot_product_attention with flash
                args = [x.half().contiguous() for x in [q, k, v]]
                with torch.backends.cuda.sdp_kernel(enable_flash=True):
                    return F.scaled_dot_product_attention(*args).to(q.dtype)
        elif hasattr(F, 'scaled_dot_product_attention'):
            args = [x.contiguous() for x in [q, k, v]]
            return F.scaled_dot_product_attention(*args).to(q.dtype)
        else:
            s = q.shape[-1] ** -0.5
            attn = F.softmax(torch.einsum('...id,...jd->...ij', q, k) * s, -1)
            return torch.einsum('...ij,...jd->...id', attn, v)


class Transformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 flash: bool = False, bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3*embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2*embed_dim, 2*embed_dim),
            nn.LayerNorm(2*embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2*embed_dim, embed_dim)
        )

    def _forward(self, x: torch.Tensor,
                 encoding: Optional[torch.Tensor] = None):
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        if encoding is not None:
            q = apply_cached_rotary_emb(encoding, q)
            k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v)
        message = self.out_proj(
            context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))

    def forward(self, x0, x1, encoding0=None, encoding1=None):
        return self._forward(x0, encoding0), self._forward(x1, encoding1)


class CrossTransformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 flash: bool = False, bias: bool = True) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2*embed_dim, 2*embed_dim),
            nn.LayerNorm(2*embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2*embed_dim, embed_dim)
        )

        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1))
        if self.flash is not None:
            m0 = self.flash(qk0, qk1, v1)
            m1 = self.flash(qk1, qk0, v0)
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum('b h i d, b h j d -> b h i j', qk0, qk1)
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = torch.einsum('bhij, bhjd -> bhid', attn01, v1)
            m1 = torch.einsum('bhji, bhjd -> bhid', attn10.transpose(-2, -1), v0)
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2),
                           m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


def sigmoid_log_double_softmax(
        sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    """ create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(
        sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m+1, n+1), 0)
    scores[:, :m, :n] = (scores0 + scores1 + certainties)
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """ build assignment matrix from descriptors """
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**.25, mdesc1 / d**.25
        sim = torch.einsum('bmd,bnd->bmn', mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def scores(self, desc0: torch.Tensor, desc1: torch.Tensor):
        m0 = torch.sigmoid(self.matchability(desc0)).squeeze(-1)
        m1 = torch.sigmoid(self.matchability(desc1)).squeeze(-1)
        return m0, m1


def filter_matches(scores: torch.Tensor, th: float):
    """ obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    if th is not None:
        valid0 = mutual0 & (mscores0 > th)
    else:
        valid0 = mutual0
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1


class LightGlue(nn.Module):
    default_conf = {
        'name': 'lightglue',  # just for interfacing
        'input_dim': 256,  # input descriptor dimension (autoselected from weights)
        'descriptor_dim': 256,
        'n_layers': 9,
        'num_heads': 4,
        'flash': True,  # enable FlashAttention if available.
        'mp': False,  # enable mixed precision
        'depth_confidence': 0.95,  # early stopping, disable with -1
        'width_confidence': 0.99,  # point pruning, disable with -1
        'filter_threshold': 0.1,  # match threshold
        'weights': None,
    }

    required_data_keys = [
        'image0', 'image1']

    version = "v0.1_arxiv"
    url = "https://github.com/cvg/LightGlue/releases/download/{}/{}_lightglue.pth"

    features = {
        'superpoint': ('superpoint_lightglue', 256),
        'disk': ('disk_lightglue', 128)
    }

    def __init__(self, features='superpoint', **conf) -> None:
        super().__init__()
        self.conf = {**self.default_conf, **conf}
        if features is not None:
            assert (features in list(self.features.keys()))
            self.conf['weights'], self.conf['input_dim'] = \
                self.features[features]
        self.conf = conf = SimpleNamespace(**self.conf)

        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(
                conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(2, head_dim, head_dim)

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim
        self.self_attn = nn.ModuleList(
            [Transformer(d, h, conf.flash) for _ in range(n)])
        self.cross_attn = nn.ModuleList(
            [CrossTransformer(d, h, conf.flash) for _ in range(n)])
        self.log_assignment = nn.ModuleList(
            [MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList([
            TokenConfidence(d) for _ in range(n-1)])

        if features is not None:
            fname = f'{conf.weights}_{self.version}.pth'.replace('.', '-')
            state_dict = torch.hub.load_state_dict_from_url(
                self.url.format(self.version, features), file_name=fname)
            self.load_state_dict(state_dict, strict=False)
        elif conf.weights is not None:
            path = Path(__file__).parent
            path = path / 'weights/{}.pth'.format(self.conf.weights)
            state_dict = torch.load(str(path), map_location='cpu')
            self.load_state_dict(state_dict, strict=False)

    def forward(self, data: dict) -> dict:
        """
        Match keypoints and descriptors between two images

        Input (dict):
            image0: dict
                keypoints: [B x M x 2]
                descriptors: [B x M x D]
                image: [B x C x H x W] or image_size: [B x 2]
            image1: dict
                keypoints: [B x N x 2]
                descriptors: [B x N x D]
                image: [B x C x H x W] or image_size: [B x 2]
        Output (dict):
            log_assignment: [B x M+1 x N+1]
            matches0: [B x M]
            matching_scores0: [B x M]
            matches1: [B x N]
            matching_scores1: [B x N]
            matches: List[[Si x 2]], scores: List[[Si]]
        """
        with torch.autocast(enabled=self.conf.mp, device_type='cuda'):
            return self._forward(data)

    def _forward(self, data: dict) -> dict:
        for key in self.required_data_keys:
            assert key in data, f'Missing key {key} in data'
        data0, data1 = data['image0'], data['image1']
        kpts0_, kpts1_ = data0['keypoints'], data1['keypoints']
        b, m, _ = kpts0_.shape
        b, n, _ = kpts1_.shape
        size0, size1 = data0.get('image_size'), data1.get('image_size')
        size0 = size0 if size0 is not None else data0['image'].shape[-2:][::-1]
        size1 = size1 if size1 is not None else data1['image'].shape[-2:][::-1]
        kpts0 = normalize_keypoints(kpts0_, size=size0)
        kpts1 = normalize_keypoints(kpts1_, size=size1)

        assert torch.all(kpts0 >= -1) and torch.all(kpts0 <= 1)
        assert torch.all(kpts1 >= -1) and torch.all(kpts1 <= 1)

        desc0 = data0['descriptors'].detach()
        desc1 = data1['descriptors'].detach()

        assert desc0.shape[-1] == self.conf.input_dim
        assert desc1.shape[-1] == self.conf.input_dim

        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()

        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)

        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        # GNN + final_proj + assignment
        do_early_stop = self.conf.depth_confidence > 0
        do_point_pruning = self.conf.width_confidence > 0
        if do_point_pruning:
            ind0 = torch.arange(0, m, device=kpts0.device)[None]
            ind1 = torch.arange(0, n, device=kpts0.device)[None]
            # We store the index of the layer at which pruning is detected.
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)
        token0, token1 = None, None
        for i in range(self.conf.n_layers):
            desc0, desc1 = self.self_attn[i](
                desc0, desc1, encoding0, encoding1)
            desc0, desc1 = self.cross_attn[i](desc0, desc1)
            if i == self.conf.n_layers - 1:
                continue  # no early stopping or adaptive width at last layer

            if do_early_stop:
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self.check_if_stop(token0, token1, i, m+n):
                    break
            if do_point_pruning:
                scores0, scores1 = self.log_assignment[i].scores(desc0, desc1)
                mask0 = self.get_pruning_mask(token0, scores0, i)
                mask1 = self.get_pruning_mask(token1, scores1, i)
                ind0, ind1 = ind0[mask0][None], ind1[mask1][None]
                desc0, desc1 = desc0[mask0][None], desc1[mask1][None]
                if desc0.shape[-2] == 0 or desc1.shape[-2] == 0:
                    break
                encoding0 = encoding0[:, :, mask0][:, None]
                encoding1 = encoding1[:, :, mask1][:, None]
                prune0[:, ind0] += 1
                prune1[:, ind1] += 1

        scores, _ = self.log_assignment[i](desc0, desc1)
        m0, m1, mscores0, mscores1 = filter_matches(
            scores, self.conf.filter_threshold)

        matches, mscores = [], []
        for k in range(b):
            valid = m0[k] > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[k][valid]
            if do_point_pruning:
                m_indices_0 = ind0[k, m_indices_0]
                m_indices_1 = ind1[k, m_indices_1]
            matches.append(torch.stack([m_indices_0, m_indices_1], -1))
            mscores.append(mscores0[k][valid])

        # TODO: Remove when hloc switches to the compact format.
        if do_point_pruning:
            m0_ = torch.full((b, m), -1, device=m0.device, dtype=m0.dtype)
            m1_ = torch.full((b, n), -1, device=m1.device, dtype=m1.dtype)
            m0_[:, ind0] = torch.where(
                m0 == -1, -1, ind1.gather(1, m0.clamp(min=0)))
            m1_[:, ind1] = torch.where(
                m1 == -1, -1, ind0.gather(1, m1.clamp(min=0)))
            mscores0_ = torch.zeros((b, m), device=mscores0.device)
            mscores1_ = torch.zeros((b, n), device=mscores1.device)
            mscores0_[:, ind0] = mscores0
            mscores1_[:, ind1] = mscores1
            m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_

        pred = {
            'matches0': m0,
            'matches1': m1,
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
            'stop': i+1,
            'matches': matches,
            'scores': mscores,
        }
        if do_point_pruning:
            pred.update(dict(prune0=prune0, prune1=prune1))
        return pred

    def confidence_threshold(self, layer_index: int) -> float:
        """ scaled confidence threshold """
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.conf.n_layers)
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(self, confidences: torch.Tensor, scores: torch.Tensor,
                         layer_index: int) -> torch.Tensor:
        """ mask points which should be removed """
        threshold = self.confidence_threshold(layer_index)
        if confidences is not None:
            scores = torch.where(
                confidences > threshold, scores, scores.new_tensor(1.0))
        return scores > (1 - self.conf.width_confidence)

    def check_if_stop(self,
                      confidences0: torch.Tensor,
                      confidences1: torch.Tensor,
                      layer_index: int, num_points: int) -> torch.Tensor:
        """ evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_threshold(layer_index)
        pos = 1.0 - (confidences < threshold).float().sum() / num_points
        return pos > self.conf.depth_confidence
