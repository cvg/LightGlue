
# Benchmark script for LightGlue on real images
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import numpy as np
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image

torch.set_grad_enabled(False)


def measure(matcher, data, device, r=100):
    timings = np.zeros((r, 1))
    if device.type == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
    # warmup
    for _ in range(10):
        _ = matcher(data)
    # measurements
    with torch.no_grad():
        for rep in range(r):
            if device.type == 'cuda':
                starter.record()
                _ = matcher(data)
                ender.record()
                # sync gpu
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
            else:
                start = time.perf_counter()
                _ = matcher(data)
                curr_time = (time.perf_counter() - start) * 1e3
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / r
    std_syn = np.std(timings)
    return {'mean': mean_syn, 'std': std_syn}


def print_as_table(d, title, cnames):
    print()
    header = f'{title:12} '+' '.join([f'{x:>7}' for x in cnames])
    print(header)
    print('-'*len(header))
    for k, l in d.items():
        print(f'{k:12}', ' '.join([f'{x:>7.1f}' for x in l]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark script for LightGlue')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu', 'mps'],
                        default='auto', help='device to benchmark on')
    parser.add_argument('--no_flash', action='store_true',
                        help='disable FlashAttention')
    parser.add_argument('--no_prune_thresholds', action='store_true',
                        help='disable pruning thresholds (i.e. always do pruning)')
    parser.add_argument('--add_superglue', action='store_true',
                        help='add SuperGlue to the benchmark (requires hloc)')
    parser.add_argument('--measure', default='throughput',
                        choices=['time', 'log-time', 'throughput'])
    parser.add_argument('--repeat', '--r', type=int, default=100,
                        help='repetitions of measurements')
    parser.add_argument('--num_keypoints', nargs="+", type=int,
                        default=[512, 1024, 2048, 4096],
                        help='number of keypoints (list separated by spaces)')
    args = parser.parse_intermixed_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.device != 'auto':
        device = torch.device(args.device)

    print('Running benchmark on device:', device)

    images = Path('assets')
    inputs = {
        'easy': (load_image(images / 'DSC_0411.JPG'),
                 load_image(images / 'DSC_0410.JPG')),
        'difficult': (load_image(images / 'sacre_coeur1.jpg'),
                      load_image(images / 'sacre_coeur2.jpg'))
    }

    configs = {
        'LG-full': {
            'depth_confidence': -1,
            'width_confidence': -1,
        },
        'LG-prune': {
            'depth_confidence': -1,
        },
        'LG-depth': {
            'width_confidence': -1,
        },
        'LG-adaptive': {}
    }

    sg_configs = {
        'SG': {},
        'SG-fast': {'sinkhorn_iterations': 5}
    }

    results = {k: defaultdict(list) for k, v in inputs.items()}

    extractor = SuperPoint(max_num_keypoints=None, detection_threshold=-1)
    extractor = extractor.eval().to(device)
    figsize = (len(inputs)*4.5, 4.5)
    fig, axes = plt.subplots(1, len(inputs), sharey=True, figsize=figsize)
    fig.canvas.manager.set_window_title(f'LightGlue benchmark ({device.type})')

    for title, ax in zip(inputs.keys(), axes):
        ax.set_xscale('log', base=2)
        ax.set_xticks(args.num_keypoints, args.num_keypoints)
        if args.measure == 'log-time':
            ax.set_yscale('log')
            yticks = [10**x for x in range(6)]
            ax.set_yticks(yticks, yticks)
        ax.grid()
        ax.set_title(title)

        ax.set_xlabel("# keypoints")
        if args.measure == 'throughput':
            ax.set_ylabel("Throughput [pairs/s]")
        else:
            ax.set_ylabel("Latency [ms]")

    for name, conf in configs.items():
        print('Run benchmark for:', name)
        torch.cuda.empty_cache()
        matcher = LightGlue(
            features='superpoint', flash=not args.no_flash, **conf)
        if args.no_prune_thresholds:
            matcher.pruning_keypoint_thresholds = {
                k: -1 for k in matcher.pruning_keypoint_thresholds}
        matcher = matcher.eval().to(device)
        for (pair_name, ax) in zip(inputs.keys(), axes):
            image0, image1 = [x.to(device) for x in inputs[pair_name]]
            runtimes = []
            for num_kpts in args.num_keypoints:
                extractor.conf['max_num_keypoints'] = num_kpts
                feats0 = extractor.extract(image0)
                feats1 = extractor.extract(image1)
                runtime = measure(matcher,
                                  {'image0': feats0, 'image1': feats1},
                                  device, r=args.repeat)['mean']
                results[pair_name][name].append(
                    1000/runtime if args.measure == 'throughput' else runtime)
            ax.plot(args.num_keypoints, results[pair_name][name], label=name,
                    marker='o')

    if args.add_superglue:
        from hloc.matchers.superglue import SuperGlue
        for name, conf in sg_configs.items():
            print('Run benchmark for:', name)
            torch.cuda.empty_cache()
            matcher = SuperGlue(conf)
            matcher = matcher.eval().to(device)
            for (pair_name, ax) in zip(inputs.keys(), axes):
                image0, image1 = [x.to(device) for x in inputs[pair_name]]
                runtimes = []
                for num_kpts in args.num_keypoints:
                    extractor.conf['max_num_keypoints'] = num_kpts
                    feats0 = extractor.extract(image0)
                    feats1 = extractor.extract(image1)
                    data = {
                        'image0': image0[None],
                        'image1': image1[None],
                        **{k+'0': v for k, v in feats0.items()},
                        **{k+'1': v for k, v in feats1.items()}
                    }
                    data['scores0'] = data['keypoint_scores0']
                    data['scores1'] = data['keypoint_scores1']
                    data['descriptors0'] = data['descriptors0'].transpose(-1, -2)
                    data['descriptors1'] = data['descriptors1'].transpose(-1, -2)
                    runtime = measure(matcher, data, device, r=args.repeat)['mean']
                    results[pair_name][name].append(
                        1000/runtime if args.measure == 'throughput' else runtime)
                ax.plot(args.num_keypoints, results[pair_name][name], label=name,
                        marker='o')

    for name, runtimes in results.items():
        print_as_table(runtimes, name, args.num_keypoints)

    axes[0].legend()
    fig.tight_layout()
    plt.show()
