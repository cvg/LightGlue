"""Geometric verification module for feature matching.

This module provides RANSAC-based geometric verification for matched keypoints.
It can estimate fundamental matrix (F-matrix) or homography matrix (H-matrix)
and filter outliers based on epipolar or homography constraints.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch


def compute_fundamental_matrix(
    points0: np.ndarray,
    points1: np.ndarray,
    threshold: float = 1.0,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Compute fundamental matrix and inlier mask using RANSAC.

    Args:
        points0: Keypoints in image 0, shape (N, 2)
        points1: Keypoints in image 1, shape (N, 2)
        threshold: RANSAC reprojection error threshold in pixels

    Returns:
        F: Fundamental matrix of shape (3, 3), or None if computation fails
        inliers: Boolean array of inlier matches, shape (N,)
    """
    if len(points0) < 8:
        # Not enough points for F-matrix estimation
        return None, np.ones(len(points0), dtype=bool)

    try:
        F, mask = cv2.findFundamentalMat(
            points0, points1, cv2.FM_RANSAC, ransacReprojThreshold=threshold, confidence=0.99
        )
        if F is None:
            return None, np.ones(len(points0), dtype=bool)
        inliers = mask.ravel().astype(bool)
        return F, inliers
    except Exception:
        return None, np.ones(len(points0), dtype=bool)


def compute_homography_matrix(
    points0: np.ndarray,
    points1: np.ndarray,
    threshold: float = 4.0,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Compute homography matrix and inlier mask using RANSAC.

    This is useful when the scene is planar or when images are related
    by a perspective transformation.

    Args:
        points0: Keypoints in image 0, shape (N, 2)
        points1: Keypoints in image 1, shape (N, 2)
        threshold: RANSAC reprojection error threshold in pixels

    Returns:
        H: Homography matrix of shape (3, 3), or None if computation fails
        inliers: Boolean array of inlier matches, shape (N,)
    """
    if len(points0) < 4:
        # Not enough points for H-matrix estimation
        return None, np.ones(len(points0), dtype=bool)

    try:
        H, mask = cv2.findHomography(
            points0, points1, cv2.RANSAC, ransacReprojThreshold=threshold
        )
        if H is None:
            return None, np.ones(len(points0), dtype=bool)
        inliers = mask.ravel().astype(bool)
        return H, inliers
    except Exception:
        return None, np.ones(len(points0), dtype=bool)


def verify_matches(
    points0: np.ndarray,
    points1: np.ndarray,
    method: str = "fundamental",
    threshold: float = 1.0,
    min_inliers: int = 8,
) -> Tuple[np.ndarray, dict]:
    """Verify and filter matches using geometric constraints.

    Args:
        points0: Keypoints in image 0, shape (N, 2)
        points1: Keypoints in image 1, shape (N, 2)
        method: Verification method - 'fundamental' or 'homography'
        threshold: RANSAC reprojection error threshold
        min_inliers: Minimum number of inliers required

    Returns:
        inliers: Boolean array indicating inlier matches, shape (N,)
        stats: Dictionary with verification statistics including:
            - inlier_count: Number of inlier matches
            - inlier_ratio: Ratio of inliers to total matches
            - matrix: The estimated geometric matrix (F or H), or None
    """
    if method == "fundamental":
        matrix, inliers = compute_fundamental_matrix(points0, points1, threshold)
    elif method == "homography":
        matrix, inliers = compute_homography_matrix(points0, points1, threshold)
    else:
        raise ValueError(f"Unknown verification method: {method}")

    # Calculate statistics
    inlier_count = np.sum(inliers)
    inlier_ratio = inlier_count / len(inliers) if len(inliers) > 0 else 0.0

    # If not enough inliers, reject all matches
    if inlier_count < min_inliers:
        inliers = np.zeros(len(inliers), dtype=bool)

    stats = {
        "inlier_count": int(inlier_count),
        "inlier_ratio": float(inlier_ratio),
        "matrix": matrix,
        "method": method,
    }

    return inliers, stats


def filter_matches_by_geometry(
    matches: np.ndarray,
    points0: np.ndarray,
    points1: np.ndarray,
    method: str = "fundamental",
    threshold: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter matched keypoints using geometric verification.

    Args:
        matches: Match indices, shape (M, 2) where matches[i] = [idx0, idx1]
        points0: Keypoints in image 0, shape (N, 2)
        points1: Keypoints in image 1, shape (N, 2)
        method: Verification method - 'fundamental' or 'homography'
        threshold: RANSAC reprojection error threshold

    Returns:
        filtered_matches: Geometrically verified matches, shape (K, 2)
        stats: Dictionary with verification statistics
    """
    if len(matches) == 0:
        return matches, {"inlier_count": 0, "inlier_ratio": 0.0, "matrix": None}

    # Extract matched points
    matched_points0 = points0[matches[:, 0]]
    matched_points1 = points1[matches[:, 1]]

    # Verify matches
    inliers, stats = verify_matches(
        matched_points0, matched_points1, method=method, threshold=threshold
    )

    # Filter matches
    filtered_matches = matches[inliers]

    return filtered_matches, stats


def verify_matches_torch(
    points0: torch.Tensor,
    points1: torch.Tensor,
    method: str = "fundamental",
    threshold: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """PyTorch wrapper for verify_matches.

    Converts torch tensors to numpy, performs verification, and returns results.

    Args:
        points0: Keypoints in image 0, shape (N, 2)
        points1: Keypoints in image 1, shape (N, 2)
        method: Verification method - 'fundamental' or 'homography'
        threshold: RANSAC reprojection error threshold

    Returns:
        inliers: Boolean tensor indicating inlier matches, shape (N,)
        stats: Dictionary with verification statistics
    """
    points0_np = points0.detach().cpu().numpy()
    points1_np = points1.detach().cpu().numpy()

    inliers_np, stats = verify_matches(
        points0_np, points1_np, method=method, threshold=threshold
    )

    inliers = torch.tensor(inliers_np, dtype=torch.bool, device=points0.device)

    return inliers, stats
