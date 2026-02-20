"""Example script showing geometric verification of LightGlue matches.

This example demonstrates how to use the new geometric verification module
to filter outliers from feature matches using RANSAC.
"""

from lightglue import LightGlue, SuperPoint, verify_matches, filter_matches_by_geometry
from lightglue.utils import load_image, rbd
import torch
import numpy as np


def example_with_fundamental_matrix():
    """Example: Verify matches using Fundamental Matrix (F-matrix)."""
    print("Example 1: Geometric Verification with F-matrix (Epipolar Geometry)")
    print("-" * 70)
    
    # Initialize models
    extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
    matcher = LightGlue(features='superpoint').eval().cuda()
    
    # Load images
    image0 = load_image('path/to/image0.jpg').cuda()
    image1 = load_image('path/to/image1.jpg').cuda()
    
    # Extract features
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    
    # Match features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    
    matches = matches01['matches']  # shape (K, 2)
    points0 = feats0['keypoints'][matches[:, 0]].cpu().numpy()
    points1 = feats1['keypoints'][matches[:, 1]].cpu().numpy()
    
    print(f"Total matches: {len(matches)}")
    
    # Geometric verification with F-matrix
    filtered_matches, stats = filter_matches_by_geometry(
        matches.cpu().numpy(),
        points0,
        points1,
        method='fundamental',
        threshold=1.0
    )
    
    print(f"Inlier matches: {stats['inlier_count']}")
    print(f"Inlier ratio: {stats['inlier_ratio']:.2%}")
    print(f"Outliers removed: {len(matches) - stats['inlier_count']}")
    
    return filtered_matches, stats


def example_with_homography():
    """Example: Verify matches using Homography Matrix."""
    print("\nExample 2: Geometric Verification with Homography (Planar Scene)")
    print("-" * 70)
    
    # Initialize models
    extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
    matcher = LightGlue(features='superpoint').eval().cuda()
    
    # Load images (e.g., different viewpoints of a planar scene)
    image0 = load_image('path/to/planar_image0.jpg').cuda()
    image1 = load_image('path/to/planar_image1.jpg').cuda()
    
    # Extract and match features
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    
    matches = matches01['matches']
    points0 = feats0['keypoints'][matches[:, 0]].cpu().numpy()
    points1 = feats1['keypoints'][matches[:, 1]].cpu().numpy()
    
    print(f"Total matches: {len(matches)}")
    
    # Geometric verification with Homography
    filtered_matches, stats = filter_matches_by_geometry(
        matches.cpu().numpy(),
        points0,
        points1,
        method='homography',
        threshold=4.0
    )
    
    print(f"Inlier matches: {stats['inlier_count']}")
    print(f"Inlier ratio: {stats['inlier_ratio']:.2%}")
    print(f"Outliers removed: {len(matches) - stats['inlier_count']}")
    
    return filtered_matches, stats


def example_direct_verification():
    """Example: Direct verification of point pairs using numpy arrays."""
    print("\nExample 3: Direct Geometric Verification")
    print("-" * 70)
    
    # Synthetic example with randomly generated points
    np.random.seed(42)
    n_points = 100
    
    # Generate random inlier points with F-matrix constraint
    points0 = np.random.rand(n_points, 2) * 640
    # Generate corresponding points satisfying epipolar constraint
    points1 = points0 + np.random.randn(n_points, 2) * 2
    
    # Add some outliers
    outlier_idx = np.random.choice(n_points, 10, replace=False)
    points1[outlier_idx] += np.random.randn(10, 2) * 50
    
    print(f"Total points: {n_points}")
    print(f"Synthetic outliers added: 10")
    
    # Verify using F-matrix
    inliers, stats = verify_matches(
        points0,
        points1,
        method='fundamental',
        threshold=1.0,
        min_inliers=8
    )
    
    print(f"Detected inliers: {stats['inlier_count']}")
    print(f"Inlier ratio: {stats['inlier_ratio']:.2%}")


if __name__ == '__main__':
    print("=" * 70)
    print("LightGlue Geometric Verification Examples")
    print("=" * 70)
    
    # Uncomment the example you want to run:
    
    # example_with_fundamental_matrix()
    # example_with_homography()
    example_direct_verification()
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
