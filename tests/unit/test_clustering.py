"""Unit tests for clustering and similarity services.

TDD Red Phase: T046-T047 [US3] - Write failing tests FIRST before implementation

These tests validate similarity computation and DBSCAN clustering for
VLM_UNCERTAIN regions to identify patterns.
"""

import pytest
import numpy as np
from uuid import uuid4
from pathlib import Path
from datetime import datetime

from src.models.uncertain_region import UncertainRegion, RegionStatus


# T046 [P] [US3] - Test similarity computation


def test_compute_heuristic_similarity():
    """Test heuristic similarity computation with weighted components.

    TDD: T046 [US3] - This test should FAIL until implementation is complete

    Validates:
    - Similarity score combines bbox (0.3), mask (0.4), color (0.3) weights
    - Score ranges from 0.0 (dissimilar) to 1.0 (identical)
    - Individual component scores are computed correctly
    """
    from src.services.similarity_service import SimilarityService

    # ARRANGE
    similarity_service = SimilarityService()

    # Create two similar regions
    region1 = UncertainRegion(
        id=uuid4(),
        session_id=uuid4(),
        frame_id=uuid4(),
        frame_index=0,
        bbox=[100, 100, 50, 50],  # x, y, width, height
        uncertainty_score=0.6,
        cropped_image_path=Path("/tmp/crop1.jpg"),
        mask_path=Path("/tmp/mask1.npz"),
        status=RegionStatus.QUERIED,
        vlm_query_id=uuid4(),
        created_at=datetime.now(),
    )

    region2 = UncertainRegion(
        id=uuid4(),
        session_id=uuid4(),
        frame_id=uuid4(),
        frame_index=1,
        bbox=[105, 105, 52, 48],  # Similar bbox
        uncertainty_score=0.6,
        cropped_image_path=Path("/tmp/crop2.jpg"),
        mask_path=Path("/tmp/mask2.npz"),
        status=RegionStatus.QUERIED,
        vlm_query_id=uuid4(),
        created_at=datetime.now(),
    )

    # ACT: Compute similarity
    similarity = similarity_service.compute_similarity(region1, region2)

    # ASSERT: Similarity score computed correctly
    assert isinstance(similarity, dict), "Should return similarity dictionary"
    assert "overall" in similarity, "Should have overall similarity score"
    assert "bbox" in similarity, "Should have bbox component score"
    assert "mask" in similarity, "Should have mask component score"
    assert "color" in similarity, "Should have color component score"

    # Scores should be in valid range
    assert 0.0 <= similarity["overall"] <= 1.0, "Overall score should be in [0, 1]"
    assert 0.0 <= similarity["bbox"] <= 1.0, "Bbox score should be in [0, 1]"
    assert 0.0 <= similarity["mask"] <= 1.0, "Mask score should be in [0, 1]"
    assert 0.0 <= similarity["color"] <= 1.0, "Color score should be in [0, 1]"

    # For similar bboxes, similarity should be relatively high
    assert similarity["bbox"] > 0.8, f"Similar bboxes should have high similarity, got {similarity['bbox']:.3f}"


def test_compute_bbox_similarity():
    """Test bbox similarity using area and aspect ratio.

    TDD: T046 [US3] - This test should FAIL until implementation is complete
    """
    from src.services.similarity_service import SimilarityService

    # ARRANGE
    similarity_service = SimilarityService()

    # Identical bboxes
    bbox1 = [100, 100, 50, 50]
    bbox2 = [100, 100, 50, 50]

    # ACT & ASSERT: Identical bboxes
    sim_identical = similarity_service._compute_bbox_similarity(bbox1, bbox2)
    assert sim_identical == 1.0, "Identical bboxes should have similarity 1.0"

    # Similar size and shape
    bbox3 = [110, 110, 52, 48]
    sim_similar = similarity_service._compute_bbox_similarity(bbox1, bbox3)
    assert 0.8 <= sim_similar < 1.0, f"Similar bboxes should have high similarity, got {sim_similar:.3f}"

    # Different aspect ratio
    bbox4 = [100, 100, 80, 30]  # Wide rectangle
    sim_different = similarity_service._compute_bbox_similarity(bbox1, bbox4)
    assert sim_different < 0.7, f"Different aspect ratios should have lower similarity, got {sim_different:.3f}"


def test_compute_mask_similarity_iou():
    """Test mask similarity using IoU after resizing.

    TDD: T046 [US3] - This test should FAIL until implementation is complete
    """
    from src.services.similarity_service import SimilarityService

    # ARRANGE
    similarity_service = SimilarityService()

    # Create identical binary masks (128x128)
    mask1 = np.zeros((128, 128), dtype=np.uint8)
    mask1[40:80, 40:80] = 1  # Square mask

    mask2 = np.zeros((128, 128), dtype=np.uint8)
    mask2[40:80, 40:80] = 1  # Identical square mask

    # ACT & ASSERT: Identical masks
    sim_identical = similarity_service._compute_mask_iou(mask1, mask2)
    assert sim_identical == 1.0, "Identical masks should have IoU 1.0"

    # Overlapping masks
    mask3 = np.zeros((128, 128), dtype=np.uint8)
    mask3[45:85, 45:85] = 1  # Overlapping square

    sim_overlap = similarity_service._compute_mask_iou(mask1, mask3)
    assert 0.5 < sim_overlap < 1.0, f"Overlapping masks should have moderate IoU, got {sim_overlap:.3f}"

    # Non-overlapping masks
    mask4 = np.zeros((128, 128), dtype=np.uint8)
    mask4[90:110, 90:110] = 1  # Non-overlapping square

    sim_no_overlap = similarity_service._compute_mask_iou(mask1, mask4)
    assert sim_no_overlap == 0.0, "Non-overlapping masks should have IoU 0.0"


def test_compute_color_histogram_similarity():
    """Test color histogram similarity using RGB correlation.

    TDD: T046 [US3] - This test should FAIL until implementation is complete
    """
    from src.services.similarity_service import SimilarityService

    # ARRANGE
    similarity_service = SimilarityService()

    # Create identical color histograms (32 bins/channel)
    hist1 = np.random.rand(32, 3).astype(np.float32)
    hist2 = hist1.copy()

    # ACT & ASSERT: Identical histograms
    sim_identical = similarity_service._compute_color_similarity(hist1, hist2)
    assert abs(sim_identical - 1.0) < 0.01, "Identical histograms should have similarity ~1.0"

    # Similar histograms
    hist3 = hist1 + np.random.randn(32, 3) * 0.1  # Add small noise
    hist3 = np.clip(hist3, 0, 1)

    sim_similar = similarity_service._compute_color_similarity(hist1, hist3)
    assert 0.7 < sim_similar < 1.0, f"Similar histograms should have high correlation, got {sim_similar:.3f}"

    # Different histograms
    hist4 = np.random.rand(32, 3).astype(np.float32)
    sim_different = similarity_service._compute_color_similarity(hist1, hist4)
    assert sim_different < 0.7, f"Different histograms should have lower similarity, got {sim_different:.3f}"


# T047 [P] [US3] - Test DBSCAN clustering


def test_dbscan_clusters_similar_regions():
    """Test DBSCAN clustering groups similar VLM_UNCERTAIN regions.

    TDD: T047 [US3] - This test should FAIL until implementation is complete

    Validates:
    - DBSCAN with eps=0.3, min_samples=2 groups similar regions
    - Noise points (label=-1) for dissimilar regions
    - Cluster labels assigned correctly
    """
    from src.services.clustering_service import ClusteringService
    from src.services.similarity_service import SimilarityService

    # ARRANGE
    clustering_service = ClusteringService()
    similarity_service = SimilarityService()

    # Create 6 regions: 2 clusters of 3 similar regions each
    regions = []

    # Cluster 1: Three similar small boxes at top-left
    for i in range(3):
        regions.append(UncertainRegion(
            id=uuid4(),
            session_id=uuid4(),
            frame_id=uuid4(),
            frame_index=i,
            bbox=[100 + i * 2, 100 + i * 2, 50, 50],
            uncertainty_score=0.6,
            cropped_image_path=Path(f"/tmp/cluster1_crop{i}.jpg"),
            mask_path=Path(f"/tmp/cluster1_mask{i}.npz"),
            status=RegionStatus.QUERIED,
            vlm_query_id=uuid4(),
            created_at=datetime.now(),
        ))

    # Cluster 2: Three similar large boxes at bottom-right
    for i in range(3):
        regions.append(UncertainRegion(
            id=uuid4(),
            session_id=uuid4(),
            frame_id=uuid4(),
            frame_index=i + 3,
            bbox=[300 + i * 2, 300 + i * 2, 80, 80],
            uncertainty_score=0.6,
            cropped_image_path=Path(f"/tmp/cluster2_crop{i}.jpg"),
            mask_path=Path(f"/tmp/cluster2_mask{i}.npz"),
            status=RegionStatus.QUERIED,
            vlm_query_id=uuid4(),
            created_at=datetime.now(),
        ))

    # ACT: Cluster regions using DBSCAN
    clusters = clustering_service.cluster_regions(regions, eps=0.3, min_samples=2)

    # ASSERT: Should identify 2 clusters
    assert "labels" in clusters, "Should return cluster labels"
    assert "n_clusters" in clusters, "Should return number of clusters"
    assert "noise_count" in clusters, "Should return noise count"

    labels = clusters["labels"]
    n_clusters = clusters["n_clusters"]

    assert len(labels) == 6, "Should have labels for all 6 regions"
    assert n_clusters == 2, f"Should identify 2 clusters, got {n_clusters}"

    # Verify cluster assignments
    # Regions 0, 1, 2 should be in same cluster
    cluster1_label = labels[0]
    assert cluster1_label >= 0, "First region should be in a cluster (not noise)"
    assert labels[1] == cluster1_label, "Regions 0 and 1 should be in same cluster"
    assert labels[2] == cluster1_label, "Regions 0 and 2 should be in same cluster"

    # Regions 3, 4, 5 should be in same cluster (different from cluster 1)
    cluster2_label = labels[3]
    assert cluster2_label >= 0, "Fourth region should be in a cluster (not noise)"
    assert cluster2_label != cluster1_label, "Clusters 1 and 2 should be different"
    assert labels[4] == cluster2_label, "Regions 3 and 4 should be in same cluster"
    assert labels[5] == cluster2_label, "Regions 3 and 5 should be in same cluster"


def test_dbscan_handles_noise_points():
    """Test DBSCAN marks dissimilar regions as noise.

    TDD: T047 [US3] - This test should FAIL until implementation is complete
    """
    from src.services.clustering_service import ClusteringService

    # ARRANGE
    clustering_service = ClusteringService()

    # Create 5 regions: 1 cluster of 3 + 2 noise points
    regions = []

    # Cluster: Three similar regions
    for i in range(3):
        regions.append(UncertainRegion(
            id=uuid4(),
            session_id=uuid4(),
            frame_id=uuid4(),
            frame_index=i,
            bbox=[100 + i * 2, 100 + i * 2, 50, 50],
            uncertainty_score=0.6,
            cropped_image_path=Path(f"/tmp/cluster_crop{i}.jpg"),
            mask_path=Path(f"/tmp/cluster_mask{i}.npz"),
            status=RegionStatus.QUERIED,
            vlm_query_id=uuid4(),
            created_at=datetime.now(),
        ))

    # Noise: Two dissimilar regions
    regions.append(UncertainRegion(
        id=uuid4(),
        session_id=uuid4(),
        frame_id=uuid4(),
        frame_index=3,
        bbox=[500, 500, 30, 30],  # Far away, different size
        uncertainty_score=0.6,
        cropped_image_path=Path("/tmp/noise1_crop.jpg"),
        mask_path=Path("/tmp/noise1.npz"),
        status=RegionStatus.QUERIED,
        vlm_query_id=uuid4(),
        created_at=datetime.now(),
    ))

    regions.append(UncertainRegion(
        id=uuid4(),
        session_id=uuid4(),
        frame_id=uuid4(),
        frame_index=4,
        bbox=[700, 700, 150, 20],  # Far away, extreme aspect ratio
        uncertainty_score=0.6,
        cropped_image_path=Path("/tmp/noise2_crop.jpg"),
        mask_path=Path("/tmp/noise2.npz"),
        status=RegionStatus.QUERIED,
        vlm_query_id=uuid4(),
        created_at=datetime.now(),
    ))

    # ACT: Cluster with min_samples=2
    clusters = clustering_service.cluster_regions(regions, eps=0.3, min_samples=2)

    # ASSERT: Should identify 1 cluster + 2 noise points
    labels = clusters["labels"]
    n_clusters = clusters["n_clusters"]
    noise_count = clusters["noise_count"]

    assert n_clusters == 1, f"Should identify 1 cluster, got {n_clusters}"
    assert noise_count == 2, f"Should identify 2 noise points, got {noise_count}"

    # First 3 regions should be clustered
    cluster_label = labels[0]
    assert cluster_label >= 0, "First region should be in a cluster"
    assert labels[1] == cluster_label, "Regions 0 and 1 should be in same cluster"
    assert labels[2] == cluster_label, "Regions 0 and 2 should be in same cluster"

    # Last 2 regions should be noise
    assert labels[3] == -1, "Fourth region should be noise"
    assert labels[4] == -1, "Fifth region should be noise"


def test_cluster_to_pattern_conversion():
    """Test converting DBSCAN clusters to SemanticUncertaintyPattern objects.

    TDD: T047 [US3] - This test should FAIL until implementation is complete
    """
    from src.services.clustering_service import ClusteringService
    from src.models.semantic_uncertainty import SemanticUncertaintyPattern

    # ARRANGE
    clustering_service = ClusteringService()
    job_id = uuid4()

    # Create 4 regions in 2 clusters
    regions = []
    for i in range(4):
        regions.append(UncertainRegion(
            id=uuid4(),
            session_id=uuid4(),
            frame_id=uuid4(),
            frame_index=i,
            bbox=[100 + (i % 2) * 200, 100, 50, 50],  # 2 locations
            uncertainty_score=0.6,
            cropped_image_path=Path(f"/tmp/crop{i}.jpg"),
            mask_path=Path(f"/tmp/mask{i}.npz"),
            status=RegionStatus.QUERIED,
            vlm_query_id=uuid4(),
            created_at=datetime.now(),
        ))

    # Mock cluster labels: [0, 0, 1, 1]
    cluster_labels = [0, 0, 1, 1]

    # ACT: Convert to patterns
    patterns = clustering_service.clusters_to_patterns(job_id, regions, cluster_labels)

    # ASSERT: Should create 2 patterns
    assert len(patterns) == 2, f"Should create 2 patterns, got {len(patterns)}"

    for pattern in patterns:
        assert isinstance(pattern, SemanticUncertaintyPattern), "Should return Pattern objects"
        assert pattern.job_id == job_id, "Pattern should reference correct job"
        assert pattern.region_count == 2, "Each pattern should have 2 regions"
        assert len(pattern.region_ids) == 2, "Each pattern should track 2 region IDs"
        assert len(pattern.frames_affected) == 2, "Each pattern should span 2 frames"
        assert len(pattern.sample_image_paths) <= 5, "Should limit sample images to 5"
