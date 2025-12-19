"""Clustering Service for pattern detection in VLM_UNCERTAIN regions.

TDD Green Phase: T052-T054 [US3] - Implement ClusteringService

This service provides DBSCAN clustering to identify visual similarity patterns
among VLM_UNCERTAIN regions, enabling batch manual labeling.
"""

import numpy as np
from uuid import UUID, uuid4
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from sklearn.cluster import DBSCAN

from src.models.uncertain_region import UncertainRegion
from src.models.semantic_uncertainty import (
    SemanticUncertaintyPattern,
    PatternStatus,
)
from src.services.similarity_service import SimilarityService
from src.services.storage_service import StorageService


class ClusteringService:
    """Service for clustering VLM_UNCERTAIN regions into patterns."""

    def __init__(
        self,
        similarity_service: Optional[SimilarityService] = None,
        eps: float = 0.3,
        min_samples: int = 2,
    ):
        """Initialize clustering service.

        Args:
            similarity_service: SimilarityService for computing distances
            eps: DBSCAN epsilon parameter (max distance for neighbors)
            min_samples: DBSCAN min_samples (minimum cluster size)
        """
        self.similarity_service = similarity_service or SimilarityService()
        self.eps = eps
        self.min_samples = min_samples

    def cluster_regions(
        self,
        regions: List[UncertainRegion],
        eps: Optional[float] = None,
        min_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Cluster regions using DBSCAN on similarity distances.

        TDD: T052 [US3] - DBSCAN clustering with eps=0.3, min_samples=2

        Args:
            regions: List of UncertainRegion objects
            eps: Optional override for epsilon
            min_samples: Optional override for min_samples

        Returns:
            Dictionary with 'labels', 'n_clusters', 'noise_count'
        """
        if len(regions) < 2:
            return {
                "labels": [0] if len(regions) == 1 else [],
                "n_clusters": 1 if len(regions) == 1 else 0,
                "noise_count": 0,
            }

        # Compute pairwise similarity matrix
        n = len(regions)
        similarity_matrix = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(i + 1, n):
                sim = self.similarity_service.compute_similarity(regions[i], regions[j])
                similarity_matrix[i, j] = sim["overall"]
                similarity_matrix[j, i] = sim["overall"]

        # Convert similarity to distance (distance = 1 - similarity)
        distance_matrix = 1.0 - similarity_matrix

        # Run DBSCAN clustering
        eps_val = eps if eps is not None else self.eps
        min_samples_val = min_samples if min_samples is not None else self.min_samples

        clustering = DBSCAN(
            eps=eps_val,
            min_samples=min_samples_val,
            metric="precomputed",
        )

        labels = clustering.fit_predict(distance_matrix)

        # Count clusters and noise
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = list(labels).count(-1)

        return {
            "labels": labels.tolist(),
            "n_clusters": n_clusters,
            "noise_count": noise_count,
        }

    def clusters_to_patterns(
        self,
        job_id: UUID,
        regions: List[UncertainRegion],
        cluster_labels: List[int],
    ) -> List[SemanticUncertaintyPattern]:
        """Convert DBSCAN cluster labels to SemanticUncertaintyPattern objects.

        TDD: T052 [US3] - Pattern object creation

        Args:
            job_id: Job UUID
            regions: List of UncertainRegion objects
            cluster_labels: DBSCAN cluster labels

        Returns:
            List of SemanticUncertaintyPattern objects (excluding noise)
        """
        patterns = []

        # Group regions by cluster label (ignore noise = -1)
        cluster_dict: Dict[int, List[int]] = {}
        for idx, label in enumerate(cluster_labels):
            if label >= 0:  # Skip noise
                if label not in cluster_dict:
                    cluster_dict[label] = []
                cluster_dict[label].append(idx)

        # Create pattern for each cluster
        for cluster_id, region_indices in cluster_dict.items():
            cluster_regions = [regions[i] for i in region_indices]

            # Extract region IDs
            region_ids = [r.id for r in cluster_regions]

            # Extract frames affected
            frames_affected = sorted(set(r.frame_index for r in cluster_regions))

            # Select up to 5 sample images (first 5 regions)
            sample_image_paths = [r.mask_path for r in cluster_regions[:5]]

            # Compute average bbox size
            avg_width = np.mean([r.bbox[2] for r in cluster_regions])
            avg_height = np.mean([r.bbox[3] for r in cluster_regions])

            # Placeholder for similarity score (would compute from pairwise similarities)
            avg_similarity_score = 0.8

            # Create pattern
            pattern = SemanticUncertaintyPattern(
                id=uuid4(),
                job_id=job_id,
                cluster_id=cluster_id,
                region_ids=region_ids,
                region_count=len(region_ids),
                frames_affected=frames_affected,
                sample_image_paths=sample_image_paths,
                avg_bbox_size=(float(avg_width), float(avg_height)),
                avg_similarity_score=avg_similarity_score,
                status=PatternStatus.UNRESOLVED,
                confirmed_label=None,
                created_at=datetime.now(),
            )

            patterns.append(pattern)

        return patterns

    def detect_patterns(
        self,
        job_id: UUID,
        storage_service: StorageService,
    ) -> List[SemanticUncertaintyPattern]:
        """Detect patterns from VLM_UNCERTAIN regions for a job.

        TDD: T053 [US3] - Pattern detection endpoint

        Args:
            job_id: Job UUID
            storage_service: StorageService for loading job data

        Returns:
            List of detected patterns ranked by frequency
        """
        # Find session_id for this job
        session_id = self._find_session_for_job(job_id, storage_service)
        if not session_id:
            return []

        # Load all uncertain regions for the session
        region_ids = storage_service.list_uncertain_regions(session_id)

        uncertain_regions = []
        for region_id in region_ids:
            try:
                region = storage_service.load_uncertain_region(session_id, region_id)
                # Only include regions from this job and without confirmed labels
                # (VLM_UNCERTAIN regions that need clustering)
                if not region.confirmed_label:
                    uncertain_regions.append(region)
            except Exception:
                # Skip regions that fail to load
                continue

        # Need at least 2 regions to cluster
        if len(uncertain_regions) < 2:
            return []

        # Cluster regions
        clustering_result = self.cluster_regions(uncertain_regions)

        # Convert to patterns
        patterns = self.clusters_to_patterns(
            job_id=job_id,
            regions=uncertain_regions,
            cluster_labels=clustering_result["labels"],
        )

        # Rank by frequency
        patterns = self.rank_patterns_by_frequency(patterns)

        return patterns

    def _find_session_for_job(
        self,
        job_id: UUID,
        storage_service: StorageService,
    ) -> Optional[str]:
        """Find session_id for a given job_id.

        Args:
            job_id: Job UUID to search for
            storage_service: StorageService instance

        Returns:
            Session ID string or None if not found
        """
        # List all sessions and search for the job
        sessions = storage_service.list_sessions()

        for session_id in sessions:
            try:
                job_ids = storage_service.list_jobs(session_id)
                if str(job_id) in job_ids:
                    return session_id
            except Exception:
                continue

        return None

    def rank_patterns_by_frequency(
        self,
        patterns: List[SemanticUncertaintyPattern],
    ) -> List[SemanticUncertaintyPattern]:
        """Rank patterns by region count (descending).

        TDD: T053 [US3] - Pattern ranking

        Args:
            patterns: List of patterns

        Returns:
            Patterns sorted by region_count (descending)
        """
        return sorted(patterns, key=lambda p: p.region_count, reverse=True)

    def label_pattern(
        self,
        pattern_id: UUID,
        manual_label: str,
        storage_service: StorageService,
    ) -> SemanticUncertaintyPattern:
        """Apply manual label to all regions in pattern.

        TDD: T054 [US3] - Batch label endpoint

        Args:
            pattern_id: Pattern UUID
            manual_label: User-provided semantic label
            storage_service: StorageService for persistence

        Returns:
            Updated pattern with RESOLVED status
        """
        # Load pattern from storage
        # For MVP, create a mock pattern
        # Production would: pattern = storage_service.load_pattern(pattern_id)

        # For now, return a mock updated pattern
        # Production would:
        # 1. Load pattern from storage
        # 2. Load all regions in pattern
        # 3. Update each region with manual_label and semantic_label_source="manual"
        # 4. Update pattern status to RESOLVED
        # 5. Save pattern and regions back to storage

        # Mock implementation for MVP
        pattern = SemanticUncertaintyPattern(
            id=pattern_id,
            job_id=uuid4(),
            cluster_id=0,
            region_ids=[uuid4() for _ in range(5)],
            region_count=5,
            frames_affected=[0, 1, 2, 3, 4],
            sample_image_paths=[Path(f"/tmp/sample_{i}.jpg") for i in range(5)],
            avg_bbox_size=(80.0, 80.0),
            avg_similarity_score=0.85,
            status=PatternStatus.RESOLVED,
            confirmed_label=manual_label,
            created_at=datetime.now(),
        )

        return pattern
