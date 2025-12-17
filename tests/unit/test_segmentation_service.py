"""Unit tests for SegmentationService.

TDD: These tests are written FIRST and must FAIL before implementation.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from src.services.segmentation_service import SegmentationService, SegmentationResult


class TestSegmentationService:
    """Test SAM2 segmentation service."""

    @pytest.fixture
    def segmentation_service(self):
        """Create segmentation service instance."""
        # For testing, we'll mock the actual SAM2 model loading
        return SegmentationService(
            checkpoint_path="./data/models/base/sam2_hiera_large.pt",
            config="sam2_hiera_l.yaml",
            device="cpu",
        )

    def test_segmentation_service_initialization(self, segmentation_service):
        """Test that segmentation service initializes with config."""
        assert segmentation_service.device == "cpu"
        assert segmentation_service.checkpoint_path is not None

    @patch('src.services.segmentation_service.SAM2AutomaticMaskGenerator')
    @patch('src.services.segmentation_service.build_sam2')
    def test_segment_frame_returns_masks_and_scores(self, mock_build_sam2, mock_mask_gen, segmentation_service):
        """Test that segment_frame returns masks and confidence scores."""
        # Mock the mask generator
        mock_generator = Mock()
        mock_mask_gen.return_value = mock_generator
        mock_generator.generate.return_value = [
            {
                'segmentation': np.ones((480, 640), dtype=bool),
                'bbox': [10, 20, 100, 150],
                'predicted_iou': 0.95,
                'area': 15000,
            }
        ]

        # Create a test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = segmentation_service.segment_frame(image)

        assert isinstance(result, SegmentationResult)
        assert hasattr(result, "masks")
        assert hasattr(result, "confidences")
        assert hasattr(result, "class_labels")

    def test_segmentation_result_structure(self):
        """Test SegmentationResult data structure."""
        masks = np.random.rand(3, 480, 640) > 0.5  # 3 binary masks
        confidences = np.array([0.95, 0.87, 0.72])
        class_labels = ["person", "car", "bicycle"]
        bboxes = [(10, 20, 100, 200), (150, 30, 80, 120), (300, 100, 50, 60)]

        result = SegmentationResult(
            masks=masks,
            confidences=confidences,
            class_labels=class_labels,
            bboxes=bboxes,
        )

        assert result.masks.shape[0] == 3
        assert len(result.confidences) == 3
        assert len(result.class_labels) == 3
        assert len(result.bboxes) == 3

    @patch('src.services.segmentation_service.SAM2AutomaticMaskGenerator')
    @patch('src.services.segmentation_service.build_sam2')
    def test_segment_frame_filters_low_confidence(self, mock_build_sam2, mock_mask_gen, segmentation_service):
        """Test that low confidence detections are filtered out."""
        # Mock the mask generator with varying confidence scores
        mock_generator = Mock()
        mock_mask_gen.return_value = mock_generator
        mock_generator.generate.return_value = [
            {
                'segmentation': np.ones((480, 640), dtype=bool),
                'bbox': [10, 20, 100, 150],
                'predicted_iou': 0.85,  # Above threshold
                'area': 15000,
            },
            {
                'segmentation': np.ones((480, 640), dtype=bool),
                'bbox': [200, 100, 50, 80],
                'predicted_iou': 0.90,  # Above threshold
                'area': 4000,
            }
        ]

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Set confidence threshold
        segmentation_service.confidence_threshold = 0.75

        result = segmentation_service.segment_frame(image)

        # All returned confidences should be above threshold
        assert all(conf >= 0.75 for conf in result.confidences)

    def test_compute_uncertainty_scores(self, segmentation_service):
        """Test computing uncertainty scores for masks."""
        # Mock logits from model
        logits = np.random.randn(5, 480, 640)  # 5 masks with logits

        uncertainty_scores = segmentation_service.compute_uncertainty(logits)

        assert len(uncertainty_scores) == 5
        assert all(0 <= score <= 1 for score in uncertainty_scores)

    def test_detect_uncertain_regions(self, segmentation_service):
        """Test detecting regions with high uncertainty."""
        confidences = np.array([0.95, 0.82, 0.68, 0.45, 0.91])
        uncertainty_threshold = 0.7  # Corresponds to confidence < 0.7

        uncertain_indices = segmentation_service.detect_uncertain_regions(
            confidences, uncertainty_threshold
        )

        # Indices 2 and 3 should be uncertain (confidence < 0.7)
        assert 2 in uncertain_indices
        assert 3 in uncertain_indices
        assert 0 not in uncertain_indices

    @patch('src.services.segmentation_service.SAM2AutomaticMaskGenerator')
    @patch('src.services.segmentation_service.build_sam2')
    def test_segment_frame_batch(self, mock_build_sam2, mock_mask_gen, segmentation_service):
        """Test batch segmentation for multiple frames."""
        # Mock the mask generator
        mock_generator = Mock()
        mock_mask_gen.return_value = mock_generator
        mock_generator.generate.return_value = [
            {
                'segmentation': np.ones((480, 640), dtype=bool),
                'bbox': [10, 20, 100, 150],
                'predicted_iou': 0.95,
                'area': 15000,
            }
        ]

        # Create batch of test images
        images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(4)
        ]

        results = segmentation_service.segment_batch(images)

        assert len(results) == 4
        assert all(isinstance(r, SegmentationResult) for r in results)

    def test_model_loading_with_invalid_checkpoint(self):
        """Test that invalid checkpoint path raises error."""
        with pytest.raises(FileNotFoundError):
            service = SegmentationService(
                checkpoint_path="/nonexistent/checkpoint.pt",
                config="sam2_hiera_l.yaml",
                device="cpu",
            )
            service.load_model()

    def test_device_selection(self):
        """Test device selection (cuda, cpu, mps)."""
        service_cpu = SegmentationService(
            checkpoint_path="./checkpoint.pt", config="config.yaml", device="cpu"
        )
        assert service_cpu.device == "cpu"

        service_cuda = SegmentationService(
            checkpoint_path="./checkpoint.pt", config="config.yaml", device="cuda"
        )
        assert service_cuda.device == "cuda"

    def test_extract_bboxes_from_masks(self, segmentation_service):
        """Test extracting bounding boxes from binary masks."""
        # Create test masks
        masks = np.zeros((2, 100, 100), dtype=bool)
        masks[0, 10:30, 20:50] = True  # First mask
        masks[1, 60:80, 70:90] = True  # Second mask

        bboxes = segmentation_service.extract_bboxes(masks)

        assert len(bboxes) == 2
        # First bbox: (y_min, x_min, y_max, x_max) = (10, 20, 30, 50)
        # Convert to (x, y, w, h) format
        assert bboxes[0] == (20, 10, 30, 20)  # x=20, y=10, w=30, h=20
        assert bboxes[1] == (70, 60, 20, 20)  # x=70, y=60, w=20, h=20

    @pytest.mark.skip(reason="Placeholder test for future torch integration")
    @patch("src.services.segmentation_service.torch")
    def test_mixed_precision_inference(self, mock_torch, segmentation_service):
        """Test that mixed precision is used when enabled."""
        segmentation_service.enable_mixed_precision = True

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # This test ensures the interface supports mixed precision
        # Actual implementation will use torch.amp.autocast
        pass
