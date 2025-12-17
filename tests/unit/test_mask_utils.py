"""Unit tests for mask utilities.

TDD: These tests are written FIRST and must FAIL before implementation.
"""

import pytest
import numpy as np
from pathlib import Path
from src.utils.mask_utils import (
    save_masks,
    load_masks,
    overlay_masks_on_image,
    masks_to_colored,
    compute_mask_iou,
    filter_masks_by_area,
    get_mask_bbox,
)


class TestMaskUtils:
    """Test mask storage, visualization, and manipulation utilities."""

    def test_save_masks_to_npz(self, tmp_path):
        """Test saving masks to .npz file."""
        masks = np.random.rand(5, 100, 100) > 0.5  # 5 binary masks
        confidences = np.array([0.9, 0.8, 0.7, 0.95, 0.85])
        save_path = tmp_path / "masks.npz"

        save_masks(masks, confidences, save_path)

        assert save_path.exists()
        assert save_path.suffix == ".npz"

    def test_load_masks_from_npz(self, tmp_path):
        """Test loading masks from .npz file."""
        original_masks = np.random.rand(3, 50, 50) > 0.5
        original_confidences = np.array([0.9, 0.8, 0.7])
        save_path = tmp_path / "masks.npz"

        save_masks(original_masks, original_confidences, save_path)
        loaded_masks, loaded_confidences = load_masks(save_path)

        assert loaded_masks.shape == original_masks.shape
        np.testing.assert_array_equal(loaded_masks, original_masks)
        np.testing.assert_array_almost_equal(loaded_confidences, original_confidences)

    def test_load_nonexistent_masks_raises_error(self):
        """Test loading non-existent mask file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_masks("/nonexistent/masks.npz")

    def test_overlay_masks_on_image(self):
        """Test overlaying colored masks on image."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White image
        masks = np.zeros((2, 100, 100), dtype=bool)
        masks[0, 10:30, 10:30] = True  # First mask
        masks[1, 50:70, 50:70] = True  # Second mask
        colors = [(255, 0, 0), (0, 255, 0)]  # Red and Green

        overlaid = overlay_masks_on_image(image, masks, colors, alpha=0.5)

        assert overlaid.shape == image.shape
        assert overlaid.dtype == np.uint8
        # Check that overlay modified the image
        assert not np.array_equal(overlaid, image)

    def test_masks_to_colored_generates_colors(self):
        """Test converting masks to colored visualization."""
        masks = np.zeros((3, 50, 50), dtype=bool)
        masks[0, 5:15, 5:15] = True
        masks[1, 20:30, 20:30] = True
        masks[2, 35:45, 35:45] = True

        colored = masks_to_colored(masks)

        assert colored.shape == (50, 50, 3)
        assert colored.dtype == np.uint8
        # Check that different masks have different colors
        color1 = tuple(colored[10, 10])
        color2 = tuple(colored[25, 25])
        assert color1 != color2

    def test_compute_mask_iou_perfect_overlap(self):
        """Test IoU computation with perfect overlap."""
        mask1 = np.zeros((50, 50), dtype=bool)
        mask1[10:30, 10:30] = True

        iou = compute_mask_iou(mask1, mask1)

        assert iou == pytest.approx(1.0)

    def test_compute_mask_iou_no_overlap(self):
        """Test IoU computation with no overlap."""
        mask1 = np.zeros((50, 50), dtype=bool)
        mask1[10:20, 10:20] = True

        mask2 = np.zeros((50, 50), dtype=bool)
        mask2[30:40, 30:40] = True

        iou = compute_mask_iou(mask1, mask2)

        assert iou == 0.0

    def test_compute_mask_iou_partial_overlap(self):
        """Test IoU computation with partial overlap."""
        mask1 = np.zeros((50, 50), dtype=bool)
        mask1[10:30, 10:30] = True

        mask2 = np.zeros((50, 50), dtype=bool)
        mask2[20:40, 20:40] = True

        iou = compute_mask_iou(mask1, mask2)

        assert 0 < iou < 1

    def test_filter_masks_by_area(self):
        """Test filtering masks by minimum area."""
        masks = np.zeros((3, 100, 100), dtype=bool)
        masks[0, 10:20, 10:20] = True  # Area: 100
        masks[1, 10:60, 10:60] = True  # Area: 2500
        masks[2, 10:15, 10:15] = True  # Area: 25

        filtered = filter_masks_by_area(masks, min_area=50)

        assert filtered.shape[0] == 2  # Only first two masks remain
        assert filtered[0].sum() == 100
        assert filtered[1].sum() == 2500

    def test_get_mask_bbox(self):
        """Test extracting bounding box from mask."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:60, 30:80] = True

        bbox = get_mask_bbox(mask)

        assert bbox == (20, 30, 60, 80)  # (y_min, x_min, y_max, x_max)

    def test_get_mask_bbox_empty_mask(self):
        """Test bounding box for empty mask returns None."""
        mask = np.zeros((100, 100), dtype=bool)

        bbox = get_mask_bbox(mask)

        assert bbox is None

    def test_masks_array_shape_validation(self):
        """Test validation of mask array shapes."""
        invalid_masks = np.random.rand(5, 100)  # 2D instead of 3D

        with pytest.raises(ValueError, match="Expected 3D array"):
            save_masks(invalid_masks, np.array([0.9] * 5), Path("test.npz"))
