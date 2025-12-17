"""Test script to verify SAM2 integration."""

import sys
from pathlib import Path
import numpy as np
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.segmentation_service import SegmentationService
from src.core.config import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_sam2_integration():
    """Test SAM2 model loading and inference."""
    logger.info("=" * 60)
    logger.info("Testing SAM2 Integration")
    logger.info("=" * 60)

    # Load configuration
    config = ConfigManager(config_path="config.yaml")
    logger.info(f"Checkpoint: {config.sam2.checkpoint}")
    logger.info(f"Config: {config.sam2.config}")
    logger.info(f"Device: {config.sam2.device}")

    # Initialize segmentation service
    logger.info("\n1. Initializing SegmentationService...")
    service = SegmentationService(
        checkpoint_path=config.sam2.checkpoint,
        config=config.sam2.config,
        device=config.sam2.device,
        confidence_threshold=config.sam2.confidence_threshold,
    )

    # Load model
    logger.info("\n2. Loading SAM2 model...")
    try:
        service.load_model()
        logger.info("✅ Model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

    # Create a test image (random RGB image)
    logger.info("\n3. Creating test image...")
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    logger.info(f"Test image shape: {test_image.shape}")

    # Run segmentation
    logger.info("\n4. Running segmentation...")
    try:
        result = service.segment_frame(test_image)
        logger.info("✅ Segmentation completed!")
        logger.info(f"   Detected instances: {len(result.masks)}")
        logger.info(f"   Masks shape: {result.masks.shape if len(result.masks) > 0 else 'N/A'}")
        logger.info(f"   Confidences: {result.confidences}")
        logger.info(f"   Labels: {result.class_labels}")
    except Exception as e:
        logger.error(f"❌ Segmentation failed: {e}")
        raise

    logger.info("\n" + "=" * 60)
    logger.info("✅ SAM2 Integration Test PASSED!")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_sam2_integration()
