"""Download SAM2 model weights from Meta's repositories."""

import os
import urllib.request
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, destination: Path) -> None:
    """Download file with progress indicator.

    Args:
        url: URL to download from
        destination: Local path to save file
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {url}")
    logger.info(f"Saving to {destination}")

    def reporthook(blocknum, blocksize, totalsize):
        percent = min(100, blocknum * blocksize * 100 / totalsize)
        print(f"\rProgress: {percent:.1f}%", end="")

    urllib.request.urlretrieve(url, destination, reporthook)
    print()  # New line after progress
    logger.info(f"Downloaded {destination.name} ({destination.stat().st_size / 1024 / 1024:.1f} MB)")


def download_sam2_weights():
    """Download SAM2 model weights."""
    # SAM2 weights URLs (using SAM2.1 - latest version)
    models = {
        "sam2_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "sam2_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "sam2_hiera_small.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "sam2_hiera_tiny.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    }

    # Create models directory
    models_dir = Path("./data/models/base")
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SAM2 Model Download")
    logger.info("=" * 60)

    # Default: download the large model (best quality)
    model_name = "sam2_hiera_large.pt"
    url = models[model_name]
    destination = models_dir / model_name

    if destination.exists():
        logger.info(f"{model_name} already exists. Skipping download.")
        logger.info(f"Location: {destination}")
        return

    logger.info(f"Downloading {model_name}...")
    logger.info("This will download approximately 900MB. Please wait...")

    try:
        download_file(url, destination)
        logger.info("✅ SAM2 model downloaded successfully!")
        logger.info(f"Model saved to: {destination}")
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        logger.info("\nAlternative: Download manually from:")
        logger.info(f"  {url}")
        logger.info(f"  Save to: {destination}")
        raise


if __name__ == "__main__":
    download_sam2_weights()
