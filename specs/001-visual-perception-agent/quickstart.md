# Quickstart Guide: Self-Improving Visual Perception Agent

**Purpose**: Get the visual perception agent running quickly for development and testing

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 4080, A100, or similar)
  - CUDA 11.8 or later
  - Alternatively: Apple Silicon Mac with 32GB+ unified memory (MPS support)
- **CPU**: 8+ cores recommended
- **RAM**: 32GB+ system RAM
- **Storage**: 100GB+ free space for models and data

### Software Requirements
- Python 3.11 or later
- pip or conda for package management
- Git for version control
- FFmpeg for video processing

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd inquis
git checkout 001-visual-perception-agent
```

### 2. Create Virtual Environment

```bash
# Using venv
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using conda
conda create -n vision-agent python=3.11
conda activate vision-agent
```

### 3. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install -r requirements.txt
```

**requirements.txt** (minimal):
```
torch>=2.0.0
torchvision>=0.15.0
streamlit>=1.28.0
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
pydantic>=2.0.0
openai>=1.0.0
diffusers>=0.24.0
transformers>=4.35.0
accelerate>=0.24.0
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.12.0
pyyaml>=6.0.0
```

### 4. Download Model Weights

```bash
# Create models directory
mkdir -p data/models/base

# Download SAM2 weights
python scripts/download_sam2.py

# Download Z-Image weights
python scripts/download_zimage.py
```

**scripts/download_sam2.py**:
```python
from sam2.build_sam import download_checkpoint

# Download SAM2 Hiera-Large checkpoint
checkpoint_path = download_checkpoint("sam2_hiera_large")
print(f"SAM2 checkpoint downloaded to: {checkpoint_path}")
```

**scripts/download_zimage.py**:
```python
from diffusers import DiffusionPipeline

# Download Z-Image-Turbo
pipeline = DiffusionPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    cache_dir="./data/models/base/zimage"
)
print("Z-Image model downloaded successfully")
```

### 5. Configure API Keys

```bash
# Create .env file
cat > .env << 'EOF'
# OpenAI API for GPT-5.2
OPENAI_API_KEY=sk-your-key-here

# Optional: Weights & Biases for experiment tracking
WANDB_API_KEY=your-wandb-key

# Data paths
DATA_DIR=./data
MODELS_DIR=./data/models
EOF

# Load environment variables
source .env  # Or use python-dotenv in code
```

## Quick Test

### Run Unit Tests

```bash
# Run fast unit tests
pytest tests/unit -v

# Run with coverage
pytest tests/unit --cov=src --cov-report=html
```

### Test Individual Components

```python
# Test SAM2 segmentation
python tests/manual/test_sam2_inference.py

# Test VLM labeling
python tests/manual/test_vlm_query.py

# Test Z-Image generation
python tests/manual/test_zimage_generation.py
```

## Running the Application

### Launch Streamlit GUI

```bash
streamlit run src/gui/app.py
```

This will start the web interface at `http://localhost:8501`

### Command-Line Usage

For headless processing without GUI:

```python
from src.core.pipeline import PerceptionPipeline
from pathlib import Path

# Initialize pipeline
pipeline = PerceptionPipeline(
    sam2_checkpoint="./data/models/base/sam2_hiera_large.pt",
    zimage_model="./data/models/base/zimage",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Process video
video_path = Path("test_video.mp4")
session = pipeline.process_video(video_path)

# View results
print(f"Processed {session.metadata['frame_count']} frames")
print(f"Found {len(session.uncertain_regions)} uncertain regions")
```

## Basic Workflow

### 1. Upload and Segment Video

1. Open Streamlit interface (`http://localhost:8501`)
2. Click "Upload Video" and select MP4/AVI/MOV file
3. Click "Start Processing"
4. View real-time segmentation results

### 2. Review Uncertain Regions

1. Navigate to "Uncertainty" tab
2. Browse highlighted uncertain regions
3. Click "Query VLM" to get suggested labels
4. Review VLM reasoning and accept/modify labels

### 3. Generate Synthetic Data

1. Navigate to "Synthesis" tab
2. Select labeled regions to generate from
3. Configure generation parameters (number of images, prompts)
4. Click "Generate"
5. Review generated images and quality scores
6. Accept/reject images for training

### 4. Train Model

1. Navigate to "Training" tab
2. Review training dataset composition
3. Configure training parameters (epochs, learning rate, LoRA rank)
4. Click "Start Training"
5. Monitor training progress and metrics

### 5. Evaluate Improvement

1. Navigate to "Metrics" tab
2. View performance comparison across model versions
3. Check uncertainty reduction on original video
4. Export metrics for analysis

## Directory Structure After Setup

```
inquis/
├── .venv/                      # Virtual environment
├── .env                        # Environment variables (API keys)
├── data/
│   ├── models/
│   │   ├── base/               # Base model checkpoints
│   │   │   ├── sam2_hiera_large.pt
│   │   │   └── zimage/
│   │   └── versions/           # Fine-tuned model versions
│   ├── sessions/               # Processed video sessions
│   └── datasets/               # Training datasets
├── src/
│   ├── models/                 # Data models
│   ├── services/               # Core services
│   ├── gui/                    # Streamlit interface
│   ├── core/                   # Pipeline orchestration
│   └── utils/                  # Utilities
├── tests/
│   ├── unit/
│   ├── integration/
│   └── contract/
├── specs/                      # Feature specifications
└── requirements.txt
```

## Configuration

### config.yaml

```yaml
# SAM2 Configuration
sam2:
  checkpoint: ./data/models/base/sam2_hiera_large.pt
  config: sam2_hiera_l.yaml
  device: cuda
  confidence_threshold: 0.75
  uncertainty_threshold: 0.25

# Z-Image Configuration
zimage:
  model_path: ./data/models/base/zimage
  model_variant: turbo  # turbo | base | edit
  device: cuda
  inference_steps: 8
  quality_threshold: 0.7

# VLM Configuration
vlm:
  model: gpt-5.2
  max_tokens: 300
  temperature: 0.1
  retry_max_attempts: 3
  retry_backoff_factor: 2

# Training Configuration
training:
  batch_size: 4
  learning_rate: 0.0001
  epochs: 5
  lora_rank: 8
  validation_split: 0.1

# Storage Configuration
storage:
  data_dir: ./data
  max_session_size_gb: 50
  cleanup_after_days: 30

# Performance Configuration
performance:
  batch_process_frames: 4
  max_concurrent_vlm_queries: 5
  enable_mixed_precision: true
  cache_embeddings: true
```

## Troubleshooting

### GPU Out of Memory

**Symptoms**: CUDA OOM errors during processing

**Solutions**:
- Reduce `batch_process_frames` in config
- Use `sam2_hiera_small` instead of `large`
- Enable mixed precision (fp16): `enable_mixed_precision: true`
- Process video at lower resolution

### VLM Rate Limits

**Symptoms**: 429 errors from OpenAI API

**Solutions**:
- Reduce `max_concurrent_vlm_queries` in config
- Add delays between batches
- Check API tier limits
- Use caching for duplicate queries

### Slow Video Processing

**Symptoms**: Processing time >> 2x video duration

**Solutions**:
- Check GPU utilization (`nvidia-smi`)
- Increase `batch_process_frames`
- Use Z-Image-Turbo (8 steps) instead of Base (50 steps)
- Preload models at startup
- Enable embedding cache

### Model Quality Issues

**Symptoms**: Poor segmentation or high uncertainty

**Solutions**:
- Lower `confidence_threshold` to be more permissive
- Increase `uncertainty_threshold` to flag fewer regions
- Use larger SAM2 variant (`hiera_large`)
- Check input video quality (resolution, lighting)

## Next Steps

1. **Read Documentation**:
   - [spec.md](./spec.md) - Feature specification
   - [research.md](./research.md) - Technology deep-dive
   - [data-model.md](./data-model.md) - Entity definitions

2. **Run Tests**:
   ```bash
   pytest tests/ -v --cov=src
   ```

3. **Try Sample Videos**:
   - Download test videos from `tests/fixtures/videos/`
   - Process and review results

4. **Customize**:
   - Modify prompts for VLM queries
   - Adjust generation prompts for synthetic data
   - Fine-tune training parameters

5. **Deploy**:
   - See deployment guide for Docker containerization
   - Configure for cloud GPU (AWS, GCP, Azure)

## Support

- **Issues**: Report bugs on GitHub Issues
- **Documentation**: See `specs/` directory
- **Tests**: Run `pytest tests/` for examples

## Performance Benchmarks

Expected performance on RTX 4080 (16GB):

| Operation | Time | Notes |
|-----------|------|-------|
| Video upload | < 5s | For 10-minute 1080p video |
| Frame segmentation | 0.3-0.5s/frame | With batch size 4 |
| Uncertainty detection | < 0.1s/frame | Lightweight computation |
| VLM query | 1-2s/query | Depends on API latency |
| Synthetic generation | 10-15s/batch | 20 images per batch |
| Model training | 5-10 min/epoch | With LoRA, 500 images |

**Total time for complete cycle**: ~30-40 minutes for 10-minute video with 50 uncertain regions.

## Development Workflow

### Adding New Features

1. Write failing test (TDD)
2. Implement minimal code to pass
3. Refactor if needed
4. Run all tests
5. Commit with appropriate prefix (`feat:`, `fix:`, `refactor:`)

### Code Quality

```bash
# Linting
ruff check src/

# Type checking
mypy src/

# Format code
black src/

# Run all checks
pre-commit run --all-files
```

## Example Session

```python
# Complete example: Process video through full pipeline
import os
from pathlib import Path
from src.core.pipeline import PerceptionPipeline

# Initialize
pipeline = PerceptionPipeline(
    sam2_checkpoint="./data/models/base/sam2_hiera_large.pt",
    zimage_model="./data/models/base/zimage",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    config_path="./config.yaml"
)

# 1. Segment video
video_path = Path("wildlife.mp4")
session = pipeline.process_video(video_path)
print(f"Segmented {len(session.frames)} frames")

# 2. Query uncertain regions
uncertain_regions = pipeline.get_uncertain_regions(session.id)
vlm_results = pipeline.label_uncertain_regions(uncertain_regions)
print(f"Labeled {len(vlm_results)} regions")

# 3. Generate synthetic data
for vlm_result in vlm_results:
    if vlm_result.user_action == "accepted":
        synthetic_images = pipeline.generate_training_data(
            label=vlm_result.response["label"],
            num_images=20
        )
        print(f"Generated {len(synthetic_images)} images for {vlm_result.response['label']}")

# 4. Train model
training_job = pipeline.train_model(session.id, epochs=5)
print(f"Training started: {training_job.job_id}")

# 5. Evaluate
while training_job.status != "completed":
    time.sleep(10)
    training_job.refresh()
    print(f"Epoch {training_job.progress.epoch}/{training_job.progress.total_epochs}")

metrics = training_job.result.final_metrics
print(f"Final IoU: {metrics.iou:.3f}")
print(f"Uncertainty reduction: {metrics.uncertainty_reduction:.1%}")
```
