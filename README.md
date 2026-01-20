# Inquis: Self-Improving Visual Perception Agent

> **Status**: Active Development | **Version**: 0.2.0-alpha | **Current Feature**: Semantic Labeling for All Regions

A self-improving visual perception system that processes videos through instance segmentation (SAM2), detects uncertain regions, queries vision-language models (GPT-5.2) for labels, generates synthetic training data (Z-Image), and incrementally retrains the segmentation model.

## Project Overview

**Inquis** (from Latin *inquīrere* - to inquire, investigate) is an autonomous AI agent that creates a feedback loop for continuous visual perception improvement. Unlike traditional segmentation models that struggle with ambiguous regions, Inquis:

1. **Segments** videos using SAM2 (Meta's Segment Anything Model 2)
2. **Detects** regions with low confidence or uncertainty
3. **Queries** vision-language models (GPT-5.2/GPT-4o) for intelligent semantic labeling
4. **Generates** synthetic training data using Z-Image (Alibaba Tongyi-MAI)
5. **Retrains** the segmentation model to improve performance
6. **Repeats** the cycle with progressively better accuracy

This creates an autonomous learning system where the model identifies and corrects its own weaknesses without manual intervention.

## Development Status

### Current Progress

| Component | Status | Notes |
|-----------|--------|-------|
| **Feature 001: Core Perception Agent** | ✅ 95% Complete | Video segmentation, uncertainty detection, VLM labeling functional |
| **Feature 002: Semantic Labeling** | 🔄 In Progress | Automatic labeling of all regions with cost controls |
| **GUI Interface** | ✅ Functional | Streamlit-based, active development |
| **VLM Integration** | ✅ Complete | Rate limiting, error handling, cost tracking |
| **Test Coverage** | ✅ Strong | Unit, integration, and contract tests |
| **Documentation** | ✅ Comprehensive | Specs, plans, research docs maintained |

### Codebase Statistics

- **7,036** lines of Python source code
- **41** Python modules in `src/`
- **15+** test files with comprehensive coverage
- **2** major features specified and in development

### Active Branch

- **Main**: `main` (stable baseline)
- **Development**: `002-semantic-labeling-all-regions` (current work)

### Roadmap

- [x] **Phase 1-2**: Project setup and foundational infrastructure
- [x] **Phase 3**: Error handling and rate limiting
- [x] **Phase 4**: Video processing pipeline
- [x] **Phase 5**: VLM-assisted labeling for uncertain regions
- [ ] **Phase 6**: Semantic labeling for ALL regions (current)
- [ ] **Phase 7**: Synthetic data generation pipeline
- [ ] **Phase 8**: Model retraining and LoRA fine-tuning
- [ ] **Phase 9**: Performance optimization and cost reduction
- [ ] **Phase 10**: Production deployment

## Features

- **Video Segmentation**: Process videos frame-by-frame with SAM2 instance segmentation
- **Uncertainty Detection**: Identify and highlight low-confidence regions
- **VLM-Assisted Labeling**: Query GPT-5.2 for accurate labels with reasoning
- **Synthetic Data Generation**: Create training images with Z-Image
- **Incremental Training**: Fine-tune SAM2 with LoRA on synthetic data
- **Interactive GUI**: Streamlit-based interface for visualization

## Self-Improvement Loop

```
1. Segmentation → 2. Uncertainty Detection → 3. VLM Labeling → 4. Synthetic Generation → 5. Model Retraining
     ↑                                                                                              ↓
     └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### Hardware
- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 4080, A100) or Apple Silicon with 32GB+ unified memory
- **CPU**: 8+ cores recommended
- **RAM**: 32GB+ system RAM
- **Storage**: 100GB+ free space

### Software
- Python 3.11 or later
- CUDA 11.8+ (for NVIDIA GPUs) or MPS (for Apple Silicon)
- FFmpeg for video processing

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/zebehn/inquis.git
cd inquis

# For stable version (recommended for new users):
git checkout main

# For latest development features:
git checkout 002-semantic-labeling-all-regions
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

# Install SAM2
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### 4. Download Model Weights

```bash
# Download SAM2 weights
python scripts/download_sam2.py

# Download Z-Image weights
python scripts/download_zimage.py
```

### 5. Configure API Keys

```bash
# Copy template and add your API key
cp .env.template .env
# Edit .env and add your OPENAI_API_KEY
```

## Quick Start

### Run Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit -v

# Run with coverage
pytest --cov=src --cov-report=html
```

### Launch GUI

```bash
streamlit run src/gui/app.py
```

The interface will be available at `http://localhost:8501`

### Basic Workflow

1. **Upload Video**: Click "Upload Video" and select MP4/AVI/MOV file
2. **Process**: Click "Start Processing" to segment frames
3. **Review Uncertainty**: Navigate to "Uncertainty" tab to see low-confidence regions
4. **Query VLM**: Select uncertain regions and click "Query VLM" for labels
5. **Generate Synthetic Data**: Navigate to "Synthesis" tab and generate training images
6. **Train Model**: Navigate to "Training" tab and start model retraining
7. **Evaluate**: View performance improvements in "Metrics" tab

## Project Structure

```
inquis/
├── src/
│   ├── models/              # Pydantic data models
│   ├── services/            # Core services (segmentation, VLM, generation, training)
│   ├── gui/                 # Streamlit interface
│   ├── core/                # Pipeline orchestration, config, metrics
│   └── utils/               # Utilities (image, mask, logging)
├── tests/
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── contract/            # API contract tests
├── data/
│   ├── sessions/            # Processed video sessions
│   ├── models/              # Model checkpoints
│   └── datasets/            # Training datasets
├── specs/                   # Feature specifications and design docs
├── config.yaml              # Configuration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Configuration

Edit `config.yaml` to customize:
- Model parameters (SAM2, Z-Image, GPT-5.2)
- Processing settings (thresholds, batch sizes)
- Training hyperparameters (learning rate, epochs, LoRA rank)
- Storage and performance options

## Development

### Run Linting and Type Checking

```bash
# Linting
ruff check src/

# Type checking
mypy src/

# Format code
ruff format src/
```

### TDD Workflow

1. Write failing test (RED)
2. Implement minimum code to pass (GREEN)
3. Refactor while keeping tests green
4. Commit with appropriate prefix (`feat:`, `fix:`, `refactor:`)

## Performance Benchmarks

Expected performance on RTX 4080 (16GB VRAM):

| Operation | Time | Notes |
|-----------|------|-------|
| Frame segmentation | 0.3-0.5s/frame | Batch size 4 |
| VLM query | 1-2s/query | API latency dependent |
| Synthetic generation | 10-15s/batch | 20 images |
| Model training | 5-10 min/epoch | LoRA, 500 images |

## Success Criteria

- Video processing ≤ 2x video duration
- 90% precision on uncertain region detection
- 85% VLM labeling accuracy
- 30% uncertainty reduction after retraining
- <200ms frame navigation latency

## Documentation

- [Feature Specification](specs/001-visual-perception-agent/spec.md)
- [Implementation Plan](specs/001-visual-perception-agent/plan.md)
- [Research & Technology Decisions](specs/001-visual-perception-agent/research.md)
- [Data Model](specs/001-visual-perception-agent/data-model.md)
- [Quickstart Guide](specs/001-visual-perception-agent/quickstart.md)
- [Task Breakdown](specs/001-visual-perception-agent/tasks.md)

## Troubleshooting

See [quickstart.md](specs/001-visual-perception-agent/quickstart.md) for detailed troubleshooting guide.

## Development Methodology

Inquis follows **Test-Driven Development (TDD)** and **Tidy First** principles as defined in the [project constitution](.specify/memory/constitution.md):

- **Red-Green-Refactor cycle**: Write failing tests first, implement minimally, then refactor
- **Structural vs. Behavioral changes**: Never mix refactoring with feature work
- **Commit discipline**: All tests must pass before committing
- **Code quality**: Eliminate duplication, express intent clearly, maintain single responsibility

See [constitution.md](.specify/memory/constitution.md) for complete development principles.

## Architecture Highlights

### Modular Design

- **Independent VLM Module** (`src/vlm/`): Standalone OpenAI integration with rate limiting
- **Service Layer** (`src/services/`): Segmentation, VLM, Storage, Semantic Labeling services
- **Pydantic Models** (`src/models/`): Type-safe data structures for all domain objects
- **Streamlit GUI** (`src/gui/`): Component-based UI with clear separation of concerns

### Key Design Decisions

1. **SAM2 provides generic IDs**, VLM adds semantic labels
2. **Self-improvement focuses on segmentation quality** (predicted_iou), not classification
3. **VLM uncertainty triggers**: confidence < 0.5 OR 15 ambiguous keywords detected
4. **Synthetic data dual-scoring**: SAM2 re-segmentation quality + VLM label correspondence
5. **Cost-conscious**: Budget limits, frame sampling, and cost tracking throughout

## License

MIT License - Copyright (c) 2025 Minsu Jang

See [LICENSE](LICENSE) for full text.

## Contributing

Contributions are welcome! This project follows strict TDD practices:

1. **Read the constitution** first: [.specify/memory/constitution.md](.specify/memory/constitution.md)
2. **Write tests first** (Red phase)
3. **Implement minimally** (Green phase)
4. **Refactor separately** (never mix structural and behavioral changes)
5. **All tests must pass** before submitting PR
6. **Follow commit conventions**: `feat:`, `fix:`, `refactor:`, `tidy:`, `docs:`, `test:`

For major changes, please open an issue first to discuss the proposed changes.

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/zebehn/inquis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zebehn/inquis/discussions)
- **Author**: Minsu Jang
