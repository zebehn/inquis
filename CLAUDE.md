# Inquis Development Context

**Quick reference for AI assistants working on the Inquis project**

Last updated: 2026-01-20

## Project Overview

**Inquis** is a self-improving visual perception agent that uses SAM2 for video segmentation, detects uncertain regions, queries VLMs (GPT-4o/GPT-5.2) for semantic labels, generates synthetic training data, and retrains models to improve performance autonomously.

## Current Status

- **Active Branch**: `002-semantic-labeling-all-regions`
- **Development Phase**: Phase 6 (Semantic Labeling for All Regions)
- **Codebase**: 7,036 LOC, 41 Python modules
- **Test Coverage**: Comprehensive (unit, integration, contract tests)

## Technology Stack

- **Language**: Python 3.11+
- **ML/CV**: PyTorch 2.0+, SAM2 (Meta), Z-Image (Alibaba), OpenAI GPT-4o/GPT-5.2
- **GUI**: Streamlit 1.28+
- **Data**: Pydantic 2.0+ models, file-based JSON storage
- **Testing**: pytest with coverage, mypy, ruff
- **Utilities**: OpenCV, NumPy, Pillow, scikit-learn

## Project Structure

```text
inquis/
├── src/
│   ├── core/          # Config, metrics, pipeline orchestration
│   ├── services/      # VLMService, SegmentationService, StorageService, SemanticLabelingService
│   ├── vlm/           # Independent VLM module (client, rate limiter, models)
│   ├── models/        # Pydantic data models
│   ├── gui/           # Streamlit interface
│   └── utils/         # Image/mask utilities, logging
├── tests/
│   ├── unit/          # Unit tests
│   ├── integration/   # Integration tests
│   └── contract/      # API contract tests
├── specs/             # Feature specifications
│   ├── 001-visual-perception-agent/  # Feature 1 (95% complete)
│   └── 002-semantic-labeling-all-regions/  # Feature 2 (in progress)
├── data/              # Sessions, models, datasets
├── docs/              # Documentation
└── config.yaml        # Main configuration
```

## Development Methodology

**MUST follow TDD principles** as defined in [.specify/memory/constitution.md](.specify/memory/constitution.md):

### Red-Green-Refactor Cycle (NON-NEGOTIABLE)

1. **Red**: Write failing test first
2. **Green**: Implement minimum code to pass
3. **Refactor**: Improve structure only when tests are green

### Key Principles

- **Test-First**: No implementation without failing test
- **Structural vs. Behavioral**: Never mix refactoring with feature work
- **Commit Discipline**: All tests must pass before commit
- **Simplicity**: YAGNI - avoid premature abstraction

### Commit Message Format

- Structural: `refactor:` or `tidy:`
- Behavioral: `feat:` or `fix:`
- Documentation: `docs:`
- Tests: `test:`

## Common Commands

```bash
# Testing
pytest                                    # Run all tests
pytest tests/unit -v                      # Unit tests only
pytest --cov=src --cov-report=html        # With coverage

# Code Quality
ruff check src/                           # Linting
ruff format src/                          # Formatting
mypy src/                                 # Type checking

# Run Application
streamlit run src/gui/app.py              # Launch GUI

# Development
source .venv/bin/activate                 # Activate venv
pip install -r requirements.txt           # Install deps
```

## Important Files

- **Constitution**: `.specify/memory/constitution.md` - Development principles (READ FIRST)
- **Main Config**: `config.yaml` - Application configuration
- **Feature 001 Spec**: `specs/001-visual-perception-agent/spec.md`
- **Feature 002 Spec**: `specs/002-semantic-labeling-all-regions/spec.md`
- **VLM Module Guide**: `docs/vlm_module_usage.md` - VLM integration reference

## Key Architecture Decisions

1. **SAM2 provides generic IDs** (object_1, object_2); VLM adds semantic labels
2. **Self-improvement focuses on segmentation quality** (predicted_iou), not classification
3. **VLM uncertainty**: confidence < 0.5 OR 15 ambiguous keywords detected
4. **Dual scoring for synthetic data**: SAM2 re-segmentation + VLM label correspondence
5. **Cost-conscious design**: Budget limits, rate limiting, frame sampling

## Performance Targets

- Video processing ≤ 2x video duration
- 90% precision on uncertain region detection
- 85% VLM labeling accuracy
- 30% uncertainty reduction after retraining
- <200ms frame navigation latency

## Active Features

### Feature 001: Visual Perception Agent (95% Complete)
- ✅ Video segmentation with SAM2
- ✅ Uncertainty detection
- ✅ VLM-assisted labeling for uncertain regions
- 🔄 Synthetic data generation (planned)
- 🔄 Model retraining (planned)

### Feature 002: Semantic Labeling All Regions (In Progress)
- 🔄 Automatic VLM labeling of ALL regions
- 🔄 Cost-controlled batch processing
- 🔄 Semantic uncertainty pattern detection

## Workflow Integration

When instructed with **"go"**:
1. Consult `plan.md` to find next unmarked test
2. Write the failing test (RED)
3. Implement minimum code (GREEN)
4. Refactor if needed (keep GREEN)
5. Mark test complete in `plan.md`
6. Commit with proper prefix

## Contact

- **Author**: Minsu Jang
- **License**: MIT
- **Repository**: https://github.com/zebehn/inquis

---

**Note for AI Assistants**: Always read the constitution file before making changes. Follow TDD strictly. Never mix structural and behavioral changes. All tests must pass before committing.
