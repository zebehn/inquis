# Implementation Plan: Self-Improving Visual Perception Agent

**Branch**: `001-visual-perception-agent` | **Date**: 2025-12-17 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-visual-perception-agent/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Build a self-improving visual perception agent that processes videos through instance segmentation (SAM2), detects uncertain regions, queries VLMs (GPT-5.2) for labels, generates synthetic training data (Z-Image), and incrementally retrains the segmentation model. The system includes a Streamlit-based GUI for visualizing the agent's cognitive process in real-time.

The core self-improvement loop: **Segmentation → Uncertainty Detection → VLM Labeling → Synthetic Data Generation → Model Retraining** creates a continuously learning perception system.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**:
- SAM2 (Meta's Segment Anything Model 2) for instance segmentation
- OpenAI API (GPT-5.2) for vision-language labeling
- Z-Image (Alibaba Tongyi-MAI) for synthetic image generation
- Streamlit for web-based GUI
- PyTorch 2.0+ for deep learning inference and training
- OpenCV for video processing
- NumPy/Pillow for image manipulation

**Storage**: File-based storage for:
- Video sessions and processed frames
- Segmentation masks and uncertainty maps
- VLM query results and labels
- Synthetic training images
- Model checkpoints and version history

**Testing**: pytest with:
- Unit tests for core components (segmentation, uncertainty detection, labeling)
- Integration tests for pipeline stages
- Contract tests for VLM API interactions
- End-to-end tests for full self-improvement cycle

**Target Platform**: Linux/macOS desktop with GPU support (CUDA 11.8+ or MPS)

**Project Type**: Single project with ML pipeline and web interface

**Performance Goals**:
- Video processing ≤ 2x video duration
- Frame navigation latency < 200ms
- GUI update latency < 1 second
- Synthetic image generation: 20 images per region within 5 minutes

**Constraints**:
- Minimum 16GB VRAM required for concurrent SAM2 and Z-Image inference
- GPU acceleration required for real-time performance
- API rate limits for GPT-5.2 queries
- Storage scaling with video length and synthetic data volume

**Scale/Scope**:
- Support videos up to 10 minutes
- Handle 100+ uncertain regions per video
- Track up to 10 model versions
- Store thousands of synthetic training images

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Test-Driven Development (TDD) Cycle**: ✅ **PASS**
- Plan includes test-first approach for all components
- Each user story has testable acceptance criteria
- Implementation will follow Red-Green-Refactor cycle

**II. Test-First Discipline**: ✅ **PASS**
- All functional requirements map to testable scenarios
- Tests will be written before implementation code
- Failing tests will drive implementation

**III. Tidy First: Structural vs. Behavioral Changes**: ✅ **PASS**
- Commits will separate refactoring from feature additions
- Each phase output represents logical behavioral units

**IV. Code Quality Standards**: ✅ **PASS**
- Design emphasizes clear responsibilities for each component
- Single-purpose functions for segmentation, uncertainty detection, labeling, generation
- Explicit dependencies through dependency injection

**V. Refactoring Discipline**: ✅ **PASS**
- Refactoring only during Green phase
- Test coverage ensures safe refactoring

**VI. Commit Discipline**: ✅ **PASS**
- All commits require passing tests
- Structural changes (`refactor:`) separated from behavioral (`feat:`, `fix:`)

**VII. Simplicity and YAGNI**: ✅ **PASS**
- Start with basic implementations (P1: segmentation)
- Add features incrementally (P2-P6)
- No premature abstractions until patterns emerge

**Constitution Compliance**: ALL GATES PASSED ✅

No complexity violations to justify. The project follows TDD principles with test-first development, incremental feature delivery, and clear separation of concerns.

## Project Structure

### Documentation (this feature)

```text
specs/001-visual-perception-agent/
├── spec.md              # Feature specification with user stories
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output - technology research
├── data-model.md        # Phase 1 output - entity definitions
├── quickstart.md        # Phase 1 output - setup and usage guide
├── contracts/           # Phase 1 output - API contracts
│   ├── segmentation.yaml
│   ├── vlm.yaml
│   ├── generation.yaml
│   └── training.yaml
└── checklists/
    └── requirements.md  # Specification quality validation
```

### Source Code (repository root)

```text
src/
├── models/
│   ├── video_session.py
│   ├── segmentation_frame.py
│   ├── uncertain_region.py
│   ├── vlm_query.py
│   ├── synthetic_image.py
│   ├── model_version.py
│   └── training_dataset.py
├── services/
│   ├── segmentation_service.py      # SAM2 inference and uncertainty detection
│   ├── vlm_service.py                # GPT-5.2 API integration
│   ├── generation_service.py         # Z-Image synthetic data creation
│   ├── training_service.py           # SAM2 incremental fine-tuning
│   ├── video_processor.py            # Frame extraction and processing
│   └── storage_service.py            # File-based persistence
├── gui/
│   ├── app.py                        # Streamlit main application
│   ├── components/
│   │   ├── video_viewer.py
│   │   ├── segmentation_viz.py
│   │   ├── uncertainty_viz.py
│   │   ├── labeling_panel.py
│   │   ├── generation_panel.py
│   │   └── training_panel.py
│   └── utils/
│       ├── visualization.py
│       └── state_management.py
├── core/
│   ├── pipeline.py                   # Self-improvement pipeline orchestration
│   ├── config.py                     # Configuration management
│   └── metrics.py                    # Performance tracking
└── utils/
    ├── image_utils.py
    ├── mask_utils.py
    └── logging.py

tests/
├── unit/
│   ├── test_segmentation_service.py
│   ├── test_vlm_service.py
│   ├── test_generation_service.py
│   ├── test_training_service.py
│   ├── test_video_processor.py
│   └── test_models.py
├── integration/
│   ├── test_pipeline.py
│   ├── test_storage.py
│   └── test_gui_components.py
└── contract/
    ├── test_vlm_contract.py
    └── test_model_contracts.py
```

**Structure Decision**: Single project structure chosen because:
1. All components share Python runtime and ML dependencies
2. Tight integration between services (segmentation → uncertainty → labeling → generation → training)
3. Streamlit GUI is embedded within the same application
4. No separate frontend/backend needed - Streamlit handles both
5. Simplifies deployment and dependency management

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations detected. This section is intentionally empty.

---

## Phase 0: Research (COMPLETED)

✅ **Output**: `research.md` - Comprehensive technology research covering:
- SAM2 for segmentation with uncertainty detection
- GPT-5.2 for VLM labeling
- Z-Image for synthetic data generation
- Streamlit for GUI framework
- Implementation patterns and best practices
- Performance optimization strategies
- Security and monitoring considerations

---

## Phase 1: Design & Contracts (COMPLETED)

✅ **Output**: `data-model.md` - Complete entity definitions with:
- 7 core entities (VideoSession, SegmentationFrame, UncertainRegion, VLMQuery, SyntheticImage, ModelVersion, TrainingDataset)
- Field specifications with validation rules
- State transitions and relationships
- Storage formats (JSON-based)
- Data access patterns

✅ **Output**: `contracts/` - API contracts for all services:
- `segmentation.yaml` - SAM2 segmentation and uncertainty detection interface
- `vlm.yaml` - GPT-5.2 vision-language labeling interface
- `generation.yaml` - Z-Image synthetic data generation interface
- `training.yaml` - Incremental SAM2 training interface

✅ **Output**: `quickstart.md` - Comprehensive setup guide with:
- Hardware/software prerequisites
- Installation instructions
- Configuration examples
- Basic workflow tutorial
- Troubleshooting guide
- Performance benchmarks

✅ **Output**: `CLAUDE.md` - Updated agent context file

---

## Constitution Check Re-evaluation (POST-DESIGN)

**I. Test-Driven Development (TDD) Cycle**: ✅ **PASS**
- Design supports test-first development
- Clear service interfaces enable unit testing
- Contract tests defined for external APIs

**II. Test-First Discipline**: ✅ **PASS**
- All services have testable contracts
- Data models include validation rules
- Acceptance criteria map to automated tests

**III. Tidy First: Structural vs. Behavioral Changes**: ✅ **PASS**
- Modular architecture enables independent refactoring
- Clear service boundaries allow isolated changes

**IV. Code Quality Standards**: ✅ **PASS**
- Single responsibility per service
- Explicit dependencies via dependency injection
- Type hints and validation (Pydantic models)

**V. Refactoring Discipline**: ✅ **PASS**
- Service abstractions enable safe refactoring
- Comprehensive test coverage planned

**VI. Commit Discipline**: ✅ **PASS**
- Development workflow supports atomic commits
- Clear categorization of changes

**VII. Simplicity and YAGNI**: ✅ **PASS**
- Start with P1 (basic segmentation), add features incrementally
- No premature abstractions in design
- File-based storage (simple) over database (complex)

**Final Verdict**: ALL GATES PASSED ✅

No design changes needed. Implementation can proceed following TDD workflow.

---

## Next Steps

1. ✅ Phase 0: Research complete
2. ✅ Phase 1: Design & contracts complete
3. ➡️ **Phase 2**: Generate task breakdown with `/speckit.tasks`
4. ➡️ **Implementation**: Follow TDD workflow in `tasks.md`
