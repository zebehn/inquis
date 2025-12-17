# Tasks: Self-Improving Visual Perception Agent

**Input**: Design documents from `/specs/001-visual-perception-agent/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Following TDD constitution - ALL tests must be written FIRST and FAIL before implementation

**Organization**: Tasks grouped by user story for independent implementation and testing

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: User story label (US1, US2, US3, US4, US5, US6)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create project directory structure per plan.md (src/, tests/, data/)
- [x] T002 Initialize Python 3.11+ project with requirements.txt
- [x] T003 [P] Configure pytest with coverage and test markers in pytest.ini
- [x] T004 [P] Setup .env template with OPENAI_API_KEY and DATA_DIR
- [x] T005 [P] Create config.yaml template from quickstart.md
- [x] T006 [P] Initialize .gitignore for Python, models, data directories
- [x] T007 [P] Create README.md with project overview and setup instructions

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T008 Create base configuration manager in src/core/config.py
- [x] T009 [P] Setup logging infrastructure in src/utils/logging.py
- [x] T010 [P] Implement image utilities (load, save, resize) in src/utils/image_utils.py
- [x] T011 [P] Implement mask utilities (save/load .npz, overlay) in src/utils/mask_utils.py
- [x] T012 [P] Create storage service base class with file operations in src/services/storage_service.py
- [x] T013 [P] Write unit test for config manager in tests/unit/test_config.py (TDD - write FIRST, ensure FAILS)
- [x] T014 [P] Write unit tests for image utilities in tests/unit/test_image_utils.py (TDD - write FIRST, ensure FAILS)
- [x] T015 [P] Write unit tests for mask utilities in tests/unit/test_mask_utils.py (TDD - write FIRST, ensure FAILS)
- [x] T016 [P] Write unit tests for storage service in tests/unit/test_storage_service.py (TDD - write FIRST, ensure FAILS)
- [x] T017 Implement config manager to pass tests in src/core/config.py
- [x] T018 [P] Implement image utilities to pass tests in src/utils/image_utils.py
- [x] T019 [P] Implement mask utilities to pass tests in src/utils/mask_utils.py
- [x] T020 [P] Implement storage service to pass tests in src/services/storage_service.py
- [x] T021 Create metrics tracking infrastructure in src/core/metrics.py
- [x] T022 Create data directory structure (sessions/, models/base/, models/versions/, datasets/)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Basic Video Analysis with Segmentation (Priority: P1) 🎯 MVP

**Goal**: Load video, segment frames with SAM2, display results with class labels and confidence scores

**Independent Test**: Load a video file and verify segmentation masks are produced for each frame with class labels and confidence scores displayed

### Tests for User Story 1 (TDD - Write FIRST, ensure FAILS)

- [ ] T023 [P] [US1] Write unit test for VideoSession model validation in tests/unit/test_models.py
- [ ] T024 [P] [US1] Write unit test for SegmentationFrame model validation in tests/unit/test_models.py
- [ ] T025 [P] [US1] Write unit test for video_processor frame extraction in tests/unit/test_video_processor.py
- [ ] T026 [P] [US1] Write unit test for segmentation_service SAM2 inference in tests/unit/test_segmentation_service.py
- [ ] T027 [P] [US1] Write integration test for video-to-segmentation pipeline in tests/integration/test_pipeline.py
- [ ] T028 [P] [US1] Write integration test for storage persistence in tests/integration/test_storage.py

### Implementation for User Story 1

- [ ] T029 [P] [US1] Create VideoSession Pydantic model in src/models/video_session.py
- [ ] T030 [P] [US1] Create SegmentationFrame Pydantic model in src/models/segmentation_frame.py
- [ ] T031 [US1] Implement VideoProcessor with frame extraction using OpenCV in src/services/video_processor.py
- [ ] T032 [US1] Implement SegmentationService with SAM2 model loading in src/services/segmentation_service.py
- [ ] T033 [US1] Implement SAM2 inference method (segment_frame) in src/services/segmentation_service.py
- [ ] T034 [US1] Add VideoSession persistence methods to StorageService in src/services/storage_service.py
- [ ] T035 [US1] Add SegmentationFrame persistence methods to StorageService in src/services/storage_service.py
- [ ] T036 [US1] Create Streamlit app structure in src/gui/app.py
- [ ] T037 [P] [US1] Implement video upload component in src/gui/components/video_viewer.py
- [ ] T038 [P] [US1] Implement segmentation visualization component in src/gui/components/segmentation_viz.py
- [ ] T039 [P] [US1] Implement visualization utilities (overlay masks, colors) in src/gui/utils/visualization.py
- [ ] T040 [P] [US1] Implement Streamlit session state management in src/gui/utils/state_management.py
- [ ] T041 [US1] Wire video upload → processing → display in src/gui/app.py
- [ ] T042 [US1] Add frame navigation controls (prev/next) in src/gui/app.py
- [ ] T043 [US1] Add progress bar and status display in src/gui/app.py
- [ ] T044 [US1] Add error handling for invalid video formats in src/services/video_processor.py
- [ ] T045 [US1] Add logging for video processing operations in src/services/video_processor.py

**Checkpoint**: User Story 1 complete - can load video, segment, and view results

---

## Phase 4: User Story 2 - Uncertainty Detection and Visualization (Priority: P2)

**Goal**: Detect and highlight uncertain segmentation regions with visual markers and summary statistics

**Independent Test**: Process video with challenging objects and verify low-confidence regions are visually distinguished with confidence scores and statistics

### Tests for User Story 2 (TDD - Write FIRST, ensure FAILS)

- [ ] T046 [P] [US2] Write unit test for UncertainRegion model validation in tests/unit/test_models.py
- [ ] T047 [P] [US2] Write unit test for uncertainty computation in tests/unit/test_segmentation_service.py
- [ ] T048 [P] [US2] Write unit test for uncertain region extraction in tests/unit/test_segmentation_service.py
- [ ] T049 [P] [US2] Write integration test for uncertainty detection pipeline in tests/integration/test_pipeline.py

### Implementation for User Story 2

- [ ] T050 [P] [US2] Create UncertainRegion Pydantic model in src/models/uncertain_region.py
- [ ] T051 [US2] Implement uncertainty computation from SAM2 logits in src/services/segmentation_service.py
- [ ] T052 [US2] Implement uncertain region extraction and cropping in src/services/segmentation_service.py
- [ ] T053 [US2] Add UncertainRegion persistence methods to StorageService in src/services/storage_service.py
- [ ] T054 [P] [US2] Implement uncertainty map visualization in src/gui/components/uncertainty_viz.py
- [ ] T055 [P] [US2] Implement uncertain region highlighting in src/gui/components/segmentation_viz.py
- [ ] T056 [US2] Add uncertainty detection toggle to GUI in src/gui/app.py
- [ ] T057 [US2] Add uncertainty statistics panel (count, percentage) in src/gui/app.py
- [ ] T058 [US2] Add hover/click interactions for uncertain regions in src/gui/components/uncertainty_viz.py
- [ ] T059 [US2] Display top competing predictions for selected uncertain region in src/gui/app.py
- [ ] T060 [US2] Add configuration for uncertainty threshold in src/core/config.py

**Checkpoint**: User Story 2 complete - uncertainty detection and visualization working

---

## Phase 5: User Story 3 - VLM-Assisted Labeling (Priority: P3)

**Goal**: Query GPT-5.2 for labels on uncertain regions, display reasoning, allow user review

**Independent Test**: Select uncertain regions, trigger VLM queries, and verify meaningful labels with explanations are returned and can be accepted/modified

### Tests for User Story 3 (TDD - Write FIRST, ensure FAILS)

- [ ] T061 [P] [US3] Write unit test for VLMQuery model validation in tests/unit/test_models.py
- [ ] T062 [P] [US3] Write unit test for VLM API request formatting in tests/unit/test_vlm_service.py
- [ ] T063 [P] [US3] Write unit test for VLM response parsing in tests/unit/test_vlm_service.py
- [ ] T064 [P] [US3] Write contract test for GPT-5.2 API in tests/contract/test_vlm_contract.py
- [ ] T065 [P] [US3] Write integration test for VLM labeling workflow in tests/integration/test_pipeline.py

### Implementation for User Story 3

- [ ] T066 [P] [US3] Create VLMQuery Pydantic model in src/models/vlm_query.py
- [ ] T067 [US3] Implement VLMService with OpenAI client initialization in src/services/vlm_service.py
- [ ] T068 [US3] Implement label_region method with API call in src/services/vlm_service.py
- [ ] T069 [US3] Implement retry logic with exponential backoff in src/services/vlm_service.py
- [ ] T070 [US3] Implement VLM response parsing in src/services/vlm_service.py
- [ ] T071 [US3] Add VLMQuery persistence methods to StorageService in src/services/storage_service.py
- [ ] T072 [P] [US3] Implement labeling panel UI in src/gui/components/labeling_panel.py
- [ ] T073 [US3] Add "Query VLM" button for uncertain regions in src/gui/app.py
- [ ] T074 [US3] Display VLM response (label, confidence, reasoning) in src/gui/components/labeling_panel.py
- [ ] T075 [US3] Add accept/reject/modify buttons for VLM labels in src/gui/components/labeling_panel.py
- [ ] T076 [US3] Update UncertainRegion status after user action in src/services/storage_service.py
- [ ] T077 [US3] Add cost tracking for VLM queries in src/core/metrics.py
- [ ] T078 [US3] Add batch VLM query support in src/services/vlm_service.py
- [ ] T079 [US3] Add rate limit handling and error messages in src/services/vlm_service.py

**Checkpoint**: User Story 3 complete - VLM-assisted labeling functional

---

## Phase 6: User Story 4 - Synthetic Training Data Generation (Priority: P4)

**Goal**: Generate synthetic images with Z-Image based on confirmed labels, create masks via re-segmentation, allow quality review

**Independent Test**: Provide a confirmed label and verify system generates 20+ synthetic images with masks that can be reviewed and filtered

### Tests for User Story 4 (TDD - Write FIRST, ensure FAILS)

- [ ] T080 [P] [US4] Write unit test for SyntheticImage model validation in tests/unit/test_models.py
- [ ] T081 [P] [US4] Write unit test for Z-Image generation in tests/unit/test_generation_service.py
- [ ] T082 [P] [US4] Write unit test for quality scoring in tests/unit/test_generation_service.py
- [ ] T083 [P] [US4] Write integration test for generation pipeline in tests/integration/test_pipeline.py

### Implementation for User Story 4

- [ ] T084 [P] [US4] Create SyntheticImage Pydantic model in src/models/synthetic_image.py
- [ ] T085 [US4] Implement GenerationService with Z-Image model loading in src/services/generation_service.py
- [ ] T086 [US4] Implement prompt template creation for labels in src/services/generation_service.py
- [ ] T087 [US4] Implement generate method with Z-Image inference in src/services/generation_service.py
- [ ] T088 [US4] Implement re-segmentation of generated images with SAM2 in src/services/generation_service.py
- [ ] T089 [US4] Implement quality scoring (CLIP similarity) in src/services/generation_service.py
- [ ] T090 [US4] Add SyntheticImage persistence methods to StorageService in src/services/storage_service.py
- [ ] T091 [P] [US4] Implement generation panel UI in src/gui/components/generation_panel.py
- [ ] T092 [US4] Add "Generate" button for labeled regions in src/gui/app.py
- [ ] T093 [US4] Display generated images with masks in grid view in src/gui/components/generation_panel.py
- [ ] T094 [US4] Show quality scores for each generated image in src/gui/components/generation_panel.py
- [ ] T095 [US4] Add accept/reject controls for synthetic images in src/gui/components/generation_panel.py
- [ ] T096 [US4] Add batch generation progress tracking in src/gui/app.py
- [ ] T097 [US4] Add generation parameter controls (num_images, steps) in src/gui/components/generation_panel.py

**Checkpoint**: User Story 4 complete - synthetic data generation working

---

## Phase 7: User Story 5 - Model Retraining and Performance Tracking (Priority: P5)

**Goal**: Retrain SAM2 with LoRA using synthetic data, track metrics, show performance improvements

**Independent Test**: Initiate training with synthetic data and verify updated model shows improved performance with metrics timeline

### Tests for User Story 5 (TDD - Write FIRST, ensure FAILS)

- [ ] T098 [P] [US5] Write unit test for ModelVersion model validation in tests/unit/test_models.py
- [ ] T099 [P] [US5] Write unit test for TrainingDataset model validation in tests/unit/test_models.py
- [ ] T100 [P] [US5] Write unit test for LoRA setup in tests/unit/test_training_service.py
- [ ] T101 [P] [US5] Write unit test for training loop in tests/unit/test_training_service.py
- [ ] T102 [P] [US5] Write integration test for full training cycle in tests/integration/test_pipeline.py

### Implementation for User Story 5

- [ ] T103 [P] [US5] Create ModelVersion Pydantic model in src/models/model_version.py
- [ ] T104 [P] [US5] Create TrainingDataset Pydantic model in src/models/training_dataset.py
- [ ] T105 [US5] Implement TrainingService with LoRA configuration in src/services/training_service.py
- [ ] T106 [US5] Implement training dataset preparation (combine original + synthetic) in src/services/training_service.py
- [ ] T107 [US5] Implement training loop with LoRA in src/services/training_service.py
- [ ] T108 [US5] Implement validation and metrics computation (IoU, precision, recall) in src/services/training_service.py
- [ ] T109 [US5] Implement model checkpoint saving in src/services/training_service.py
- [ ] T110 [US5] Add ModelVersion persistence methods to StorageService in src/services/storage_service.py
- [ ] T111 [US5] Add TrainingDataset persistence methods to StorageService in src/services/storage_service.py
- [ ] T112 [P] [US5] Implement training panel UI in src/gui/components/training_panel.py
- [ ] T113 [US5] Add "Start Training" button with parameter controls in src/gui/app.py
- [ ] T114 [US5] Display training progress (epoch, loss, metrics) in src/gui/components/training_panel.py
- [ ] T115 [US5] Show metrics comparison across model versions in src/gui/components/training_panel.py
- [ ] T116 [US5] Display uncertainty reduction statistics in src/gui/components/training_panel.py
- [ ] T117 [US5] Add model version selection dropdown in src/gui/app.py
- [ ] T118 [US5] Implement model version switching in src/services/segmentation_service.py

**Checkpoint**: User Story 5 complete - full self-improvement loop functional

---

## Phase 8: User Story 6 - Real-Time Cognitive Process Visualization (Priority: P6)

**Goal**: Display all processing stages in real-time with status indicators, intermediate results, and hierarchical organization

**Independent Test**: Run any analysis and verify GUI updates with status indicators, side-by-side visualizations, and stage-specific displays

### Tests for User Story 6 (TDD - Write FIRST, ensure FAILS)

- [ ] T119 [P] [US6] Write integration test for real-time status updates in tests/integration/test_gui_components.py
- [ ] T120 [P] [US6] Write integration test for stage visualization in tests/integration/test_gui_components.py

### Implementation for User Story 6

- [ ] T121 [US6] Create pipeline orchestrator with stage tracking in src/core/pipeline.py
- [ ] T122 [US6] Add stage transition callbacks to pipeline in src/core/pipeline.py
- [ ] T123 [US6] Implement status indicator component in src/gui/app.py
- [ ] T124 [US6] Add side-by-side visualization layout in src/gui/app.py
- [ ] T125 [US6] Implement expandable stage sections in src/gui/app.py
- [ ] T126 [US6] Add real-time progress updates (frame count, time elapsed) in src/gui/app.py
- [ ] T127 [US6] Display intermediate results (raw frame, mask, uncertainty, VLM) in src/gui/app.py
- [ ] T128 [US6] Add stage-specific visualizations (segmentation, uncertainty, labeling, generation, training) in src/gui/app.py
- [ ] T129 [US6] Implement hierarchical organization with tabs/expanders in src/gui/app.py

**Checkpoint**: User Story 6 complete - full cognitive process visualization

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Final improvements affecting multiple user stories

- [ ] T130 [P] Add comprehensive docstrings to all services in src/services/
- [ ] T131 [P] Add type hints to all functions in src/
- [ ] T132 [P] Create download scripts for SAM2 weights in scripts/download_sam2.py
- [ ] T133 [P] Create download scripts for Z-Image weights in scripts/download_zimage.py
- [ ] T134 [P] Add performance monitoring (frame/s, GPU usage) to metrics in src/core/metrics.py
- [ ] T135 [P] Implement graceful error recovery for GPU OOM in src/services/
- [ ] T136 [P] Add data cleanup utilities (old sessions, temp files) in src/utils/cleanup.py
- [ ] T137 [P] Validate quickstart.md with end-to-end manual test
- [ ] T138 [P] Add example test video to tests/fixtures/videos/
- [ ] T139 [P] Run full test suite with coverage report
- [ ] T140 [P] Performance profiling and optimization
- [ ] T141 [P] Security audit (API keys, input validation, error messages)
- [ ] T142 Create deployment documentation in docs/deployment.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phases 3-8)**: All depend on Foundational phase completion
  - Can proceed in parallel if team capacity allows
  - Or sequentially in priority order (P1 → P2 → P3 → P4 → P5 → P6)
- **Polish (Phase 9)**: Depends on all desired user stories being complete

### User Story Dependencies

- **US1 (P1)**: Can start after Foundational - No dependencies on other stories ✅ **MVP**
- **US2 (P2)**: Can start after Foundational - Extends US1 but independently testable
- **US3 (P3)**: Can start after Foundational - Uses US2's uncertain regions but independently testable
- **US4 (P4)**: Can start after Foundational - Uses US3's labels but independently testable
- **US5 (P5)**: Can start after Foundational - Uses US4's synthetic data but independently testable
- **US6 (P6)**: Can start after Foundational - Visualizes all stages but independently testable

### Within Each User Story (TDD Workflow)

1. **RED**: Write all tests marked [P] in parallel - ensure they FAIL
2. **GREEN**: Implement code to make tests pass - models [P], then services, then GUI
3. **REFACTOR**: Improve code structure while keeping tests green
4. **COMMIT**: Commit with appropriate prefix (`feat:` for behavioral, `refactor:` for structural)
5. Verify story independently testable before moving to next priority

### Parallel Opportunities

**Phase 1 (Setup)**: All tasks T001-T007 can run in parallel

**Phase 2 (Foundational)**:
- Tests T013-T016 can run in parallel (different test files)
- Implementations T018-T020 can run in parallel after T017 (different files)

**Phase 3 (US1)**:
- Tests T023-T028 can run in parallel
- Models T029-T030 can run in parallel
- GUI components T037-T040 can run in parallel

**Phase 4 (US2)**:
- Tests T046-T049 can run in parallel
- GUI components T054-T055 can run in parallel

**Phase 5 (US3)**:
- Tests T061-T065 can run in parallel

**Phase 6 (US4)**:
- Tests T080-T083 can run in parallel

**Phase 7 (US5)**:
- Tests T098-T102 can run in parallel
- Models T103-T104 can run in parallel

**Phase 8 (US6)**:
- Tests T119-T120 can run in parallel

**Phase 9 (Polish)**: Most tasks can run in parallel (T130-T142)

---

## Parallel Example: User Story 1 (TDD Workflow)

```bash
# Step 1: Write all tests in parallel (RED phase)
claude: "Write unit test for VideoSession model validation in tests/unit/test_models.py"
claude: "Write unit test for SegmentationFrame model validation in tests/unit/test_models.py"
claude: "Write unit test for video_processor frame extraction in tests/unit/test_video_processor.py"
claude: "Write unit test for segmentation_service SAM2 inference in tests/unit/test_segmentation_service.py"
claude: "Write integration test for video-to-segmentation pipeline in tests/integration/test_pipeline.py"
claude: "Write integration test for storage persistence in tests/integration/test_storage.py"

# Step 2: Run tests - ensure they ALL FAIL
pytest tests/unit tests/integration -v

# Step 3: Implement models in parallel (GREEN phase)
claude: "Create VideoSession Pydantic model in src/models/video_session.py"
claude: "Create SegmentationFrame Pydantic model in src/models/segmentation_frame.py"

# Step 4: Run model tests - ensure they PASS
pytest tests/unit/test_models.py -v

# Step 5: Implement services sequentially (models → services)
claude: "Implement VideoProcessor with frame extraction using OpenCV in src/services/video_processor.py"
claude: "Implement SegmentationService with SAM2 model loading in src/services/segmentation_service.py"

# Step 6: Implement GUI components in parallel
claude: "Implement video upload component in src/gui/components/video_viewer.py"
claude: "Implement segmentation visualization component in src/gui/components/segmentation_viz.py"
claude: "Implement visualization utilities in src/gui/utils/visualization.py"
claude: "Implement Streamlit session state management in src/gui/utils/state_management.py"

# Step 7: Wire everything together
claude: "Wire video upload → processing → display in src/gui/app.py"

# Step 8: Run all US1 tests - ensure they ALL PASS
pytest tests/unit tests/integration -v -k "US1 or video or segmentation"

# Step 9: Manual validation
streamlit run src/gui/app.py

# Step 10: REFACTOR if needed, commit
git add . && git commit -m "feat: implement User Story 1 - basic video segmentation"
```

---

## Implementation Strategy

### MVP First (Recommended for Demo)

1. ✅ Complete Phase 1: Setup (T001-T007)
2. ✅ Complete Phase 2: Foundational (T008-T022) - **CRITICAL BLOCKER**
3. ✅ Complete Phase 3: User Story 1 (T023-T045) - **MVP DELIVERED**
4. **STOP and VALIDATE**: Test User Story 1 independently with sample video
5. Deploy/demo basic video segmentation tool

**Deliverable**: Working video segmentation tool with GUI

### Full Self-Improvement Loop

1. Complete Setup + Foundational (foundation ready)
2. Add US1 (T023-T045) → Test independently → Video segmentation ✅
3. Add US2 (T046-T060) → Test independently → Uncertainty detection ✅
4. Add US3 (T061-T079) → Test independently → VLM labeling ✅
5. Add US4 (T080-T097) → Test independently → Synthetic generation ✅
6. Add US5 (T098-T118) → Test independently → Model retraining ✅ **FULL LOOP**
7. Add US6 (T119-T129) → Test independently → Visualization complete ✅
8. Polish (T130-T142) → Production ready

**Deliverable**: Complete self-improving perception system

### Parallel Team Strategy

With 3 developers after Foundational phase completes:

- **Developer A**: US1 + US2 (segmentation + uncertainty)
- **Developer B**: US3 + US4 (VLM labeling + generation)
- **Developer C**: US5 + US6 (training + visualization)

Each developer works independently, commits separately, stories integrate cleanly.

---

## TDD Workflow Reminders

**Constitution Compliance**:
1. ✅ **RED**: Write failing test FIRST
2. ✅ **GREEN**: Implement minimum code to pass
3. ✅ **REFACTOR**: Improve structure with passing tests
4. ✅ **COMMIT**: Separate commits for behavioral (`feat:`, `fix:`) vs structural (`refactor:`, `tidy:`)

**Test-First Checklist**:
- [ ] Write test before implementation
- [ ] Verify test FAILS (RED)
- [ ] Implement to make test PASS (GREEN)
- [ ] Refactor while keeping tests GREEN
- [ ] Run ALL tests before commit
- [ ] Commit with appropriate prefix

---

## Notes

- [P] tasks = different files, no dependencies, can run in parallel
- [Story] label maps task to specific user story (US1-US6)
- Each user story independently completable and testable
- TDD: Tests MUST be written FIRST and FAIL before implementation
- Stop at any checkpoint to validate story independently
- Commit frequently with proper categorization (feat/fix/refactor/tidy)
- US1 alone = MVP, US1-US5 = Full self-improvement loop, US6 = Enhanced UX

**Total Tasks**: 142
- Setup: 7
- Foundational: 15
- US1 (P1): 23 tasks (6 tests + 17 impl) - **MVP**
- US2 (P2): 15 tasks (4 tests + 11 impl)
- US3 (P3): 19 tasks (5 tests + 14 impl)
- US4 (P4): 18 tasks (4 tests + 14 impl)
- US5 (P5): 21 tasks (5 tests + 16 impl)
- US6 (P6): 11 tasks (2 tests + 9 impl)
- Polish: 13 tasks

**Parallel Opportunities**: 67 tasks marked [P] can run concurrently
