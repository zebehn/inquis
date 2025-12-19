# Tasks: Semantic Labeling for All Regions

**Input**: Design documents from `/specs/002-semantic-labeling-all-regions/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Following TDD approach per constitution - all tests written BEFORE implementation

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- All paths are absolute from `/data1/Developments/inquis/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and configuration for semantic labeling feature

- [ ] T001 Verify Python 3.11+ and existing dependencies (OpenAI SDK, pytest, Streamlit, scikit-learn, cv2, numpy)
- [ ] T002 [P] Create feature branch `002-semantic-labeling-all-regions` from main
- [ ] T003 [P] Review existing VLMService implementation in src/services/vlm_service.py for extension points
- [ ] T004 [P] Review existing StorageService implementation in src/services/storage_service.py for checkpoint patterns

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data models and infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Data Models (Foundational)

- [ ] T005 [P] Create SemanticLabelingJob model in src/models/semantic_labeling_job.py with JobStatus enum, progress tracking, cost tracking, and configuration fields per data-model.md
- [ ] T006 [P] Extend SegmentedRegion model in src/models/session.py with tracking fields (tracking_status, parent_region_id, tracked_iou, frames_since_last_query, semantic_label_source)
- [ ] T007 [P] Create SemanticUncertaintyPattern model in src/models/semantic_uncertainty.py with cluster metadata, region_ids, sample paths, similarity scores per data-model.md

### Storage Extensions (Foundational)

- [ ] T008 Extend StorageService in src/services/storage_service.py with atomic checkpoint methods (checkpoint_labeling_job, load_job_checkpoint) using write-then-rename pattern per research.md
- [ ] T009 [P] Add job persistence methods to StorageService (save_job, load_job, list_jobs) with JSON serialization to /data/sessions/{session_id}/checkpoints/
- [ ] T010 [P] Add pattern persistence methods to StorageService (save_pattern, load_pattern, list_patterns) with JSON serialization to /data/sessions/{session_id}/patterns/

### VLM Service Extensions (Foundational)

- [ ] T011 Extend VLMService in src/services/vlm_service.py with query_regions_parallel() method using ThreadPoolExecutor (10 workers default) per research.md
- [ ] T012 Add exponential backoff retry wrapper to VLMService with base_delay=1s, max_delay=60s, jitter=±25%, max_retries=5 per research.md
- [ ] T013 Add proactive rate limiting to VLMService (rpm_limit configurable, default 450 for Tier 1) per research.md

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Automatic Semantic Labeling of All Regions (Priority: P1) 🎯 MVP

**Goal**: Enable automatic VLM querying of ALL segmented regions in a frame, regardless of segmentation quality, to detect semantic uncertainty

**Independent Test**: Process a video frame with 10 regions (mix of high/low quality), verify all 10 regions queued for VLM labeling and receive semantic labels or VLM_UNCERTAIN status

### Tests for User Story 1 (TDD - Write First)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T014 [P] [US1] Write integration test for automatic region queuing in tests/integration/test_semantic_labeling.py::test_all_regions_queued_regardless_of_quality (Scenario 1.1 from quickstart.md)
- [ ] T015 [P] [US1] Write integration test for high-quality regions sent to VLM in tests/integration/test_semantic_labeling.py::test_high_quality_regions_sent_to_vlm (Scenario 1.2)
- [ ] T016 [P] [US1] Write integration test for VLM_UNCERTAIN detection in tests/integration/test_semantic_labeling.py::test_vlm_uncertain_flagged_for_manual_labeling (Scenario 1.3)
- [ ] T017 [P] [US1] Write unit test for job creation in tests/unit/test_semantic_labeling_job.py::test_create_job_with_valid_params
- [ ] T018 [P] [US1] Write unit test for region queue building in tests/unit/test_semantic_labeling_job.py::test_build_pending_regions_queue

### Implementation for User Story 1

- [ ] T019 [US1] Implement SemanticLabelingService.create_job() in src/services/semantic_labeling_service.py to create job in PENDING status with session validation
- [ ] T020 [US1] Implement SemanticLabelingService.start_job() to transition job to RUNNING and begin region processing per contracts/semantic_labeling_api.yaml POST /jobs/{jobId}/start
- [ ] T021 [US1] Implement region queue builder in SemanticLabelingService._build_pending_regions_queue() to enumerate all regions across sampled frames (ignoring predicted_iou thresholds)
- [ ] T022 [US1] Implement VLM query loop in SemanticLabelingService._process_regions() using VLMService.query_regions_parallel() with progress updates after each region
- [ ] T023 [US1] Implement VLM_UNCERTAIN detection logic in SemanticLabelingService._evaluate_vlm_confidence() comparing response confidence to job.confidence_threshold (default 0.5)
- [ ] T024 [US1] Update SegmentedRegion with VLM results in SemanticLabelingService._update_region_with_label() setting semantic_label and vlm_query status
- [ ] T025 [US1] Implement atomic checkpointing after each region in SemanticLabelingService._checkpoint_progress() using StorageService.checkpoint_labeling_job()
- [ ] T026 [US1] Add job status endpoint GET /jobs/{jobId} to src/gui/app.py (or API layer) returning JobResponse per contracts/semantic_labeling_api.yaml

**Checkpoint**: At this point, User Story 1 should be fully functional - all regions automatically labeled with VLM_UNCERTAIN detection

---

## Phase 4: User Story 2 - Cost-Controlled Batch Processing (Priority: P2)

**Goal**: Provide cost control mechanisms including budget limits, cost estimation, frame sampling, and pause/resume to make automatic labeling practical for real-world use

**Independent Test**: Set $5 budget, process video with 100 frames/10 regions per frame, verify system pauses at 95% budget consumption with accurate cost tracking

### Tests for User Story 2 (TDD - Write First)

- [ ] T027 [P] [US2] Write integration test for cost estimation in tests/integration/test_semantic_labeling.py::test_cost_estimation_within_10_percent_accuracy (Scenario 2.1 from quickstart.md)
- [ ] T028 [P] [US2] Write integration test for budget limit auto-pause in tests/integration/test_semantic_labeling.py::test_budget_limit_auto_pause_at_95_percent (Scenario 2.2)
- [ ] T029 [P] [US2] Write integration test for frame sampling cost reduction in tests/integration/test_semantic_labeling.py::test_frame_sampling_reduces_cost_proportionally (Scenario 2.3)
- [ ] T030 [P] [US2] Write integration test for pause/resume in tests/integration/test_semantic_labeling.py::test_pause_resume_preserves_state (Scenario 2.4)
- [ ] T031 [P] [US2] Write unit test for cost calculation in tests/unit/test_cost_tracking.py::test_calculate_cost_per_region
- [ ] T032 [P] [US2] Write unit test for budget enforcement in tests/unit/test_cost_tracking.py::test_budget_limit_enforcement

### Implementation for User Story 2

- [ ] T033 [US2] Implement CostTrackingService in src/services/cost_tracking_service.py with real-time cost accumulation and budget monitoring per data-model.md
- [ ] T034 [US2] Add cost estimation logic in CostTrackingService.estimate_job_cost() using 15-20% stratified frame sampling per research.md
- [ ] T035 [US2] Implement tracking rate detection in CostTrackingService._analyze_scene_stability() using CV-based scene classification (stable/moderate/dynamic) per research.md
- [ ] T036 [US2] Add budget limit check in SemanticLabelingService._process_regions() calling CostTrackingService.check_budget_limit() before each query
- [ ] T037 [US2] Implement auto-pause at 95% budget in SemanticLabelingService._handle_budget_limit() transitioning job to PAUSED_BUDGET_LIMIT status
- [ ] T038 [US2] Add frame sampling logic in SemanticLabelingService._build_pending_regions_queue() filtering frames by job.frame_sampling parameter (every Nth frame)
- [ ] T039 [US2] Implement pause endpoint POST /jobs/{jobId}/pause in SemanticLabelingService.pause_job() with graceful shutdown (waits for current query) per contracts/
- [ ] T040 [US2] Implement resume endpoint POST /jobs/{jobId}/resume in SemanticLabelingService.resume_job() with idempotent queue rebuild per research.md
- [ ] T041 [US2] Add cost estimate endpoint POST /jobs/{jobId}/estimate to src/gui/app.py calling CostTrackingService.estimate_job_cost() per contracts/semantic_labeling_api.yaml
- [ ] T042 [US2] Create cost monitoring widget in src/gui/components/cost_monitor.py displaying real-time total_cost, budget_consumed_percentage, estimated_remaining_cost

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently - automatic labeling with full cost control

---

## Phase 5: User Story 3 - Semantic Uncertainty Pattern Detection (Priority: P3)

**Goal**: Analyze VLM_UNCERTAIN regions across frames to identify visual similarity patterns, enabling batch manual labeling and identification of systematic VLM struggles

**Independent Test**: Process video with construction equipment (VLM struggles), verify system clusters similar uncertain regions into patterns ranked by frequency

### Tests for User Story 3 (TDD - Write First)

- [ ] T043 [P] [US3] Write integration test for clustering in tests/integration/test_semantic_labeling.py::test_cluster_vlm_uncertain_regions_by_similarity (Scenario 3.1 from quickstart.md)
- [ ] T044 [P] [US3] Write integration test for pattern dashboard in tests/integration/test_semantic_labeling.py::test_pattern_analysis_dashboard_ranked_by_frequency (Scenario 3.2)
- [ ] T045 [P] [US3] Write integration test for batch labeling in tests/integration/test_semantic_labeling.py::test_batch_manual_labeling_for_pattern (Scenario 3.3)
- [ ] T046 [P] [US3] Write unit test for similarity computation in tests/unit/test_clustering.py::test_compute_heuristic_similarity
- [ ] T047 [P] [US3] Write unit test for DBSCAN clustering in tests/unit/test_clustering.py::test_dbscan_clusters_similar_regions

### Implementation for User Story 3

- [ ] T048 [P] [US3] Implement SimilarityService in src/services/similarity_service.py with heuristic similarity computation (bbox 0.3, mask 0.4, color 0.3) per research.md
- [ ] T049 [P] [US3] Add bbox similarity calculation in SimilarityService._compute_bbox_similarity() using area and aspect ratio comparison
- [ ] T050 [P] [US3] Add mask similarity calculation in SimilarityService._compute_mask_similarity() using IoU after 128×128 resize
- [ ] T051 [P] [US3] Add color histogram similarity in SimilarityService._compute_color_similarity() using RGB correlation (32 bins/channel) per research.md
- [ ] T052 [US3] Implement ClusteringService in src/services/clustering_service.py with DBSCAN clustering (eps=0.3, min_samples=2) per research.md
- [ ] T053 [US3] Add pattern detection endpoint GET /jobs/{jobId}/patterns in ClusteringService.detect_patterns() computing clusters on-demand from VLM_UNCERTAIN regions
- [ ] T054 [US3] Implement batch label endpoint POST /patterns/{patternId}/label in ClusteringService.label_pattern() updating all regions in cluster to CONFIRMED status
- [ ] T055 [US3] Create uncertainty patterns widget in src/gui/components/uncertainty_patterns.py displaying top patterns with region counts, frames affected, and sample images
- [ ] T056 [US3] Add pattern visualization in uncertainty_patterns.py showing up to 5 sample images per pattern for visual inspection

**Checkpoint**: All user stories should now be independently functional - complete automatic labeling with cost controls and pattern detection

---

## Phase 6: Region Tracking Optimization (Cross-Cutting Enhancement)

**Goal**: Implement region tracking across consecutive frames to skip redundant VLM queries, achieving 60-70% cost reduction for stable scenes

**Independent Test**: Process stable scene video with/without tracking, verify 60-70% cost reduction with tracking enabled

### Tests for Region Tracking (TDD - Write First)

- [ ] T057 [P] Write unit test for IoU tracking in tests/unit/test_region_tracker.py::test_high_iou_match_tracks_region
- [ ] T058 [P] Write unit test for occlusion handling in tests/unit/test_region_tracker.py::test_occlusion_tolerance_3_frames
- [ ] T059 [P] Write unit test for split detection in tests/unit/test_region_tracker.py::test_split_regions_queried_separately
- [ ] T060 [P] Write integration test for tracking cost savings in tests/integration/test_semantic_labeling.py::test_region_tracking_achieves_60_percent_cost_reduction (Performance benchmark from quickstart.md)

### Implementation for Region Tracking

- [ ] T061 Implement RegionTracker in src/services/region_tracker.py with three-stage matching pipeline (bbox IoU ≥0.5, centroid distance, mask IoU ≥0.7) per research.md
- [ ] T062 Add tracking integration in SemanticLabelingService._process_frame() calling RegionTracker.track_regions() to identify matched pairs
- [ ] T063 Skip VLM queries for tracked regions in SemanticLabelingService._process_regions() copying semantic_label from parent_region_id for tracked regions
- [ ] T064 Update TrackedRegion fields in RegionTracker._mark_as_tracked() setting tracking_status="tracked", parent_region_id, tracked_iou, semantic_label_source="tracked"
- [ ] T065 Handle occlusion in RegionTracker._handle_occlusion() with 3-frame tolerance before marking region as ended
- [ ] T066 Handle splits/merges in RegionTracker._handle_topology_changes() querying VLM for all split/merged regions (labels may differ)

**Checkpoint**: Region tracking optimization complete - significant cost savings for stable scenes

---

## Phase 7: Contract & API Validation

**Purpose**: Validate all API endpoints match contracts/semantic_labeling_api.yaml specification

- [ ] T067 [P] Write contract test for POST /jobs in tests/contract/test_semantic_labeling_api.py::test_create_job_contract
- [ ] T068 [P] Write contract test for GET /jobs/{jobId} in tests/contract/test_semantic_labeling_api.py::test_get_job_status_contract
- [ ] T069 [P] Write contract test for POST /jobs/{jobId}/start in tests/contract/test_semantic_labeling_api.py::test_start_job_contract
- [ ] T070 [P] Write contract test for POST /jobs/{jobId}/pause in tests/contract/test_semantic_labeling_api.py::test_pause_job_contract
- [ ] T071 [P] Write contract test for POST /jobs/{jobId}/resume in tests/contract/test_semantic_labeling_api.py::test_resume_job_contract
- [ ] T072 [P] Write contract test for POST /jobs/{jobId}/cancel in tests/contract/test_semantic_labeling_api.py::test_cancel_job_contract
- [ ] T073 [P] Write contract test for POST /jobs/{jobId}/estimate in tests/contract/test_semantic_labeling_api.py::test_estimate_cost_contract
- [ ] T074 [P] Write contract test for GET /jobs/{jobId}/patterns in tests/contract/test_semantic_labeling_api.py::test_list_patterns_contract
- [ ] T075 [P] Write contract test for POST /patterns/{patternId}/label in tests/contract/test_semantic_labeling_api.py::test_label_pattern_contract
- [ ] T076 Implement cancel endpoint POST /jobs/{jobId}/cancel in SemanticLabelingService.cancel_job() per contracts/semantic_labeling_api.yaml

**Checkpoint**: All contracts validated - API matches specification

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories and final validation

- [ ] T077 [P] Add comprehensive logging for job lifecycle events in SemanticLabelingService (created, started, paused, resumed, completed, failed)
- [ ] T078 [P] Add error handling for VLM API failures in VLMService with VLM_FAILED status distinction per FR-009
- [ ] T079 [P] Add validation for job configuration parameters (budget_limit > 0, frame_sampling >= 1, confidence_threshold 0.0-1.0, tracking_iou_threshold 0.0-1.0)
- [ ] T080 [P] Add progress percentage calculation in SemanticLabelingService updating job.progress.progress_percentage after each region
- [ ] T081 [P] Add estimated_remaining_cost calculation in CostTrackingService based on average_cost_per_region and regions_pending
- [ ] T082 [P] Add warning thresholds in CostTrackingService.estimate_job_cost() for high region count (>30/frame), budget risk (>80%), high variance (CV >40%) per research.md
- [ ] T083 [P] Optimize checkpoint frequency in SemanticLabelingService for high region counts (switch to every-5-regions for 100+ regions/frame) per research.md
- [ ] T084 Run all quickstart.md test scenarios for end-to-end validation
- [ ] T085 Run performance benchmarks from quickstart.md (region tracking 60-70% savings, cost estimation ±10%, frame sampling proportional reduction)
- [ ] T086 Code review for constitution compliance (TDD cycle, simplicity/YAGNI, no premature optimization)
- [ ] T087 Update documentation in specs/002-semantic-labeling-all-regions/ with any implementation learnings

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phases 3-5)**: All depend on Foundational phase completion
  - User Story 1 (P1): Can start after Foundational - No dependencies on other stories
  - User Story 2 (P2): Can start after Foundational - No dependencies on other stories (integrates with US1 but independently testable)
  - User Story 3 (P3): Can start after Foundational - No dependencies on other stories (builds on VLM_UNCERTAIN from US1 but independently testable)
- **Region Tracking (Phase 6)**: Can start after Foundational - Cross-cuts all stories but independently testable
- **Contract Validation (Phase 7)**: Can start after User Stories 1-3 implementation complete
- **Polish (Phase 8)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Independent - Can start after Foundational (Phase 2)
- **User Story 2 (P2)**: Independent - Can start after Foundational (Phase 2)
- **User Story 3 (P3)**: Independent - Can start after Foundational (Phase 2)

**Note**: All user stories are designed to be independently implementable and testable after Foundational phase completes.

### Within Each User Story

- Tests MUST be written and FAIL before implementation (TDD Red-Green-Refactor)
- Models before services (data structures before logic)
- Services before endpoints (business logic before API)
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- **Setup (Phase 1)**: T002, T003, T004 can run in parallel
- **Foundational (Phase 2)**:
  - Data models (T005, T006, T007) can run in parallel
  - Storage methods (T009, T010) can run in parallel after T008
- **User Story 1 Tests**: T014, T015, T016, T017, T018 can run in parallel
- **User Story 2 Tests**: T027, T028, T029, T030, T031, T032 can run in parallel
- **User Story 3 Tests**: T043, T044, T045, T046, T047 can run in parallel
- **User Story 3 Similarity**: T048, T049, T050, T051 can run in parallel
- **Region Tracking Tests**: T057, T058, T059, T060 can run in parallel
- **Contract Tests**: T067-T075 can all run in parallel
- **Polish Tasks**: T077, T078, T079, T080, T081, T082, T083 can run in parallel
- **After Foundational completes**: All user stories (Phases 3-5) can start in parallel if team capacity allows

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together (TDD - Write First):
Task T014: "Write integration test for automatic region queuing in tests/integration/test_semantic_labeling.py"
Task T015: "Write integration test for high-quality regions sent to VLM in tests/integration/test_semantic_labeling.py"
Task T016: "Write integration test for VLM_UNCERTAIN detection in tests/integration/test_semantic_labeling.py"
Task T017: "Write unit test for job creation in tests/unit/test_semantic_labeling_job.py"
Task T018: "Write unit test for region queue building in tests/unit/test_semantic_labeling_job.py"

# All tests should FAIL at this point (Red phase)

# Implementation proceeds sequentially due to dependencies (Green phase)
```

---

## Parallel Example: User Story 3 (Similarity Computation)

```bash
# Launch all similarity calculation methods together:
Task T049: "Add bbox similarity calculation in SimilarityService"
Task T050: "Add mask similarity calculation in SimilarityService"
Task T051: "Add color histogram similarity in SimilarityService"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T004)
2. Complete Phase 2: Foundational (T005-T013) - CRITICAL, blocks all stories
3. Complete Phase 3: User Story 1 (T014-T026)
4. **STOP and VALIDATE**: Test User Story 1 independently using quickstart.md Scenarios 1.1-1.3
5. Deploy/demo if ready

**MVP Deliverable**: Automatic semantic labeling of all regions with VLM_UNCERTAIN detection

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready (T001-T013)
2. Add User Story 1 → Test independently → Deploy/Demo (MVP: Automatic labeling) (T014-T026)
3. Add User Story 2 → Test independently → Deploy/Demo (Cost controls) (T027-T042)
4. Add User Story 3 → Test independently → Deploy/Demo (Pattern detection) (T043-T056)
5. Add Region Tracking → Test independently → Deploy/Demo (Cost optimization) (T057-T066)
6. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together (T001-T013)
2. Once Foundational is done:
   - Developer A: User Story 1 (T014-T026)
   - Developer B: User Story 2 (T027-T042)
   - Developer C: User Story 3 (T043-T056)
   - Developer D: Region Tracking (T057-T066)
3. Stories complete and integrate independently
4. Team collaborates on Contract Validation (T067-T076)
5. Team completes Polish together (T077-T087)

---

## Notes

- **TDD Red-Green-Refactor**: All test tasks MUST be written first and FAIL before implementation
- **[P] tasks**: Different files, no dependencies - can run in parallel
- **[Story] label**: Maps task to specific user story for traceability
- **Each user story**: Independently completable and testable after Foundational phase
- **Constitution compliance**: Zero new dependencies, simple heuristics, atomic checkpointing, clear upgrade paths
- **Commit discipline**: Commit after each task or logical group with passing tests
- **Checkpoints**: Stop at any checkpoint to validate story independently per quickstart.md scenarios
- **Avoid**: Vague tasks, same file conflicts, cross-story dependencies that break independence

---

## Summary

**Total Tasks**: 87 tasks
- **Phase 1 (Setup)**: 4 tasks
- **Phase 2 (Foundational)**: 9 tasks (BLOCKING)
- **Phase 3 (User Story 1 - P1)**: 13 tasks (5 tests + 8 implementation) - MVP
- **Phase 4 (User Story 2 - P2)**: 16 tasks (6 tests + 10 implementation)
- **Phase 5 (User Story 3 - P3)**: 14 tasks (5 tests + 9 implementation)
- **Phase 6 (Region Tracking)**: 10 tasks (4 tests + 6 implementation)
- **Phase 7 (Contract Validation)**: 10 tasks (9 tests + 1 implementation)
- **Phase 8 (Polish)**: 11 tasks

**Parallel Opportunities**:
- Setup: 3 parallel tasks
- Foundational: 5 parallel opportunities (within phase dependencies)
- User Stories: All 3 stories + region tracking can run in parallel after Foundational (if staffed)
- Tests: 30+ test tasks can run in parallel within their phases
- Contract tests: 9 parallel tasks
- Polish: 7 parallel tasks

**Independent Test Criteria**:
- **User Story 1**: Process 10-region frame, verify all queued and labeled (Scenarios 1.1-1.3)
- **User Story 2**: Set budget, verify pause at 95%, estimate within 10% (Scenarios 2.1-2.4)
- **User Story 3**: Process construction video, verify pattern clustering (Scenarios 3.1-3.3)
- **Region Tracking**: Compare costs with/without tracking, verify 60-70% reduction (Performance benchmark)

**Suggested MVP Scope**: User Story 1 only (T001-T026) - Automatic labeling with uncertainty detection

**Format Validation**: ✅ All tasks follow checklist format with checkboxes, IDs, labels, and file paths
