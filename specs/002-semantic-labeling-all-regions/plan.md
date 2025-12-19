# Implementation Plan: Semantic Labeling for All Regions

**Branch**: `002-semantic-labeling-all-regions` | **Date**: 2025-12-19 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-semantic-labeling-all-regions/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

This feature extends the VLM-assisted labeling system to automatically query ALL segmented regions for semantic classification, not just uncertain regions. This enables detection of semantic uncertainty—situations where segmentation quality is high but the VLM cannot confidently identify the object. The implementation includes cost control mechanisms (budget limits, frame sampling, real-time tracking), region tracking optimization to skip redundant queries across consecutive frames (50-70% cost reduction), and semantic uncertainty pattern detection through clustering.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: OpenAI SDK (VLM integration), existing VLMService, StorageService, Streamlit (GUI)
**Storage**: File-based JSON storage (existing StorageService) for labeling jobs, VLM queries, semantic patterns
**Testing**: pytest (existing test infrastructure)
**Target Platform**: Linux/macOS desktop (Streamlit GUI application)
**Project Type**: Single project (extends existing src/ structure)
**Performance Goals**: Handle 100+ regions per frame with batch VLM queries; real-time cost tracking with <100ms update latency
**Constraints**: VLM API rate limits (exponential backoff required); cost awareness (budget controls mandatory); pause/resume state persistence
**Scale/Scope**: Process videos with 50-200 frames, 5-50 regions per frame; track cumulative costs across sessions; cluster 100+ uncertain regions for pattern detection

**Key Unknowns for Research**:
- NEEDS CLARIFICATION: Region tracking algorithm (how to determine "nearly identical regions" across frames - IoU threshold? mask similarity? centroid distance?)
- NEEDS CLARIFICATION: VLM batch query implementation (send all regions in one prompt vs sequential queries - API limits, cost optimization, error handling)
- NEEDS CLARIFICATION: Semantic uncertainty clustering approach (visual embeddings? bounding box + mask similarity? CLIP features?)
- NEEDS CLARIFICATION: Cost estimation accuracy (how to predict regions per frame before processing? sample-based estimation?)
- NEEDS CLARIFICATION: Pause/resume state serialization (what job state must be persisted? how to handle in-flight VLM queries?)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Pre-Research Check (Phase 0 Gate)

| Principle | Status | Notes |
|-----------|--------|-------|
| **I. TDD Cycle** | ✅ PASS | Feature will follow Red-Green-Refactor for all new components (SemanticLabelingJob, batch processing, clustering) |
| **II. Test-First** | ✅ PASS | All tests will be written before implementation; integration tests for each user story acceptance scenario |
| **III. Tidy First** | ✅ PASS | Structural changes (refactoring existing VLMService) committed separately from behavioral changes (new auto-labeling) |
| **IV. Code Quality** | ✅ PASS | Extends existing patterns (VLMService, StorageService); no new architectural layers |
| **V. Refactoring Discipline** | ✅ PASS | Will refactor only when tests pass; one refactoring at a time |
| **VI. Commit Discipline** | ✅ PASS | Commits only with passing tests; proper prefixes (feat:/refactor:/tidy:) |
| **VII. Simplicity/YAGNI** | ✅ PASS | Start with simplest clustering (IoU + mask similarity), upgrade only if needed; no premature optimization |

**Result**: ✅ ALL GATES PASSED - Proceed to Phase 0 research

### Post-Design Check (Phase 1 Gate)

*Completed after research.md, data-model.md, quickstart.md, and contracts/ generated*

| Principle | Status | Notes |
|-----------|--------|-------|
| **I. TDD Cycle** | ✅ PASS | All test scenarios defined in quickstart.md with clear Red-Green-Refactor expectations; 15+ test scenarios across 3 user stories |
| **II. Test-First** | ✅ PASS | Test Execution Checklist in quickstart.md enforces writing failing tests before implementation; each scenario includes validation criteria |
| **III. Tidy First** | ✅ PASS | Design preserves separation: VLMService extensions (structural) committed separately from new SemanticLabelingJob (behavioral) |
| **IV. Code Quality** | ✅ PASS | Uses Pydantic for validation; extends existing patterns (StorageService, VLMService); no architectural violations |
| **V. Refactoring Discipline** | ✅ PASS | Clear upgrade paths: Tier 1 heuristics → Tier 2 embeddings only when needed; no premature abstractions |
| **VI. Commit Discipline** | ✅ PASS | quickstart.md enforces test-passing commits; integration tests validate each phase completion |
| **VII. Simplicity/YAGNI** | ✅ PASS | Zero new dependencies for MVP; simple heuristics (bbox+mask+color) before embeddings; atomic JSON checkpointing sufficient for single-user desktop app |

**Result**: ✅ ALL GATES PASSED - Design maintains constitution compliance

**Justification for Complexity**:
- **3 new entities** (SemanticLabelingJob, TrackedRegion, SemanticUncertaintyPattern) - Required for pause/resume state, tracking optimization, and pattern detection (core FRs)
- **5 new services** - Each addresses distinct concern: job orchestration, cost tracking, region tracking, similarity computation, clustering
- **No new dependencies** - Leverages existing cv2, numpy, scikit-learn, OpenAI SDK, Streamlit
- **Clear upgrade paths** - Tier 1 (heuristics) → Tier 2 (ResNet-50) → Tier 3 (CLIP) only when accuracy requirements demand it
- **Atomic checkpointing** - Matches existing JSON storage pattern; no premature optimization to database

**Constitution Compliance Summary**:
- Follows existing architecture (no new layers)
- Test scenarios written before implementation
- Simple approaches with documented upgrade paths
- All unknowns resolved with concrete decisions
- Performance benchmarks and validation targets defined

## Project Structure

### Documentation (this feature)

```text
specs/002-semantic-labeling-all-regions/
├── spec.md              # Feature specification (completed)
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
│   └── semantic_labeling_api.yaml
├── checklists/
│   └── requirements.md  # Validation checklist (completed)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/
├── models/
│   ├── session.py                    # Existing (VideoSession, SegmentedRegion)
│   ├── vlm_query.py                  # Existing (VLMQuery, VLMQueryStatus)
│   ├── semantic_labeling_job.py      # NEW - labeling job state
│   └── semantic_uncertainty.py       # NEW - uncertainty pattern clustering
├── services/
│   ├── storage_service.py            # Existing - extend with job/pattern persistence
│   ├── vlm_service.py                # Existing - extend with batch queries, tracking
│   ├── semantic_labeling_service.py  # NEW - orchestrates auto-labeling workflow
│   └── cost_tracking_service.py      # NEW - real-time cost/budget management
└── gui/
    ├── app.py                         # Existing - add auto-labeling controls
    └── components/
        ├── cost_monitor.py            # NEW - budget visualization widget
        └── uncertainty_patterns.py    # NEW - pattern detection UI

tests/
├── integration/
│   ├── test_vlm_workflow.py          # Existing (6 tests passing)
│   └── test_semantic_labeling.py     # NEW - auto-labeling integration tests
├── unit/
│   ├── test_semantic_labeling_job.py # NEW - job state management tests
│   ├── test_cost_tracking.py         # NEW - budget/cost calculation tests
│   └── test_clustering.py            # NEW - uncertainty pattern tests
└── contract/
    └── test_semantic_labeling_api.py # NEW - API contract validation
```

**Structure Decision**: Single project structure extending existing `src/` layout. This feature builds on Phase 5 VLM implementation from feature 001, adding new services for job orchestration and cost tracking while extending existing VLMService and StorageService with batch processing and region tracking capabilities.

## Complexity Tracking

No constitution violations. This feature extends existing architecture without adding new layers or patterns.
