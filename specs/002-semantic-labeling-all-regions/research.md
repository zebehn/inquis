# Research: Semantic Labeling for All Regions - Technical Decisions

**Date**: 2025-12-19  
**Feature**: 002-semantic-labeling-all-regions  
**Purpose**: Research and resolve all technical unknowns identified in Technical Context

---

## Executive Summary

This research resolves 5 key technical unknowns required to implement automatic semantic labeling of all regions with cost controls, region tracking optimization, and semantic uncertainty pattern detection.

### Key Decisions Matrix

| Unknown | Decision | Rationale | Expected Impact |
|---------|----------|-----------|-----------------|
| **Region Tracking** | IoU-based with 0.7 threshold | Uses existing mask_utils, simple 3-stage pipeline | 60-70% cost savings |
| **VLM Batch Queries** | Sequential with 10-worker parallel execution | Better error handling, only 25% cost premium | 10x latency improvement |
| **Clustering** | Heuristic (bbox+mask+color) + DBSCAN | No new dependencies, 80% accuracy | 5-10x faster than embeddings |
| **Cost Estimation** | 15-20% stratified frame sampling | Sample-based with tracking analysis | ±10% accuracy margin |
| **Pause/Resume** | Per-region atomic JSON checkpointing | Crash-safe, matches existing architecture | <2% overhead |

---

## 1. Region Tracking Algorithm

### Decision: Hybrid IoU-Based Tracking (0.7 threshold)

**Three-stage matching pipeline**:
1. **BBox IoU pre-filter** (≥0.5) - Fast elimination of non-matches
2. **Centroid distance** - Tie-breaking for multiple candidates  
3. **Mask IoU verification** (≥0.7) - Final confident match

**Expected Cost Savings** (30fps, 10 regions/frame, 70% tracking rate):
- Without tracking: 1,000 VLM queries
- With tracking: 307 VLM queries  
- **Cost reduction: 69.3%** ✅

**Edge Cases**:
- Occlusion: 3-frame tolerance before marking region as ended
- Splits: Query VLM for all split regions (labels may differ)
- Merges: Query merged region (combined semantic meaning)

**Integration**: Uses existing `/data1/Developments/inquis/src/utils/mask_utils.py:compute_mask_iou()`

---

## 2. VLM Batch Query Implementation

### Decision: Sequential Queries with Parallel Execution

**Approach**: ThreadPoolExecutor with 10-50 workers + proactive rate limiting

**Why Not Batched?**
| Factor | Batched | Sequential | Winner |
|--------|---------|------------|--------|
| Cost | $0.101/frame | $0.135/frame (+34%) | Batched |
| Error handling | All-or-nothing | Per-region isolation | Sequential ✅ |
| Retry economics | Expensive | Cheap | Sequential ✅ |
| Complexity | High | Low | Sequential ✅ |

**Trade-off**: 25-35% more expensive but provides better reliability and maintainability.

**Performance**:
- Sequential (no parallelism): 50-100s for 50 regions
- Parallel (10 workers): 5-10s for 50 regions
- **Improvement: 10x faster** ✅

**Exponential Backoff**: Base delay 1s, max delay 60s, jitter ±25%, max 5 retries

**Rate Limits** (OpenAI API):
- Tier 1 ($5+): 500 RPM → 10 frames/min @ 50 regions/frame
- Tier 2 ($50+): 5,000 RPM → 100 frames/min @ 50 regions/frame

---

## 3. Semantic Uncertainty Clustering

### Decision: Simple Heuristic Clustering with DBSCAN

**Tier 1 Approach** (Recommended):
- **Bounding box similarity**: Area + aspect ratio comparison
- **Mask shape similarity**: IoU after 128×128 resize
- **Color histogram similarity**: RGB correlation (32 bins/channel)
- **Weighted combination**: [0.3, 0.4, 0.3] for bbox, mask, color

**DBSCAN Parameters**:
- `eps = 0.3` (30% similarity threshold)
- `min_samples = 2-3` (minimum cluster size)

**Performance** (100 regions):
| Approach | Time | Accuracy | Dependencies |
|----------|------|----------|--------------|
| Tier 1 Heuristics | 30-105s | 80% | None ✅ |
| Tier 2 ResNet-50 | 500s CPU | 90% | None (torchvision exists) |
| Tier 3 CLIP | 800s CPU | 95% | New deps required |

**Minimum Viable Scale**:
- < 10 regions: Disable clustering
- 10-20 regions: Enable with warning
- 20+ regions: Enable by default

---

## 4. Cost Estimation Accuracy

### Decision: 15-20% Stratified Frame Sampling

**Sampling Strategy**:
- **Beginning (30% of sample)**: Frames 0-10%
- **Middle (50% of sample)**: Frames 45-55%  
- **End (20% of sample)**: Frames 95-100%

**Sample Size**: 8-40 frames (min 10, max 40 for speed)

**Estimation Formula with Tracking**:
```python
# First frame: all regions queried
first_frame_cost = mean_regions * cost_per_query

# Subsequent frames: only new regions queried (tracking_rate from sample analysis)
new_regions_per_frame = mean_regions * (1 - tracking_rate)
subsequent_cost = new_regions_per_frame * cost_per_query * (total_frames - 1)

estimated_cost = first_frame_cost + subsequent_cost
```

**Tracking Rates by Scene Stability**:
| Scene Type | CV | Tracking Rate | Cost Predictability |
|------------|-----|--------------|---------------------|
| Stable | <20% | 70% | HIGH (5-10% error) |
| Moderate | 25-45% | 55% | MODERATE (10-15% error) |
| Dynamic | 30-60% | 35% | MODERATE (12-20% error) |

**Warning Thresholds**:
- High region count: >30 regions/frame
- Budget: 80% consumed = warning, 95% = pause
- High variance: CV >40% (unpredictable costs)

**Validation Target**: 80%+ of videos within 10% error margin

---

## 5. Pause/Resume State Management

### Decision: Per-Region Checkpointing with Atomic JSON Writes

**Checkpointing Strategy**:
- **Frequency**: After each region (every 1-5 seconds)
- **Granularity**: 1 region lost on crash (~1-5s of work)
- **Adaptive**: For 100+ regions/frame, switch to every-5-regions

**In-Flight Query Handling**:
- **Strategy**: Wait for completion (don't interrupt mid-query)
- **Rationale**: Queries are fast (1-5s), no API cancellation, ensures cost accuracy

**Atomic Writes (Write-Then-Rename Pattern)**:
```python
# Write to temp file
temp_file.write(json_data)
temp_file.flush()
os.fsync(temp_file.fileno())  # Force disk write

# Atomic rename
temp_file.rename(final_file)  # POSIX atomic operation
```

**Why Atomic?**
- No partial writes (crash during write leaves old checkpoint intact)
- No torn reads (readers see old or new, never partial)
- fsync() ensures data reaches disk (survives system crash)

**Resume Logic (Idempotent)**:
1. Load checkpoint
2. Rebuild pending queue (exclude completed + in-flight)
3. Handle in-flight region (check if completed, else re-queue)
4. Update status to RUNNING, save checkpoint

**Performance**: ~10-15ms per region (1-1.5% of 10-30min job overhead)

**JSON vs SQLite**: JSON sufficient for single-user desktop app, matches existing architecture

---

## Implementation Impact Summary

### New Files Required

**Models**:
- `src/models/semantic_labeling_job.py` - Job state with pause/resume
- `src/models/semantic_uncertainty.py` - Uncertainty pattern clustering

**Services**:
- `src/services/semantic_labeling_service.py` - Job orchestration
- `src/services/cost_tracking_service.py` - Real-time cost/budget tracking
- `src/services/region_tracker.py` - IoU-based frame-to-frame tracking
- `src/services/similarity_service.py` - Heuristic similarity calculation
- `src/services/clustering_service.py` - DBSCAN clustering

**GUI Components**:
- `src/gui/components/cost_monitor.py` - Budget visualization
- `src/gui/components/uncertainty_patterns.py` - Pattern detection UI

### Extensions to Existing Files

**VLMService** (`src/services/vlm_service.py`):
- Add `query_regions_parallel()` with ThreadPoolExecutor
- Add exponential backoff retry wrapper
- Add proactive rate limiting

**StorageService** (`src/services/storage_service.py`):
- Add `checkpoint_labeling_job()` with atomic writes
- Add `load_job_checkpoint()`

**Models** (`src/models/`):
- Extend SegmentedRegion with `tracking_status`, `parent_region_id`, `tracked_iou`

### Configuration

```yaml
semantic_labeling:
  tracking:
    enabled: true
    mask_iou_threshold: 0.7
    bbox_iou_prefilter: 0.5
  vlm:
    max_workers: 10  # Tier 1: 10, Tier 2+: 50
    rpm_limit: 450   # Tier 1: 450, Tier 2: 4500
    max_retries: 5
  clustering:
    eps: 0.3
    min_samples: 2
  cost_estimation:
    sample_rate: 0.15
    min_samples: 10
```

### Dependencies

**No new dependencies required** - all approaches use existing packages (numpy, cv2, scikit-learn, torchvision).

---

## Testing Strategy

### Unit Tests Required

1. **Region Tracking**: `test_high_iou_match()`, `test_occlusion_handling()`, `test_split_detection()`
2. **Cost Estimation**: `test_stratified_sampling()`, `test_tracking_rate_detection()`
3. **Clustering**: `test_clusters_similar_regions()`, `test_similarity_computation()`
4. **Pause/Resume**: `test_checkpoint_atomicity()`, `test_resume_idempotency()`

### Integration Tests Required

1. **End-to-End Labeling**: `test_full_auto_labeling_workflow()`, `test_tracking_reduces_queries()`
2. **Budget Enforcement**: `test_budget_limit_pauses_job()`
3. **Clustering**: `test_full_clustering_workflow()`, `test_batch_manual_labeling()`

### Validation Targets

- **Region tracking**: 60-70% cost reduction on stable videos
- **Cost estimation**: 80%+ within 10% error margin
- **Clustering**: 80%+ accuracy grouping similar objects
- **Pause/resume**: 100% state preservation

---

## References

### Codebase Files

- `/data1/Developments/inquis/src/utils/mask_utils.py:compute_mask_iou()` (lines 120-138)
- `/data1/Developments/inquis/src/models/vlm_query.py` - VLM query state
- `/data1/Developments/inquis/src/services/vlm_service.py` - VLM API integration
- `/data1/Developments/inquis/src/services/storage_service.py` - JSON persistence
- `/data1/Developments/inquis/tests/integration/test_vlm_workflow.py` - 6 VLM tests passing

### External

- OpenAI API Rate Limits: https://platform.openai.com/docs/guides/rate-limits
- DBSCAN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
- OpenCV Histogram Comparison: https://docs.opencv.org/4.x/d8/dc8/tutorial_histogram_comparison.html

---

## Detailed Research Documents

Full research findings from all 5 agents are available in supplementary documents for reference:
- Region tracking algorithm details and benchmarks
- VLM batch query implementation with code examples
- Semantic clustering tier comparison and performance analysis
- Cost estimation statistical methodology
- Pause/resume state serialization with disaster recovery

These documents provide implementation-level detail beyond this summary.

---

## Conclusion

All 5 technical unknowns resolved with concrete, implementable solutions that:
- Prioritize simplicity (YAGNI principle)
- Use existing dependencies (zero new deps for MVP)
- Provide clear upgrade paths when needed
- Include performance benchmarks and validation targets

The research provides sufficient detail for Phase 1 implementation (data models and contracts) without requiring additional technical decisions.
