# Quickstart: Semantic Labeling for All Regions

**Feature**: 002-semantic-labeling-all-regions
**Date**: 2025-12-19
**Purpose**: Test scenarios and workflows for validating automatic semantic labeling with cost controls

---

## Prerequisites

### Environment Setup

```bash
# Ensure Python 3.11+ and dependencies installed
python --version  # Must be 3.11+

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Verify OpenAI API key configured
echo $OPENAI_API_KEY  # Must be set

# Verify test data available
ls /data/videos/test_dashcam.mp4  # Example test video
```

### Test Data Requirements

**Minimum Test Videos**:
- **Stable scene video** (30fps, 50 frames, 5-10 regions/frame) - for region tracking validation
- **Dynamic scene video** (30fps, 100 frames, 15-25 regions/frame) - for cost estimation and budget tests
- **High region count video** (30fps, 20 frames, 40+ regions/frame) - for cost warning validation

**Expected Region Characteristics**:
- Mix of high-quality segmentations (predicted_iou > 0.75) and low-quality segmentations
- At least 2-3 object types that trigger VLM_UNCERTAIN (e.g., construction equipment, specialized tools)
- Consecutive frames with tracked regions (IoU > 0.7) for tracking optimization validation

---

## User Story 1: Automatic Semantic Labeling of All Regions

### Test Scenario 1.1 - All Regions Queued Regardless of Quality

**Objective**: Verify all regions are automatically sent to VLM, not just uncertain ones

**Setup**:
```python
# Create session with test video
session = create_video_session(
    video_path="/data/videos/mixed_quality_regions.mp4",
    start_frame=0,
    end_frame=1  # Single frame test
)

# Process frame to get segmentations
regions = segment_frame(session_id=session.id, frame_idx=0)
# Expected: 10 regions (8 high-quality, 2 low-quality)
```

**Execution**:
```python
# Create automatic labeling job
job = create_labeling_job(
    session_id=session.id,
    video_path=session.video_path,
    enable_tracking=False  # Disable for single-frame test
)

# Start job
start_labeling_job(job_id=job.id)

# Wait for completion
wait_for_job_completion(job.id, timeout_seconds=60)
```

**Validation**:
```python
# Verify all regions labeled
job_status = get_job_status(job.id)
assert job_status.progress.regions_total == 10
assert job_status.progress.regions_completed == 10
assert job_status.status == "completed"

# Verify VLM queries executed for all regions (no skips)
assert job_status.cost_tracking.queries_successful >= 10
```

**Expected Result**: ✅ All 10 regions queued and labeled, regardless of segmentation quality

---

### Test Scenario 1.2 - High-Quality Regions Still Sent to VLM

**Objective**: Verify system sends high-quality segmentations to VLM to detect semantic uncertainty

**Setup**:
```python
# Create session with only high-quality segmentations
session = create_video_session(
    video_path="/data/videos/high_quality_segmentations.mp4",
    start_frame=0,
    end_frame=1
)

regions = segment_frame(session_id=session.id, frame_idx=0)
# Verify all regions have predicted_iou > 0.75
assert all(r.predicted_iou > 0.75 for r in regions)
```

**Execution**:
```python
job = create_labeling_job(session_id=session.id, video_path=session.video_path)
start_labeling_job(job_id=job.id)
wait_for_job_completion(job.id, timeout_seconds=60)
```

**Validation**:
```python
job_status = get_job_status(job.id)

# All high-quality regions still sent to VLM
assert job_status.progress.regions_completed == len(regions)

# Verify labels returned (mix of confident and uncertain)
labeled_regions = get_labeled_regions(session.id, frame_idx=0)
confident_labels = [r for r in labeled_regions if r.vlm_query.status == "COMPLETED"]
uncertain_labels = [r for r in labeled_regions if r.vlm_query.status == "VLM_UNCERTAIN"]

# Even high-quality segmentations may have uncertain semantics
assert len(confident_labels) + len(uncertain_labels) == len(regions)
```

**Expected Result**: ✅ High-quality segmentations are sent to VLM and semantic uncertainty detected

---

### Test Scenario 1.3 - VLM_UNCERTAIN Regions Flagged for Manual Labeling

**Objective**: Verify low-confidence VLM responses marked as VLM_UNCERTAIN

**Setup**:
```python
# Use video with objects VLM struggles with (construction equipment, specialized tools)
session = create_video_session(
    video_path="/data/videos/construction_equipment.mp4",
    start_frame=0,
    end_frame=5
)
```

**Execution**:
```python
job = create_labeling_job(
    session_id=session.id,
    video_path=session.video_path,
    confidence_threshold=0.5  # Mark regions with confidence < 0.5 as uncertain
)
start_labeling_job(job_id=job.id)
wait_for_job_completion(job.id, timeout_seconds=120)
```

**Validation**:
```python
job_status = get_job_status(job.id)

# Verify some regions marked as uncertain
assert job_status.cost_tracking.queries_uncertain > 0

# Retrieve uncertain regions
uncertain_regions = get_uncertain_regions(session.id)
for region in uncertain_regions:
    assert region.vlm_query.status == "VLM_UNCERTAIN"
    assert region.vlm_query.confidence < 0.5
    # Verify flagged for manual labeling
    assert region.requires_manual_label is True
```

**Expected Result**: ✅ Low-confidence regions marked VLM_UNCERTAIN and flagged for manual labeling

---

## User Story 2: Cost-Controlled Batch Processing

### Test Scenario 2.1 - Cost Estimation Before Processing

**Objective**: Verify system estimates VLM cost within 10% accuracy before processing

**Setup**:
```python
session = create_video_session(
    video_path="/data/videos/dynamic_scene.mp4",  # 100 frames
    start_frame=0,
    end_frame=99
)
```

**Execution**:
```python
# Create job (PENDING state)
job = create_labeling_job(
    session_id=session.id,
    video_path=session.video_path,
    frame_sampling=1,  # Process all frames
    enable_tracking=True  # Enable region tracking optimization
)

# Request cost estimate (samples 15-20% of frames)
estimate = estimate_job_cost(job_id=job.id)
print(f"Estimated cost: ${estimate.estimated_cost:.2f}")
print(f"Confidence: {estimate.confidence_95}")
print(f"Tracking efficiency: {estimate.tracking_efficiency}")
print(f"Scene stability: {estimate.scene_stability}")
```

**Validation**:
```python
# Verify estimate structure
assert estimate.estimated_cost > 0
assert estimate.min_cost < estimate.estimated_cost < estimate.max_cost
assert estimate.sample_size >= 10
assert estimate.scene_stability in ["stable", "moderate", "dynamic"]

# Record estimate for actual cost comparison
estimated_cost = estimate.estimated_cost

# Start job and complete
start_labeling_job(job_id=job.id)
wait_for_job_completion(job.id, timeout_seconds=300)

# Verify actual cost within 10% of estimate
job_status = get_job_status(job.id)
actual_cost = job_status.cost_tracking.total_cost

error_margin = abs(actual_cost - estimated_cost) / estimated_cost
assert error_margin <= 0.10, f"Cost estimate error {error_margin:.1%} exceeds 10% threshold"
```

**Expected Result**: ✅ Cost estimate within 10% of actual cost (SC-002 validated)

---

### Test Scenario 2.2 - Budget Limit Auto-Pause

**Objective**: Verify processing pauses when 95% of budget consumed

**Setup**:
```python
session = create_video_session(
    video_path="/data/videos/high_region_count.mp4",  # 50 frames, 30 regions/frame
    start_frame=0,
    end_frame=49
)
```

**Execution**:
```python
# Set low budget to trigger pause
job = create_labeling_job(
    session_id=session.id,
    video_path=session.video_path,
    budget_limit=5.00,  # $5 budget limit
    frame_sampling=1
)

# Start job
start_labeling_job(job_id=job.id)

# Poll until paused or completed
while True:
    job_status = get_job_status(job.id)
    if job_status.status in ["paused_budget_limit", "completed"]:
        break
    time.sleep(1)
```

**Validation**:
```python
job_status = get_job_status(job.id)

# Verify paused due to budget
assert job_status.status == "paused_budget_limit"

# Verify paused at 95%+ budget consumption
budget_consumed_pct = job_status.cost_tracking.budget_consumed_percentage
assert budget_consumed_pct >= 95.0
assert budget_consumed_pct <= 100.0  # Should not exceed

# Verify total cost does not exceed budget
assert job_status.cost_tracking.total_cost <= job_status.cost_tracking.budget_limit

# Verify partial progress saved
assert job_status.progress.regions_completed > 0
assert job_status.progress.regions_pending > 0
```

**Expected Result**: ✅ Processing paused at 95% budget, no cost overrun (SC-003 validated)

---

### Test Scenario 2.3 - Frame Sampling Cost Reduction

**Objective**: Verify frame sampling reduces costs proportionally

**Setup**:
```python
session = create_video_session(
    video_path="/data/videos/stable_scene.mp4",  # 100 frames
    start_frame=0,
    end_frame=99
)
```

**Execution**:
```python
# Job 1: Process all frames (baseline)
job_full = create_labeling_job(
    session_id=session.id,
    video_path=session.video_path,
    frame_sampling=1,  # Every frame
    enable_tracking=False  # Disable for cost comparison
)
start_labeling_job(job_id=job_full.id)
wait_for_job_completion(job_full.id, timeout_seconds=300)

full_cost = get_job_status(job_full.id).cost_tracking.total_cost

# Job 2: Process every 5th frame (sampled)
job_sampled = create_labeling_job(
    session_id=session.id,
    video_path=session.video_path,
    frame_sampling=5,  # Every 5th frame
    enable_tracking=False
)
start_labeling_job(job_id=job_sampled.id)
wait_for_job_completion(job_sampled.id, timeout_seconds=120)

sampled_cost = get_job_status(job_sampled.id).cost_tracking.total_cost
```

**Validation**:
```python
# Verify cost reduction proportional to sampling ratio
expected_ratio = 1.0 / 5.0  # 20% of frames processed
actual_ratio = sampled_cost / full_cost

# Allow 10% tolerance for region count variance
assert 0.18 <= actual_ratio <= 0.22, f"Sampling ratio {actual_ratio:.2%} not ~20%"

# Verify frames processed matches sampling
frames_processed = get_job_status(job_sampled.id).progress.frames_processed
assert frames_processed == 20  # 100 frames / 5 = 20 frames
```

**Expected Result**: ✅ Frame sampling (every 5th) reduces cost to ~20% of full processing (SC-005 validated)

---

### Test Scenario 2.4 - Pause and Resume Preserves State

**Objective**: Verify pause/resume without data loss

**Setup**:
```python
session = create_video_session(
    video_path="/data/videos/stable_scene.mp4",
    start_frame=0,
    end_frame=49
)
```

**Execution**:
```python
# Create and start job
job = create_labeling_job(session_id=session.id, video_path=session.video_path)
start_labeling_job(job_id=job.id)

# Wait for partial progress (10 frames)
while get_job_status(job.id).progress.frames_processed < 10:
    time.sleep(1)

# Pause job
pause_labeling_job(job_id=job.id)

# Wait for pause to complete (after current query)
while get_job_status(job.id).status == "running":
    time.sleep(0.5)

# Record state at pause
paused_status = get_job_status(job.id)
paused_frames = paused_status.progress.frames_processed
paused_regions = paused_status.progress.regions_completed
paused_cost = paused_status.cost_tracking.total_cost

# Resume job
resume_labeling_job(job_id=job.id)
wait_for_job_completion(job.id, timeout_seconds=300)

# Final state
final_status = get_job_status(job.id)
```

**Validation**:
```python
# Verify paused state preserved
assert paused_status.status == "paused"
assert paused_status.progress.regions_completed > 0

# Verify resume continued from checkpoint
assert final_status.status == "completed"
assert final_status.progress.frames_processed == 50
assert final_status.progress.regions_completed >= paused_regions

# Verify cost accumulated correctly (no double-charging)
assert final_status.cost_tracking.total_cost > paused_cost

# Verify no data loss (all labeled regions accessible)
all_labeled_regions = get_all_labeled_regions(session.id)
assert len(all_labeled_regions) == final_status.progress.regions_completed
```

**Expected Result**: ✅ Pause/resume preserves 100% of progress and labels (SC-007 validated)

---

## User Story 3: Semantic Uncertainty Pattern Detection

### Test Scenario 3.1 - Cluster VLM_UNCERTAIN Regions by Similarity

**Objective**: Verify system groups similar uncertain regions into patterns

**Setup**:
```python
# Use video with multiple similar objects VLM struggles with
session = create_video_session(
    video_path="/data/videos/construction_site.mp4",  # 30 frames, 5-8 construction equipment regions per frame
    start_frame=0,
    end_frame=29
)

# Label all regions
job = create_labeling_job(
    session_id=session.id,
    video_path=session.video_path,
    confidence_threshold=0.5
)
start_labeling_job(job_id=job.id)
wait_for_job_completion(job.id, timeout_seconds=300)
```

**Execution**:
```python
# Detect semantic uncertainty patterns
patterns = detect_uncertainty_patterns(
    job_id=job.id,
    min_regions=2,  # Minimum 2 regions per pattern
    eps=0.3  # DBSCAN similarity threshold
)
```

**Validation**:
```python
# Verify patterns detected
assert len(patterns) > 0, "No patterns detected"

# Verify pattern structure
for pattern in patterns:
    assert pattern.region_count >= 2
    assert len(pattern.region_ids) == pattern.region_count
    assert len(pattern.sample_image_paths) <= 5
    assert 0 <= pattern.avg_similarity_score <= 1.0
    assert pattern.status == "unresolved"

# Verify clustering grouped similar objects
# Manually inspect sample images for one pattern
largest_pattern = max(patterns, key=lambda p: p.region_count)
print(f"Largest pattern: {largest_pattern.region_count} regions")
print(f"Sample images: {largest_pattern.sample_image_paths[:3]}")

# Optional: Visual inspection - display sample images
# for img_path in largest_pattern.sample_image_paths:
#     display_image(img_path)
```

**Expected Result**: ✅ Patterns identified, regions grouped by visual similarity

---

### Test Scenario 3.2 - Pattern Analysis Dashboard

**Objective**: Verify uncertainty patterns displayed in dashboard ranked by frequency

**Setup**:
```python
# Use previous job with detected patterns
patterns = detect_uncertainty_patterns(job_id=job.id)
```

**Execution**:
```python
# Retrieve patterns sorted by frequency
sorted_patterns = sorted(patterns, key=lambda p: p.region_count, reverse=True)

# Display in dashboard (GUI component test)
dashboard_data = {
    "total_patterns": len(sorted_patterns),
    "total_uncertain_regions": sum(p.region_count for p in sorted_patterns),
    "top_patterns": [
        {
            "pattern_id": p.id,
            "region_count": p.region_count,
            "frames_affected": p.frames_affected,
            "sample_images": p.sample_image_paths[:3],
            "avg_similarity": f"{p.avg_similarity_score:.2f}"
        }
        for p in sorted_patterns[:5]  # Top 5 patterns
    ]
}
```

**Validation**:
```python
# Verify top patterns have highest region counts
top_3_counts = [p["region_count"] for p in dashboard_data["top_patterns"][:3]]
assert top_3_counts == sorted(top_3_counts, reverse=True)

# Verify sample images provided for visualization
for pattern in dashboard_data["top_patterns"]:
    assert len(pattern["sample_images"]) > 0
    assert len(pattern["sample_images"]) <= 3
```

**Expected Result**: ✅ Dashboard displays top patterns ranked by frequency with sample images

---

### Test Scenario 3.3 - Batch Manual Labeling for Pattern

**Objective**: Verify user can apply single label to all regions in a pattern

**Setup**:
```python
# Select largest uncertainty pattern
patterns = detect_uncertainty_patterns(job_id=job.id)
target_pattern = max(patterns, key=lambda p: p.region_count)
print(f"Labeling pattern with {target_pattern.region_count} regions")
```

**Execution**:
```python
# Apply manual label to entire pattern
label_result = label_pattern(
    pattern_id=target_pattern.id,
    label="excavator"  # Manual semantic label
)
```

**Validation**:
```python
# Verify pattern marked as resolved
updated_pattern = get_pattern(target_pattern.id)
assert updated_pattern.status == "resolved"
assert updated_pattern.confirmed_label == "excavator"

# Verify all regions in pattern updated
for region_id in updated_pattern.region_ids:
    region = get_region(session.id, region_id)
    assert region.semantic_label == "excavator"
    assert region.vlm_query.status == "CONFIRMED"  # No longer VLM_UNCERTAIN

# Verify regions_updated count matches
assert label_result["regions_updated"] == updated_pattern.region_count
```

**Expected Result**: ✅ Single label applied to all regions in pattern, pattern marked resolved

---

## Integration Tests

### End-to-End Workflow Test

**Objective**: Validate complete workflow from job creation to pattern resolution

**Setup**:
```python
session = create_video_session(
    video_path="/data/videos/construction_site.mp4",
    start_frame=0,
    end_frame=49
)
```

**Execution**:
```python
# Step 1: Create job with budget and tracking
job = create_labeling_job(
    session_id=session.id,
    video_path=session.video_path,
    budget_limit=15.00,
    frame_sampling=1,
    enable_tracking=True,  # Enable region tracking optimization
    tracking_iou_threshold=0.7
)

# Step 2: Estimate cost
estimate = estimate_job_cost(job_id=job.id)
assert estimate.estimated_cost <= 15.00, "Estimated cost exceeds budget"

# Step 3: Start job
start_labeling_job(job_id=job.id)
wait_for_job_completion(job.id, timeout_seconds=600)

# Step 4: Verify completion
job_status = get_job_status(job.id)
assert job_status.status == "completed"
assert job_status.cost_tracking.total_cost <= 15.00

# Step 5: Detect uncertainty patterns
patterns = detect_uncertainty_patterns(job_id=job.id, min_regions=3)
assert len(patterns) > 0

# Step 6: Batch label top pattern
top_pattern = max(patterns, key=lambda p: p.region_count)
label_pattern(pattern_id=top_pattern.id, label="construction_equipment")

# Step 7: Verify pattern resolved
updated_pattern = get_pattern(top_pattern.id)
assert updated_pattern.status == "resolved"
```

**Validation**:
```python
# End-to-end validation
final_status = get_job_status(job.id)

# All regions processed
assert final_status.progress.frames_processed == 50
assert final_status.progress.regions_pending == 0

# Cost tracking accurate
assert final_status.cost_tracking.total_cost > 0
assert final_status.cost_tracking.queries_successful > 0

# Region tracking provided cost savings
tracking_rate = (
    1.0 - (final_status.cost_tracking.queries_successful / final_status.progress.regions_total)
)
assert tracking_rate >= 0.50, "Expected 50%+ regions tracked (not queried)"

# Patterns detected and resolved
resolved_patterns = [p for p in patterns if p.status == "resolved"]
assert len(resolved_patterns) >= 1
```

**Expected Result**: ✅ Complete workflow executes successfully with cost savings from region tracking

---

## Performance Benchmarks

### Region Tracking Cost Savings

**Expected**: 60-70% cost reduction for stable scene videos

**Validation**:
```python
# Compare costs with/without tracking
session = create_video_session(video_path="/data/videos/stable_dashcam.mp4")

# Job without tracking
job_no_tracking = create_labeling_job(session_id=session.id, enable_tracking=False)
start_labeling_job(job_no_tracking.id)
wait_for_job_completion(job_no_tracking.id)
cost_no_tracking = get_job_status(job_no_tracking.id).cost_tracking.total_cost

# Job with tracking
job_with_tracking = create_labeling_job(session_id=session.id, enable_tracking=True)
start_labeling_job(job_with_tracking.id)
wait_for_job_completion(job_with_tracking.id)
cost_with_tracking = get_job_status(job_with_tracking.id).cost_tracking.total_cost

# Verify savings
cost_reduction = (cost_no_tracking - cost_with_tracking) / cost_no_tracking
assert cost_reduction >= 0.60, f"Cost reduction {cost_reduction:.1%} below 60% target"
print(f"Region tracking achieved {cost_reduction:.1%} cost reduction")
```

---

### VLM API Rate Limit Handling

**Expected**: Exponential backoff successfully retries rate-limited requests

**Validation**:
```python
# Create job with high parallelism to trigger rate limits
session = create_video_session(video_path="/data/videos/high_region_count.mp4")

job = create_labeling_job(
    session_id=session.id,
    video_path=session.video_path,
    frame_sampling=1
)

# Monitor rate limit retries
start_labeling_job(job_id=job.id)

retry_count = 0
while True:
    job_status = get_job_status(job.id)
    if job_status.status == "completed":
        break
    # Check for rate limit warnings in logs
    time.sleep(2)

# Verify completion despite rate limits
assert job_status.status == "completed"
assert job_status.cost_tracking.queries_failed == 0  # All retries successful
print("Rate limit handling successful (SC-006 validated)")
```

---

## Troubleshooting

### Common Issues

**Issue**: Cost estimate significantly off (>10% error)
- **Cause**: High scene variance (dynamic video with unpredictable region counts)
- **Solution**: Check `scene_stability` in estimate response - "dynamic" scenes have 12-20% error margin
- **Validation**: Run test on "stable" scene video to verify estimation accuracy baseline

**Issue**: Job paused but budget not at 95%
- **Cause**: Budget limit set too low relative to per-query cost
- **Solution**: Increase budget_limit to at least $1.00 for meaningful progress
- **Validation**: Verify `budget_consumed_percentage` in job status response

**Issue**: No uncertainty patterns detected
- **Cause**: Insufficient VLM_UNCERTAIN regions (need 10+ for meaningful clustering)
- **Solution**: Use test video with objects VLM struggles with (construction equipment, specialized tools)
- **Validation**: Check `queries_uncertain` in cost_tracking - must be >= 10 for pattern detection

**Issue**: Region tracking not reducing costs
- **Cause**: High scene dynamics - regions not tracked across frames (IoU < 0.7)
- **Solution**: Use stable scene video (dashcam, stationary camera) for tracking validation
- **Validation**: Compare `queries_successful` vs `regions_total` - difference = tracked regions

---

## Test Execution Checklist

### Pre-Implementation (Test-First)

- [ ] Write failing test for automatic region queuing (Scenario 1.1)
- [ ] Write failing test for cost estimation (Scenario 2.1)
- [ ] Write failing test for budget pause (Scenario 2.2)
- [ ] Write failing test for pause/resume (Scenario 2.4)
- [ ] Write failing test for pattern detection (Scenario 3.1)
- [ ] Write failing test for batch labeling (Scenario 3.3)
- [ ] Write failing integration test (End-to-End Workflow)

### Post-Implementation (Validation)

- [ ] All unit tests pass (automatic queuing, cost tracking, clustering)
- [ ] All integration tests pass (end-to-end workflow, pause/resume)
- [ ] Success Criteria SC-001 validated (100% regions labeled)
- [ ] Success Criteria SC-002 validated (cost estimate ±10%)
- [ ] Success Criteria SC-003 validated (budget enforcement 100%)
- [ ] Success Criteria SC-004 validated (pattern detection 80%+ accuracy)
- [ ] Success Criteria SC-005 validated (frame sampling cost reduction)
- [ ] Success Criteria SC-006 validated (rate limit handling)
- [ ] Success Criteria SC-007 validated (pause/resume 100% data preservation)
- [ ] Performance benchmark: Region tracking achieves 60-70% cost reduction

---

## Next Steps

After validating all test scenarios:

1. **Generate tasks.md**: Run `/speckit.tasks` to create dependency-ordered implementation tasks
2. **Convert to GitHub Issues**: Run `/speckit.taskstoissues` to create trackable issues
3. **Begin TDD Implementation**: Start with first failing test from Test Execution Checklist
4. **Monitor Constitution Compliance**: Verify Red-Green-Refactor cycle maintained throughout implementation

---

**Note**: All test scenarios follow TDD principles - write failing tests BEFORE implementing features. Each scenario includes setup, execution, validation, and expected results for clear acceptance criteria.
