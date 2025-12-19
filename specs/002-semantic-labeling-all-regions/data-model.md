# Data Model: Semantic Labeling for All Regions

**Date**: 2025-12-19
**Feature**: 002-semantic-labeling-all-regions
**Purpose**: Define data entities, relationships, validation rules, and state transitions

---

## Overview

This feature introduces 3 new entity models and extends 2 existing models to support:
1. **Automatic VLM labeling** of all segmented regions (not just uncertain ones)
2. **Cost control** with budget limits and real-time tracking
3. **Region tracking** across frames to reduce redundant VLM queries
4. **Pause/resume** for long-running labeling jobs
5. **Semantic uncertainty pattern detection** through clustering

---

## Entity Relationship Diagram

```text
VideoSession (existing)
    ↓ 1:N
SemanticLabelingJob (NEW)
    ↓ 1:N                    ↓ 1:N
TrackedRegion (extended)    VLMQuery (existing)
    ↓ N:M
SemanticUncertaintyPattern (NEW)
```

**Relationships**:
- One `VideoSession` can have multiple `SemanticLabelingJob` instances (different runs)
- One `SemanticLabelingJob` processes many `TrackedRegion` instances
- One `SemanticLabelingJob` generates many `VLMQuery` instances
- One `TrackedRegion` can belong to multiple `SemanticUncertaintyPattern` clusters
- One `SemanticUncertaintyPattern` aggregates multiple `TrackedRegion` instances

---

## 1. SemanticLabelingJob (NEW)

**Purpose**: Represents a long-running automatic VLM labeling session with pause/resume, cost tracking, and budget enforcement.

### Model Definition

```python
from enum import Enum
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from pathlib import Path
from datetime import datetime
from typing import Optional, List

class JobStatus(str, Enum):
    """Job lifecycle states"""
    PENDING = "pending"              # Job created, not yet started
    RUNNING = "running"              # Currently processing regions
    PAUSED = "paused"                # User-initiated pause
    PAUSED_BUDGET_LIMIT = "paused_budget_limit"  # Auto-paused due to budget
    COMPLETED = "completed"          # All regions processed
    FAILED = "failed"                # Unrecoverable error occurred
    CANCELLED = "cancelled"          # User cancelled job

class SemanticLabelingJob(BaseModel):
    """
    Represents an automatic VLM labeling job with pause/resume support.

    Lifecycle:
        PENDING → RUNNING → PAUSED → RUNNING → COMPLETED
                         ↓                    ↓
                         PAUSED_BUDGET_LIMIT  FAILED/CANCELLED
    """

    # ========== Identity ==========
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID  # Foreign key to VideoSession
    video_path: Path

    # ========== Status ==========
    status: JobStatus = Field(default=JobStatus.PENDING)
    error_message: Optional[str] = None  # Populated if status=FAILED

    # ========== Progress Tracking (Critical for resume) ==========
    frames_total: int = Field(ge=0)
    frames_processed: List[int] = Field(default_factory=list)
    frames_pending: List[int] = Field(default_factory=list)
    current_frame: Optional[int] = None

    regions_total: int = Field(ge=0)
    regions_completed: List[UUID] = Field(default_factory=list)
    regions_pending: List[UUID] = Field(default_factory=list)
    regions_failed: List[UUID] = Field(default_factory=list)
    in_flight_region: Optional[UUID] = None  # Currently querying VLM

    # ========== Cost Tracking ==========
    total_cost: float = Field(default=0.0, ge=0.0)
    total_tokens: int = Field(default=0, ge=0)
    budget_limit: Optional[float] = Field(default=None, ge=0.0)
    queries_successful: int = Field(default=0, ge=0)
    queries_failed: int = Field(default=0, ge=0)
    queries_uncertain: int = Field(default=0, ge=0)

    # ========== Configuration (needed to resume with same settings) ==========
    frame_sampling: int = Field(default=1, ge=1)  # Process every Nth frame
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    model_name: str = "gpt-5.2"
    enable_tracking: bool = True  # Use region tracking optimization
    tracking_iou_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # ========== Timestamps ==========
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    resumed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # ========== Checkpoint Metadata ==========
    last_checkpoint_at: datetime = Field(default_factory=datetime.now)
    checkpoint_version: int = Field(default=0, ge=0)  # Increment on each save

    class Config:
        json_encoders = {
            UUID: str,
            Path: str,
            datetime: lambda v: v.isoformat(),
        }

    # ========== Computed Properties ==========

    def get_progress_percentage(self) -> float:
        """Calculate overall progress percentage"""
        if self.regions_total == 0:
            return 100.0
        completed = len(self.regions_completed)
        return (completed / self.regions_total) * 100.0

    def get_budget_consumed_percentage(self) -> float:
        """Calculate percentage of budget consumed"""
        if not self.budget_limit or self.budget_limit == 0:
            return 0.0
        return (self.total_cost / self.budget_limit) * 100.0

    def is_budget_exhausted(self, threshold: float = 0.95) -> bool:
        """Check if budget is exhausted (default 95% threshold)"""
        if not self.budget_limit:
            return False
        return self.get_budget_consumed_percentage() >= threshold * 100.0

    def get_average_cost_per_region(self) -> float:
        """Calculate average cost per successfully labeled region"""
        if self.queries_successful == 0:
            return 0.0
        return self.total_cost / self.queries_successful

    def get_estimated_remaining_cost(self) -> Optional[float]:
        """Estimate cost to complete remaining regions"""
        avg_cost = self.get_average_cost_per_region()
        if avg_cost == 0:
            return None
        remaining_regions = len(self.regions_pending)
        return avg_cost * remaining_regions

    def get_success_rate(self) -> float:
        """Calculate VLM query success rate"""
        total_queries = self.queries_successful + self.queries_failed + self.queries_uncertain
        if total_queries == 0:
            return 0.0
        return (self.queries_successful / total_queries) * 100.0

    # ========== Validation Rules ==========

    @validator('budget_limit')
    def validate_budget_limit(cls, v, values):
        """Ensure budget limit is positive if provided"""
        if v is not None and v <= 0:
            raise ValueError("Budget limit must be positive")
        return v

    @validator('frame_sampling')
    def validate_frame_sampling(cls, v):
        """Ensure frame sampling is at least 1"""
        if v < 1:
            raise ValueError("Frame sampling must be >= 1")
        return v

    @validator('confidence_threshold')
    def validate_confidence_threshold(cls, v):
        """Ensure confidence threshold is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        return v
```

### State Transitions

```text
                    ┌─────────┐
                    │ PENDING │
                    └────┬────┘
                         │ start_job()
                         ↓
    ┌──────────────► RUNNING ◄────────────────┐
    │                   │ ↓                    │
    │ resume_job()      │ │ request_pause()    │ resume_job()
    │                   │ ↓                    │
    ├─────────────── PAUSED                    │
    │                   │ ↓                    │
    │                   │ │ budget_exceeded()  │
    │                   │ ↓                    │
    └────────── PAUSED_BUDGET_LIMIT ───────────┘
                         │
                         │ all_regions_completed()
                         ↓
                    ┌───────────┐
                    │ COMPLETED │
                    └───────────┘

                         │ error_occurred()
                         ↓
                    ┌─────────┐
                    │ FAILED  │
                    └─────────┘
```

**Valid Transitions**:
- `PENDING → RUNNING`: User starts job
- `RUNNING → PAUSED`: User requests pause
- `RUNNING → PAUSED_BUDGET_LIMIT`: 95% budget consumed
- `RUNNING → COMPLETED`: All regions processed successfully
- `RUNNING → FAILED`: Unrecoverable error (e.g., storage failure)
- `PAUSED → RUNNING`: User resumes job
- `PAUSED_BUDGET_LIMIT → RUNNING`: User increases budget and resumes
- `ANY → CANCELLED`: User cancels job

---

## 2. TrackedRegion (EXTENSION to existing SegmentedRegion)

**Purpose**: Extends `SegmentedRegion` or `UncertainRegion` with tracking metadata to enable region tracking optimization across frames.

### Extension Fields

```python
class TrackedRegion(SegmentedRegion):  # Or extend UncertainRegion
    """
    Extension of SegmentedRegion with tracking metadata for cross-frame optimization.
    """

    # ========== Tracking Metadata ==========
    tracking_status: str = "new"  # "new", "tracked", "occluded", "split", "merged", "ended"
    parent_region_id: Optional[UUID] = None  # Region ID from previous frame (if tracked)
    tracked_iou: Optional[float] = None  # IoU with parent region
    frames_since_last_query: int = 0  # Staleness metric for periodic re-querying

    # ========== Tracking History ==========
    first_seen_frame: int  # Frame where region first appeared
    last_seen_frame: int  # Frame where region was last detected
    tracking_confidence: float = Field(default=1.0, ge=0.0, le=1.0)  # Confidence in tracking

    # ========== Semantic Label (inherited or copied from parent) ==========
    semantic_label_source: str = "vlm"  # "vlm", "tracked", "manual"
    # If tracked: semantic_label copied from parent_region
    # If vlm: semantic_label from VLM query
    # If manual: semantic_label from user input
```

### Tracking Status Values

| Status | Description | Next Action |
|--------|-------------|-------------|
| `new` | First appearance in video | Query VLM for semantic label |
| `tracked` | Matched to region in previous frame (IoU ≥ 0.7) | Skip VLM query, copy semantic label from parent |
| `occluded` | Missing for 1-3 frames, expected to reappear | Skip VLM query, wait for reappearance |
| `split` | One parent region became multiple child regions | Query VLM for each split (labels may differ) |
| `merged` | Multiple parent regions merged into one | Query VLM for merged region (combined meaning) |
| `ended` | Missing for 4+ frames, considered disappeared | N/A (no longer tracked) |

### Validation Rules

```python
@validator('tracked_iou')
def validate_tracked_iou(cls, v, values):
    """Tracked IoU must be between 0 and 1, and should be >= threshold if tracked"""
    if v is not None:
        if not 0 <= v <= 1:
            raise ValueError("Tracked IoU must be between 0 and 1")
        if values.get('tracking_status') == 'tracked' and v < 0.7:
            logger.warning(f"Tracked region has low IoU: {v}")
    return v

@validator('semantic_label_source')
def validate_label_source(cls, v):
    """Ensure label source is valid"""
    valid_sources = {'vlm', 'tracked', 'manual'}
    if v not in valid_sources:
        raise ValueError(f"Label source must be one of {valid_sources}")
    return v
```

---

## 3. SemanticUncertaintyPattern (NEW)

**Purpose**: Represents a cluster of visually similar VLM_UNCERTAIN regions, enabling batch manual labeling and targeted model improvement.

### Model Definition

```python
class SemanticUncertaintyPattern(BaseModel):
    """
    Represents a cluster of visually similar VLM_UNCERTAIN regions.
    Enables batch labeling and identification of systematic VLM struggles.
    """

    # ========== Identity ==========
    id: UUID = Field(default_factory=uuid4)
    job_id: UUID  # Foreign key to SemanticLabelingJob
    cluster_id: int = Field(ge=0)  # DBSCAN cluster label

    # ========== Cluster Membership ==========
    region_ids: List[UUID] = Field(min_items=2)  # At least 2 regions to form a pattern
    region_count: int = Field(ge=2)
    frames_affected: List[int] = Field(default_factory=list)

    # ========== Visual Characteristics ==========
    sample_image_paths: List[Path] = Field(max_items=5)  # Representative samples
    avg_bbox_size: Tuple[float, float]  # (width, height)
    avg_bbox_aspect_ratio: float = Field(gt=0.0)
    dominant_colors: List[str] = Field(default_factory=list)  # Hex color codes

    # ========== Clustering Metrics ==========
    avg_similarity_score: float = Field(ge=0.0, le=1.0)  # Average intra-cluster similarity
    cluster_compactness: float = Field(ge=0.0, le=1.0)  # How tightly grouped the cluster is

    # ========== Resolution Status ==========
    status: str = "unresolved"  # "unresolved", "in_progress", "resolved"
    confirmed_label: Optional[str] = None  # Manual label applied to all regions
    resolved_by: Optional[str] = None  # User ID or "system"

    # ========== Timestamps ==========
    created_at: datetime = Field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None

    class Config:
        json_encoders = {
            UUID: str,
            Path: str,
            datetime: lambda v: v.isoformat(),
        }

    # ========== Computed Properties ==========

    def get_frames_affected_count(self) -> int:
        """Count number of unique frames containing these regions"""
        return len(self.frames_affected)

    def get_density(self) -> float:
        """Calculate pattern density (regions per frame)"""
        if len(self.frames_affected) == 0:
            return 0.0
        return self.region_count / len(self.frames_affected)

    def is_high_priority(self) -> bool:
        """Determine if pattern requires urgent attention (many regions, many frames)"""
        return self.region_count >= 10 or len(self.frames_affected) >= 5

    # ========== Validation Rules ==========

    @validator('region_count')
    def validate_region_count_matches_ids(cls, v, values):
        """Ensure region_count matches length of region_ids"""
        region_ids = values.get('region_ids', [])
        if v != len(region_ids):
            raise ValueError(f"region_count ({v}) must match length of region_ids ({len(region_ids)})")
        return v

    @validator('status')
    def validate_status(cls, v):
        """Ensure status is valid"""
        valid_statuses = {'unresolved', 'in_progress', 'resolved'}
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")
        return v

    @validator('confirmed_label')
    def validate_label_when_resolved(cls, v, values):
        """Ensure confirmed_label is provided when status=resolved"""
        if values.get('status') == 'resolved' and not v:
            raise ValueError("confirmed_label required when status=resolved")
        return v
```

### State Transitions

```text
┌────────────┐
│ unresolved │ ─── user_starts_labeling() ───► ┌─────────────┐
└────────────┘                                   │ in_progress │
                                                 └──────┬──────┘
                                                        │
                                          user_applies_label()
                                                        ↓
                                                  ┌──────────┐
                                                  │ resolved │
                                                  └──────────┘
```

---

## 4. VLMBatchQuery (NEW - Optional Enhancement)

**Purpose**: Represents a batch of regions from a single frame sent to VLM together for efficiency. Currently not used (sequential queries preferred), but defined for future optimization.

### Model Definition

```python
class VLMBatchQuery(BaseModel):
    """
    Represents a batch of regions from a frame queried together.
    Currently unused (sequential queries preferred), but defined for future use.
    """

    # ========== Identity ==========
    id: UUID = Field(default_factory=uuid4)
    job_id: UUID  # Foreign key to SemanticLabelingJob
    frame_index: int = Field(ge=0)

    # ========== Batch Contents ==========
    region_ids: List[UUID] = Field(min_items=1)
    region_count: int = Field(ge=1)

    # ========== Query Details ==========
    model_name: str = "gpt-5.2"
    prompt: str
    queried_at: datetime = Field(default_factory=datetime.now)
    responded_at: Optional[datetime] = None

    # ========== Response ==========
    status: str = "pending"  # "pending", "success", "partial_success", "failed"
    successful_regions: List[UUID] = Field(default_factory=list)
    failed_regions: List[UUID] = Field(default_factory=list)

    # ========== Cost Tracking ==========
    total_tokens: int = Field(default=0, ge=0)
    total_cost: float = Field(default=0.0, ge=0.0)
    avg_cost_per_region: float = Field(default=0.0, ge=0.0)

    # ========== Performance ==========
    latency_seconds: float = Field(default=0.0, ge=0.0)

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }
```

---

## 5. Extensions to Existing Models

### VLMQuery (Existing - No Changes Required)

The existing `VLMQuery` model at `/data1/Developments/inquis/src/models/vlm_query.py` already supports all required fields:
- `status: VLMQueryStatus` (SUCCESS, VLM_UNCERTAIN, FAILED, RATE_LIMITED)
- `cost: float` and `token_count: int` for budget tracking
- `queried_at` and `responded_at` timestamps
- `response: Dict[str, Any]` with label, confidence, reasoning

**No modifications needed.**

### VideoSession (Existing - Minor Extension)

Add optional reference to active labeling job:

```python
class VideoSession(BaseModel):
    # ... existing fields ...

    # New field (optional)
    active_labeling_job_id: Optional[UUID] = None  # Currently running job, if any
```

---

## Storage Schema

### File Structure

```text
/data/sessions/{session_id}/
├── metadata/
│   ├── session.json                      # VideoSession
│   ├── vlm_queries/
│   │   └── {query_id}.json               # VLMQuery instances
│   └── regions/
│       └── {region_id}.json              # TrackedRegion instances
├── checkpoints/
│   ├── job_{job_id}.json                 # SemanticLabelingJob (current)
│   ├── job_{job_id}_backup_1.json        # Backup 1 (most recent)
│   ├── job_{job_id}_backup_2.json        # Backup 2
│   └── job_{job_id}_backup_3.json        # Backup 3 (oldest, rotated out)
└── patterns/
    └── pattern_{pattern_id}.json         # SemanticUncertaintyPattern instances
```

### JSON Serialization Examples

**SemanticLabelingJob**:
```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "session_id": "session-uuid",
  "video_path": "/data/videos/dashcam.mp4",
  "status": "paused",
  "frames_total": 100,
  "frames_processed": [0, 1, 2, 3, 4],
  "frames_pending": [5, 6, 7, ...],
  "regions_completed": ["region-uuid-1", "region-uuid-2", ...],
  "regions_pending": ["region-uuid-50", "region-uuid-51", ...],
  "total_cost": 2.45,
  "budget_limit": 10.00,
  "checkpoint_version": 127
}
```

**SemanticUncertaintyPattern**:
```json
{
  "id": "pattern-uuid",
  "job_id": "job-uuid",
  "cluster_id": 0,
  "region_ids": ["region-uuid-1", "region-uuid-2", "region-uuid-3"],
  "region_count": 3,
  "frames_affected": [5, 12, 18],
  "sample_image_paths": ["/data/.../crop_5_0.jpg", "/data/.../crop_12_3.jpg"],
  "avg_bbox_size": [120.5, 85.3],
  "status": "resolved",
  "confirmed_label": "construction_equipment"
}
```

---

## Indexing and Query Optimization

### Recommended Indexes

**For Fast Lookup**:
- `SemanticLabelingJob`: Index by `session_id` + `status`
- `TrackedRegion`: Index by `parent_region_id` for tracking lookups
- `SemanticUncertaintyPattern`: Index by `job_id` + `status`

**Implementation**: Use in-memory dictionaries for file-based storage:
```python
# In StorageService
jobs_by_session: Dict[UUID, List[UUID]] = {}  # session_id -> [job_ids]
regions_by_parent: Dict[UUID, List[UUID]] = {}  # parent_id -> [child_ids]
```

---

## Data Migration

### From Existing Models

**No breaking changes** - this feature adds new models and extends existing ones with optional fields.

**Migration Steps**:
1. Existing `VideoSession` instances: Add `active_labeling_job_id = None`
2. Existing `SegmentedRegion` instances: Can be converted to `TrackedRegion` on-the-fly by setting `tracking_status = "new"`
3. Existing `VLMQuery` instances: No changes required

---

## Validation Summary

### Cross-Entity Invariants

1. **Job Progress Consistency**:
   ```python
   assert job.regions_total == len(job.regions_completed) + len(job.regions_pending) + len(job.regions_failed)
   ```

2. **Budget Enforcement**:
   ```python
   if job.budget_limit:
       assert job.total_cost <= job.budget_limit * 1.05  # Allow 5% overage
   ```

3. **Pattern Membership**:
   ```python
   for pattern in patterns:
       assert pattern.region_count == len(pattern.region_ids)
       assert all(region.status == VLMQueryStatus.VLM_UNCERTAIN for region in pattern.regions)
   ```

4. **Tracking Consistency**:
   ```python
   for region in tracked_regions:
       if region.tracking_status == "tracked":
           assert region.parent_region_id is not None
           assert region.tracked_iou >= 0.7
   ```

---

## Conclusion

This data model provides:
- **Comprehensive state management** for pause/resume
- **Cost tracking** at multiple granularities (job, query, region)
- **Flexible tracking metadata** for optimization
- **Pattern detection** for semantic uncertainty analysis
- **Backward compatibility** with existing models
- **Clear validation rules** to maintain data integrity

All models use Pydantic for automatic validation and JSON serialization, matching the existing codebase architecture.
