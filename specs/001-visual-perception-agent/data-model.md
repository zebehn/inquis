# Data Model: Self-Improving Visual Perception Agent

**Date**: 2025-12-17
**Purpose**: Define entities, relationships, and validation rules

## Entity Definitions

### 1. VideoSession

**Purpose**: Represents a video analysis session from upload through processing

**Fields**:
- `id` (UUID): Unique identifier
- `filename` (str): Original video filename
- `filepath` (Path): Path to stored video file
- `upload_timestamp` (datetime): When video was uploaded
- `status` (enum): `uploading` | `processing` | `completed` | `failed`
- `metadata` (dict):
  - `resolution` (tuple): (width, height) in pixels
  - `frame_count` (int): Total number of frames
  - `duration` (float): Video duration in seconds
  - `fps` (float): Frames per second
  - `codec` (str): Video codec
- `processing_progress` (float): 0.0 to 1.0
- `error_message` (str | None): Error details if status is `failed`
- `created_at` (datetime): Session creation time
- `updated_at` (datetime): Last modification time

**Relationships**:
- Has many `SegmentationFrame` (one per video frame)
- Has many `VLMQuery` (for uncertain regions across frames)
- Has many `SyntheticImage` (generated during session)

**Validation Rules**:
- `id` must be valid UUID4
- `filename` must have extension in ['.mp4', '.avi', '.mov']
- `frame_count` must be > 0
- `duration` must be ≤ 600 seconds (10 minutes)
- `fps` must be between 15 and 60
- `processing_progress` must be between 0.0 and 1.0

**State Transitions**:
```
uploading → processing → completed
                ↓
             failed
```

**Storage Format**:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "test_video.mp4",
  "filepath": "/data/sessions/550e8400.../video.mp4",
  "upload_timestamp": "2025-12-17T10:30:00Z",
  "status": "processing",
  "metadata": {
    "resolution": [1920, 1080],
    "frame_count": 300,
    "duration": 10.0,
    "fps": 30.0,
    "codec": "h264"
  },
  "processing_progress": 0.45,
  "error_message": null,
  "created_at": "2025-12-17T10:30:00Z",
  "updated_at": "2025-12-17T10:31:15Z"
}
```

---

### 2. SegmentationFrame

**Purpose**: Stores segmentation results for a single video frame

**Fields**:
- `id` (UUID): Unique identifier
- `session_id` (UUID): Foreign key to VideoSession
- `frame_index` (int): Frame number (0-indexed)
- `timestamp` (float): Frame timestamp in video (seconds)
- `image_path` (Path): Path to extracted frame image
- `masks` (list[dict]): List of detected instance masks
  - `mask_path` (Path): Path to binary mask file (.npz)
  - `class_label` (str): Predicted class name
  - `confidence` (float): Prediction confidence [0, 1]
  - `bbox` (tuple): Bounding box (x, y, width, height)
  - `area` (int): Mask area in pixels
- `uncertainty_map_path` (Path | None): Path to uncertainty heatmap
- `processing_time` (float): Time taken to process frame (seconds)
- `model_version_id` (UUID): Which model version was used
- `processed_at` (datetime): When frame was processed

**Relationships**:
- Belongs to one `VideoSession`
- Has many `UncertainRegion` (regions flagged as uncertain)
- References one `ModelVersion`

**Validation Rules**:
- `frame_index` must be >= 0 and < session.frame_count
- `timestamp` must be >= 0 and < session.duration
- Each mask `confidence` must be between 0.0 and 1.0
- `bbox` values must be non-negative
- `processing_time` must be > 0

**Storage Format**:
```json
{
  "id": "660e9500-f39c-51e5-b827-557766550111",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "frame_index": 42,
  "timestamp": 1.4,
  "image_path": "/data/sessions/550e8400.../frames/frame_0042.jpg",
  "masks": [
    {
      "mask_path": "/data/sessions/550e8400.../segmentations/frame_0042_mask_0.npz",
      "class_label": "car",
      "confidence": 0.92,
      "bbox": [100, 200, 150, 80],
      "area": 8500
    }
  ],
  "uncertainty_map_path": "/data/sessions/550e8400.../uncertainties/frame_0042.npz",
  "processing_time": 0.85,
  "model_version_id": "770e0600-g40d-62f6-c938-668877661222",
  "processed_at": "2025-12-17T10:31:00Z"
}
```

---

### 3. UncertainRegion

**Purpose**: Represents a region where the model is uncertain about classification

**Fields**:
- `id` (UUID): Unique identifier
- `frame_id` (UUID): Foreign key to SegmentationFrame
- `bbox` (tuple): Bounding box (x, y, width, height)
- `crop_path` (Path): Path to cropped image region
- `uncertainty_score` (float): Uncertainty measure [0, 1]
- `top_predictions` (list[dict]): Top-N competing predictions
  - `class_label` (str): Predicted class
  - `confidence` (float): Prediction confidence
- `status` (enum): `pending` | `queried` | `labeled` | `rejected`
- `vlm_query_id` (UUID | None): Associated VLM query if queried
- `final_label` (str | None): User-confirmed label
- `created_at` (datetime): When region was detected

**Relationships**:
- Belongs to one `SegmentationFrame`
- Has zero or one `VLMQuery`

**Validation Rules**:
- `bbox` must be within frame dimensions
- `uncertainty_score` must be between 0.0 and 1.0
- `top_predictions` must be sorted by confidence (descending)
- Sum of `top_predictions` confidences should be ≤ 1.0
- If `status` is `labeled`, `final_label` must not be null

**State Transitions**:
```
pending → queried → labeled
            ↓
         rejected
```

**Storage Format**:
```json
{
  "id": "880e1700-h51e-73g7-d049-779988772333",
  "frame_id": "660e9500-f39c-51e5-b827-557766550111",
  "bbox": [450, 320, 80, 60],
  "crop_path": "/data/sessions/550e8400.../crops/region_880e1700.jpg",
  "uncertainty_score": 0.68,
  "top_predictions": [
    {"class_label": "dog", "confidence": 0.42},
    {"class_label": "cat", "confidence": 0.38},
    {"class_label": "fox", "confidence": 0.20}
  ],
  "status": "labeled",
  "vlm_query_id": "990e2800-i62f-84h8-e150-880099883444",
  "final_label": "fox",
  "created_at": "2025-12-17T10:31:05Z"
}
```

---

### 4. VLMQuery

**Purpose**: Records a query sent to the vision-language model for labeling

**Fields**:
- `id` (UUID): Unique identifier
- `region_id` (UUID): Foreign key to UncertainRegion
- `image_path` (Path): Path to image sent to VLM
- `prompt` (str): Prompt text sent to VLM
- `model_name` (str): VLM model used (e.g., "gpt-5.2")
- `response` (dict):
  - `label` (str): Suggested label
  - `confidence` (float | None): VLM confidence if provided
  - `reasoning` (str): Explanation from VLM
  - `raw_response` (str): Full API response
- `token_count` (int): Tokens used in query
- `cost` (float): Estimated cost in USD
- `latency` (float): Query latency in seconds
- `status` (enum): `pending` | `success` | `failed` | `rate_limited`
- `error_message` (str | None): Error details if failed
- `user_action` (enum | None): `accepted` | `rejected` | `modified`
- `user_modified_label` (str | None): If user modified the VLM suggestion
- `queried_at` (datetime): When query was sent
- `responded_at` (datetime | None): When response received

**Relationships**:
- Belongs to one `UncertainRegion`

**Validation Rules**:
- `prompt` must not be empty
- `token_count` must be > 0
- `cost` must be >= 0
- `latency` must be > 0 if `status` is `success`
- If `user_action` is `modified`, `user_modified_label` must not be null

**Storage Format**:
```json
{
  "id": "990e2800-i62f-84h8-e150-880099883444",
  "region_id": "880e1700-h51e-73g7-d049-779988772333",
  "image_path": "/data/sessions/550e8400.../crops/region_880e1700.jpg",
  "prompt": "What object is in this image? Provide: 1) Specific label 2) Confidence 3) Reasoning",
  "model_name": "gpt-5.2",
  "response": {
    "label": "fox",
    "confidence": 0.85,
    "reasoning": "The object has distinctive pointed ears, bushy tail, and orange-brown fur typical of a red fox.",
    "raw_response": "..."
  },
  "token_count": 145,
  "cost": 0.00725,
  "latency": 1.23,
  "status": "success",
  "error_message": null,
  "user_action": "accepted",
  "user_modified_label": null,
  "queried_at": "2025-12-17T10:31:10Z",
  "responded_at": "2025-12-17T10:31:11Z"
}
```

---

### 5. SyntheticImage

**Purpose**: Represents a generated synthetic training image

**Fields**:
- `id` (UUID): Unique identifier
- `source_label` (str): Label used to generate image
- `source_region_id` (UUID | None): Original uncertain region if applicable
- `image_path` (Path): Path to generated image
- `mask_path` (Path): Path to segmentation mask
- `generation_params` (dict):
  - `prompt` (str): Full generation prompt
  - `model` (str): Generator model (e.g., "z-image-turbo")
  - `steps` (int): Inference steps
  - `seed` (int): Random seed for reproducibility
- `quality_score` (float): Quality metric [0, 1] (e.g., CLIP similarity)
- `inclusion_status` (enum): `pending_review` | `accepted` | `rejected`
- `rejection_reason` (str | None): Why rejected if applicable
- `used_in_training` (bool): Whether included in training dataset
- `training_dataset_id` (UUID | None): Associated training dataset
- `generated_at` (datetime): When image was generated

**Relationships**:
- May belong to one `UncertainRegion` (source)
- May belong to one `TrainingDataset`

**Validation Rules**:
- `source_label` must not be empty
- `quality_score` must be between 0.0 and 1.0
- If `inclusion_status` is `rejected`, `rejection_reason` should not be null
- If `used_in_training` is true, `training_dataset_id` must not be null

**Storage Format**:
```json
{
  "id": "aa0e3900-j73g-95i9-f261-991100994555",
  "source_label": "fox",
  "source_region_id": "880e1700-h51e-73g7-d049-779988772333",
  "image_path": "/data/sessions/550e8400.../synthetic/syn_aa0e3900.jpg",
  "mask_path": "/data/sessions/550e8400.../synthetic/syn_aa0e3900_mask.npz",
  "generation_params": {
    "prompt": "A photorealistic fox in outdoor daytime setting with varied lighting",
    "model": "z-image-turbo",
    "steps": 8,
    "seed": 42
  },
  "quality_score": 0.87,
  "inclusion_status": "accepted",
  "rejection_reason": null,
  "used_in_training": true,
  "training_dataset_id": "bb0e4000-k84h-06j0-g372-002211005666",
  "generated_at": "2025-12-17T10:32:00Z"
}
```

---

### 6. ModelVersion

**Purpose**: Tracks trained segmentation model versions and performance

**Fields**:
- `id` (UUID): Unique identifier
- `version_number` (int): Sequential version (1, 2, 3, ...)
- `base_model` (str): Base model name (e.g., "sam2_hiera_large")
- `checkpoint_path` (Path): Path to model weights
- `training_dataset_id` (UUID | None): Dataset used for this version
- `training_params` (dict):
  - `learning_rate` (float): Training learning rate
  - `epochs` (int): Number of training epochs
  - `batch_size` (int): Batch size used
  - `lora_rank` (int): LoRA rank if used
- `metrics` (dict):
  - `iou` (float): Intersection over Union
  - `precision` (float): Precision score
  - `recall` (float): Recall score
  - `uncertainty_reduction` (float): % reduction in uncertain regions
- `parent_version_id` (UUID | None): Previous model version if incremental
- `is_active` (bool): Whether this is the currently used model
- `notes` (str): Human-readable notes about this version
- `created_at` (datetime): When version was created
- `trained_at` (datetime | None): When training completed

**Relationships**:
- May belong to one `TrainingDataset`
- May reference one parent `ModelVersion`
- Has many `SegmentationFrame` (frames processed with this version)

**Validation Rules**:
- `version_number` must be > 0 and unique
- `metrics` values must be between 0.0 and 1.0
- Only one version can have `is_active` = true at a time
- If `parent_version_id` is null, this is the base model (version 1)

**Storage Format**:
```json
{
  "id": "bb0e4000-k84h-06j0-g372-002211005666",
  "version_number": 2,
  "base_model": "sam2_hiera_large",
  "checkpoint_path": "/data/models/versions/v2/lora_weights.pt",
  "training_dataset_id": "cc0e5100-l95i-17k1-h483-113322116777",
  "training_params": {
    "learning_rate": 0.0001,
    "epochs": 5,
    "batch_size": 4,
    "lora_rank": 8
  },
  "metrics": {
    "iou": 0.78,
    "precision": 0.82,
    "recall": 0.75,
    "uncertainty_reduction": 0.32
  },
  "parent_version_id": "770e0600-g40d-62f6-c938-668877661222",
  "is_active": true,
  "notes": "Trained with 150 synthetic fox images, significant improvement on wildlife detection",
  "created_at": "2025-12-17T10:35:00Z",
  "trained_at": "2025-12-17T10:45:00Z"
}
```

---

### 7. TrainingDataset

**Purpose**: Represents a collection of images used for model training

**Fields**:
- `id` (UUID): Unique identifier
- `name` (str): Human-readable dataset name
- `description` (str): Dataset purpose and contents
- `images` (list[dict]): List of training images
  - `image_path` (Path): Path to image
  - `mask_path` (Path): Path to ground truth mask
  - `label` (str): Class label
  - `source` (enum): `original` | `synthetic`
  - `source_id` (UUID | None): ID of synthetic image if applicable
- `statistics` (dict):
  - `total_images` (int): Total number of images
  - `original_count` (int): Count of original images
  - `synthetic_count` (int): Count of synthetic images
  - `class_distribution` (dict): Label → count mapping
- `version` (int): Dataset version number
- `created_at` (datetime): When dataset was created
- `last_modified` (datetime): Last modification time

**Relationships**:
- Has many `SyntheticImage`
- Has many `ModelVersion` (models trained on this dataset)

**Validation Rules**:
- `name` must not be empty
- `total_images` must equal `original_count` + `synthetic_count`
- Sum of `class_distribution` values must equal `total_images`
- Each image in `images` must have valid paths

**Storage Format**:
```json
{
  "id": "cc0e5100-l95i-17k1-h483-113322116777",
  "name": "Training Set v2 - Wildlife Enhancement",
  "description": "Base training set + 150 synthetic fox images",
  "images": [
    {
      "image_path": "/data/datasets/original/img_001.jpg",
      "mask_path": "/data/datasets/original/mask_001.npz",
      "label": "dog",
      "source": "original",
      "source_id": null
    },
    {
      "image_path": "/data/sessions/550e8400.../synthetic/syn_aa0e3900.jpg",
      "mask_path": "/data/sessions/550e8400.../synthetic/syn_aa0e3900_mask.npz",
      "label": "fox",
      "source": "synthetic",
      "source_id": "aa0e3900-j73g-95i9-f261-991100994555"
    }
  ],
  "statistics": {
    "total_images": 500,
    "original_count": 350,
    "synthetic_count": 150,
    "class_distribution": {
      "dog": 100,
      "cat": 80,
      "fox": 170,
      "bird": 150
    }
  },
  "version": 2,
  "created_at": "2025-12-17T10:34:00Z",
  "last_modified": "2025-12-17T10:34:30Z"
}
```

---

## Entity Relationships Diagram

```
VideoSession (1) ──── (N) SegmentationFrame
                              │
                              ├── (N) UncertainRegion
                              │        │
                              │        └── (0..1) VLMQuery
                              │
                              └── (N) ModelVersion

UncertainRegion (1) ──── (0..N) SyntheticImage
                                       │
                                       └── (0..1) TrainingDataset

TrainingDataset (1) ──── (N) ModelVersion
ModelVersion (parent) ──── (N) ModelVersion (children)
```

## Data Access Patterns

### Common Queries

1. **Get all frames for a session**
   ```python
   frames = SegmentationFrame.filter(session_id=session_id).order_by('frame_index')
   ```

2. **Find uncertain regions needing VLM labeling**
   ```python
   regions = UncertainRegion.filter(status='pending').order_by('uncertainty_score', desc=True)
   ```

3. **Get synthetic images ready for training**
   ```python
   images = SyntheticImage.filter(inclusion_status='accepted', used_in_training=False)
   ```

4. **Track model performance over versions**
   ```python
   versions = ModelVersion.all().order_by('version_number')
   metrics_history = [(v.version_number, v.metrics) for v in versions]
   ```

5. **Calculate API costs for a session**
   ```python
   queries = VLMQuery.filter(session_id=session_id, status='success')
   total_cost = sum(q.cost for q in queries)
   ```

## Indexing Strategy

For efficient queries on file-based storage:

- **VideoSession**: Index by `id`, `status`, `created_at`
- **SegmentationFrame**: Index by `session_id`, `frame_index`
- **UncertainRegion**: Index by `frame_id`, `status`, `uncertainty_score`
- **VLMQuery**: Index by `region_id`, `status`, `queried_at`
- **SyntheticImage**: Index by `source_label`, `inclusion_status`, `quality_score`
- **ModelVersion**: Index by `version_number`, `is_active`
- **TrainingDataset**: Index by `id`, `version`

## Validation Functions

```python
from pydantic import BaseModel, validator, UUID4
from typing import List, Optional, Tuple
from datetime import datetime
from enum import Enum

class SessionStatus(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class VideoSession(BaseModel):
    id: UUID4
    filename: str
    filepath: Path
    status: SessionStatus
    metadata: dict
    processing_progress: float

    @validator('filename')
    def validate_filename(cls, v):
        if not v.endswith(('.mp4', '.avi', '.mov')):
            raise ValueError('Invalid video format')
        return v

    @validator('processing_progress')
    def validate_progress(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Progress must be between 0 and 1')
        return v

    @validator('metadata')
    def validate_metadata(cls, v):
        if v['duration'] > 600:
            raise ValueError('Video duration exceeds 10 minute limit')
        if not 15 <= v['fps'] <= 60:
            raise ValueError('FPS must be between 15 and 60')
        return v
```

---

## Next Steps

1. ✅ Data model defined
2. ➡️ Create API contracts (`contracts/`)
3. ➡️ Write quickstart guide (`quickstart.md`)
4. ➡️ Update agent context
