# Feature Specification: Semantic Labeling for All Regions

**Feature Branch**: `002-semantic-labeling-all-regions`
**Created**: 2025-12-19
**Status**: Draft
**Input**: User description: "if a frame does not have any uncertain regions, the labeling is not performed, but I thought that area labeling would be a default action to detect semantic uncertainty."

## User Scenarios & Testing

### User Story 1 - Automatic Semantic Labeling of All Regions (Priority: P1)

When a video frame is processed and segmented, the system automatically sends all detected regions (not just uncertain ones) to the VLM for semantic classification. This enables the detection of semantic uncertainty—situations where the segmentation quality is high but the VLM cannot confidently identify what the object is. All regions receive semantic labels by default, creating a complete semantic understanding of each frame.

**Why this priority**: This is the core feature that enables semantic uncertainty detection. Without it, frames with only high-quality segmentations remain unlabeled, missing opportunities to discover objects the VLM cannot classify confidently.

**Independent Test**: Can be fully tested by processing a video frame with multiple regions (mix of high and low segmentation quality), verifying that all regions—regardless of segmentation quality—are automatically sent to the VLM and receive semantic labels or VLM_UNCERTAIN status.

**Acceptance Scenarios**:

1. **Given** a video frame is segmented with 10 regions where 8 have high segmentation quality (predicted_iou > 0.75) and 2 have low segmentation quality (predicted_iou < 0.75), **When** the frame processing completes, **Then** all 10 regions are automatically queued for VLM labeling without requiring user selection
2. **Given** all regions in a frame have high segmentation quality, **When** the frame is processed, **Then** the system still sends all regions to the VLM for semantic classification to detect any semantic uncertainty
3. **Given** the VLM returns confident labels for 7 regions but returns low confidence for 3 regions, **When** the user reviews the frame, **Then** the 3 low-confidence regions are marked as VLM_UNCERTAIN and flagged for manual labeling, while the 7 confident regions show their semantic labels

---

### User Story 2 - Cost-Controlled Batch Processing (Priority: P2)

Given that automatic labeling of all regions increases VLM API costs, the system provides cost control mechanisms. Users can set budget limits, enable selective frame sampling (every Nth frame), and preview estimated costs before processing. The system tracks cumulative costs in real-time and pauses processing when budget limits are approached.

**Why this priority**: Automatic labeling is powerful but expensive. Cost controls make the feature practical for real-world use with API-metered VLMs.

**Independent Test**: Can be tested by setting a $5 budget limit, processing a video with 100 frames and 10 regions per frame, and verifying the system pauses when approaching the limit and shows accurate cost tracking.

**Acceptance Scenarios**:

1. **Given** a user sets a session budget of $10, **When** video processing is initiated, **Then** the system estimates total VLM cost based on frame count and average regions per frame, displays the estimate, and asks for confirmation before proceeding
2. **Given** automatic labeling is in progress with a $10 budget, **When** cumulative VLM costs reach $9.50, **Then** the system pauses processing and alerts the user that 95% of budget is consumed, offering options to increase budget or stop
3. **Given** a user wants to reduce costs, **When** configuring video processing, **Then** the user can enable frame sampling (e.g., label every 5th frame) to reduce the number of regions sent to the VLM while still detecting semantic uncertainty patterns
4. **Given** processing is paused due to budget limits, **When** the user reviews labeled frames, **Then** already-labeled regions remain accessible and the user can choose which remaining frames to prioritize for labeling within remaining budget

---

### User Story 3 - Semantic Uncertainty Pattern Detection (Priority: P3)

After all regions are labeled, the system analyzes VLM responses across frames to identify semantic uncertainty patterns. It surfaces object types the VLM consistently struggles with (e.g., "VLM uncertain about 15 regions in 8 frames, all showing similar visual characteristics"), enabling targeted manual labeling and synthetic data generation for specific problematic object classes.

**Why this priority**: Detecting patterns in semantic uncertainty helps prioritize where the model needs improvement. It builds on P1's labeling to provide actionable insights.

**Independent Test**: Can be tested by processing multiple frames containing similar objects (e.g., construction equipment) that the VLM struggles with, then verifying the system identifies this as a semantic uncertainty pattern and recommends targeted action.

**Acceptance Scenarios**:

1. **Given** VLM labeling is complete for 50 frames, **When** the system analyzes all VLM_UNCERTAIN regions, **Then** the system groups uncertain regions by visual similarity (using embedding clustering or bounding box + mask similarity) and reports clusters with counts (e.g., "12 uncertain regions appear to be similar construction equipment")
2. **Given** semantic uncertainty patterns are detected, **When** the user views the analysis dashboard, **Then** the system displays top uncertainty clusters ranked by frequency, showing example images from each cluster and suggesting priority for manual labeling
3. **Given** a specific semantic uncertainty pattern is identified (e.g., "traffic equipment" objects), **When** the user selects the pattern, **Then** the system shows all related uncertain regions across all frames for batch manual labeling and offers to generate synthetic training data for that object class
4. **Given** manual labels are provided for an uncertainty cluster, **When** the user confirms labels for all regions in the cluster, **Then** the system updates all related regions with the confirmed semantic label and marks the pattern as resolved

---

### Edge Cases

- What happens when a frame has 100+ regions and automatic labeling would cost $5 per frame?
  - System shows cost warning before processing that specific frame and offers option to skip, sample regions, or proceed

- How does the system handle VLM API rate limits when auto-labeling all regions?
  - Implements exponential backoff and retry logic; shows real-time progress with "Rate limited - retrying in Xs" messages

- What happens if the user cancels processing mid-frame while regions are being labeled?
  - Gracefully stops after current region completes labeling; saves all completed labels; allows resuming from stopping point

- How are regions handled when VLM returns completely invalid JSON or error responses?
  - Region marked as VLM_FAILED (distinct from VLM_UNCERTAIN); user can retry individual failed regions or batch-retry all failures

- What if two consecutive frames have nearly identical regions (video tracking)?
  - System skips VLM queries for tracked regions to save costs; only new or significantly changed regions are queried per frame; assumes tracking reliability and may require periodic refresh queries

## Requirements

### Functional Requirements

- **FR-001**: System MUST automatically queue all segmented regions in a frame for VLM semantic labeling, regardless of segmentation quality (predicted_iou)
- **FR-002**: System MUST display estimated VLM cost before processing a video, calculated as (number of frames) × (average regions per frame) × (cost per VLM query)
- **FR-003**: Users MUST be able to set a maximum session budget for VLM API costs, and the system MUST pause processing when 95% of the budget is consumed
- **FR-004**: System MUST support frame sampling configuration (e.g., "label every Nth frame") to reduce VLM costs while maintaining semantic uncertainty detection capabilities
- **FR-005**: System MUST track cumulative VLM costs in real-time during processing and display the running total in the GUI
- **FR-006**: System MUST identify semantic uncertainty patterns by clustering VLM_UNCERTAIN regions based on visual similarity or manual grouping
- **FR-007**: System MUST implement VLM API rate limit handling with exponential backoff retry logic to handle burst labeling of many regions
- **FR-008**: System MUST allow users to pause/resume automatic labeling mid-processing, preserving all completed labels and enabling continuation from the stopping point
- **FR-009**: System MUST distinguish between VLM_UNCERTAIN (low confidence) and VLM_FAILED (API error/invalid response) statuses to enable targeted retries
- **FR-010**: System MUST support batch manual labeling of regions grouped by semantic uncertainty patterns (e.g., label all similar "construction equipment" regions at once)
- **FR-011**: System MUST skip VLM queries for tracked regions across consecutive frames to reduce costs, querying only new or significantly changed regions per frame

### Key Entities

- **SemanticLabelingJob**: Represents an automatic labeling session for a video; tracks frames processed, regions labeled, total cost, budget limit, and pause/resume state
- **VLMBatchQuery**: Represents a batch of regions from a frame sent to the VLM together for efficiency; tracks query timestamps, costs, and response statuses
- **SemanticUncertaintyPattern**: Represents a cluster of VLM_UNCERTAIN regions that share visual similarity; includes region IDs, cluster centroid, sample images, and resolution status

## Success Criteria

### Measurable Outcomes

- **SC-001**: Users can process a complete video with automatic semantic labeling of all regions, with the system completing labeling for 100% of detected regions (not just uncertain ones)
- **SC-002**: System accurately estimates VLM costs within 10% margin before processing, enabling informed budget decisions
- **SC-003**: Cost controls prevent budget overruns in 100% of cases—processing automatically pauses when budget limit is reached
- **SC-004**: Semantic uncertainty pattern detection identifies at least 80% of VLM struggles (manually verified by reviewing VLM_UNCERTAIN regions), enabling targeted improvement efforts
- **SC-005**: Frame sampling reduces VLM costs by the configured sampling ratio (e.g., "every 5th frame" reduces costs to ~20% of full processing) while still detecting semantic uncertainty patterns
- **SC-006**: VLM API rate limit handling ensures processing completes without manual intervention, successfully retrying rate-limited requests with exponential backoff
- **SC-007**: Users can pause and resume automatic labeling without data loss, with 100% of in-progress labels preserved and accessible after resuming

## Assumptions

- VLM API costs are predictable and can be estimated per-query (assumed ~$0.005-0.01 per region based on GPT-5.2 vision pricing)
- Frame sampling (every Nth frame) provides sufficient coverage to detect semantic uncertainty patterns without labeling every single frame
- Visual similarity for clustering VLM_UNCERTAIN regions can be approximated using bounding box size, mask overlap, or simple image embeddings without requiring complex similarity models
- Users prefer explicit cost control and transparency over fully automated processing without budget awareness
- VLM API rate limits are per-minute or per-day rather than per-request, making exponential backoff an effective retry strategy
- Pausing/resuming is more valuable than canceling and restarting from scratch, justifying the complexity of state preservation

## Open Questions

None remaining after clarification session.

## Dependencies

- Existing VLM integration (Phase 5 implementation from feature 001)
- Storage service for persisting labeling job state and semantic uncertainty patterns
- Cost calculation utilities for VLM API pricing
- GUI components for budget controls and cost visualization

## Out of Scope

- Real-time video labeling (on-the-fly processing)—this feature focuses on batch offline processing
- Multi-VLM support (using multiple VLM providers)—single VLM provider per session
- Automatic semantic similarity learning—clustering uses simple heuristics, not trained models
- Cross-video semantic uncertainty tracking—patterns are detected per-video, not across entire dataset
