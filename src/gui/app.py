"""Main Streamlit application for Visual Perception Agent."""

import streamlit as st
import sys
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set PyTorch optimization environment variables
os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Filter out expected SAM2 attention warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=UserWarning, module='sam2')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gui.utils.state_management import (
    init_session_state,
    get_state,
    set_state,
    create_new_session,
    update_progress,
    set_error,
    clear_error,
)
from src.gui.components.video_viewer import (
    render_video_upload,
    render_video_info,
    render_frame_navigation,
    render_frame_navigation_range,
)
from src.gui.components.segmentation_viz import (
    render_segmentation_view,
    render_visualization_controls,
    render_confidence_filter,
)
from src.gui.components.vlm_labeling import (
    render_batch_semantic_labeling_workflow,
    render_vlm_labeling_workflow,
    render_vlm_statistics_dashboard,
)
from src.services.video_processor import VideoProcessor
from src.services.segmentation_service import SegmentationService
from src.services.storage_service import StorageService
from src.services.vlm_service import VLMService
from src.services.semantic_labeling_service import SemanticLabelingService
from src.core.config import ConfigManager
from src.utils.logging import setup_logging
from src.models.segmentation_frame import SegmentationFrame, InstanceMask
from src.models.video_session import VideoSession, SessionStatus
import numpy as np
import os
from uuid import uuid4, UUID
from datetime import datetime


# Page configuration
st.set_page_config(
    page_title="Visual Perception Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_app():
    """Initialize application components."""
    # Initialize session state
    init_session_state()

    # Load configuration
    config = ConfigManager(config_path="config.yaml")

    # Setup logging
    setup_logging(
        level=config.logging.level,
        log_file=config.logging.file,
        max_file_size_mb=config.logging.max_file_size_mb,
        backup_count=config.logging.backup_count,
    )

    # Initialize services
    if "services_initialized" not in st.session_state:
        st.session_state.video_processor = VideoProcessor()
        st.session_state.segmentation_service = SegmentationService(
            checkpoint_path=config.sam2.checkpoint,
            config=config.sam2.config,
            device=config.sam2.device,
            confidence_threshold=config.sam2.confidence_threshold,
        )
        st.session_state.storage = StorageService(base_dir=config.storage.sessions_dir)

        # Initialize VLMService if API key is available
        vlm_api_key = os.getenv("OPENAI_API_KEY")
        if vlm_api_key:
            st.session_state.vlm_service = VLMService(api_key=vlm_api_key)
        else:
            st.session_state.vlm_service = None

        # Initialize SemanticLabelingService
        if st.session_state.vlm_service:
            st.session_state.semantic_labeling_service = SemanticLabelingService(
                storage_service=st.session_state.storage,
                vlm_service=st.session_state.vlm_service,
            )
        else:
            st.session_state.semantic_labeling_service = None

        st.session_state.config = config
        st.session_state.services_initialized = True


def render_header():
    """Render application header."""
    st.title("🤖 Visual Perception Agent")
    st.markdown(
        """
        **Self-Improving Video Segmentation System**

        Upload a video to segment objects, detect uncertain regions, and improve the model over time.
        """
    )
    st.divider()


def render_sidebar():
    """Render sidebar with controls and settings."""
    st.sidebar.title("⚙️ Control Panel")

    # Visualization controls
    viz_settings = render_visualization_controls()

    # Confidence filter
    confidence_threshold = render_confidence_filter()

    st.sidebar.divider()

    # Session info
    st.sidebar.subheader("📊 Session Info")
    session_id = get_state("session_id")
    if session_id:
        st.sidebar.text(f"ID: {session_id[:8]}...")
    else:
        st.sidebar.text("No active session")

    # Reset button
    if st.sidebar.button("🔄 Reset Session"):
        from src.gui.utils.state_management import reset_session

        reset_session()
        st.rerun()

    return viz_settings, confidence_threshold


def create_segmentation_frame_from_result(
    session_id: UUID,
    frame_idx: int,
    result,
    metadata: dict,
    storage: StorageService,
) -> SegmentationFrame:
    """Convert SegmentationResult to SegmentationFrame and save masks.

    Args:
        session_id: Session UUID
        frame_idx: Frame index
        result: SegmentationResult from segmentation_service
        metadata: Video metadata dict
        storage: StorageService instance

    Returns:
        SegmentationFrame object
    """
    # Create InstanceMask objects
    instance_masks = []
    session_path = storage.get_session_path(str(session_id))

    for i, (mask, conf, label, bbox) in enumerate(zip(
        result.masks,
        result.confidences,
        result.class_labels,
        result.bboxes,
    )):
        # Create mask path (will be saved on demand if needed)
        mask_filename = f"mask_{frame_idx:06d}_{i:03d}.npz"
        mask_path = session_path / "masks" / mask_filename

        # Calculate mask area
        area = int(np.sum(mask)) if len(mask.shape) > 0 and mask.size > 0 else bbox[2] * bbox[3]

        instance_mask = InstanceMask(
            mask_path=mask_path,
            class_label=label,
            confidence=float(conf),
            bbox=bbox,
            area=area,
            semantic_label=None,
            vlm_query_id=None,
            semantic_label_source=None,
        )
        instance_masks.append(instance_mask)

    # Create SegmentationFrame
    frame = SegmentationFrame(
        id=uuid4(),
        session_id=session_id,
        frame_index=frame_idx,
        timestamp=frame_idx / metadata["fps"],
        image_path=session_path / "frames" / f"frame_{frame_idx:06d}.jpg",
        masks=instance_masks,
        processing_time=0.1,  # Placeholder
        model_version_id=uuid4(),  # Placeholder
        processed_at=datetime.now(),
    )

    return frame


def process_video(video_path: Path):
    """Process uploaded video.

    Args:
        video_path: Path to video file
    """
    try:
        clear_error()
        set_state("processing", True)

        video_processor = st.session_state.video_processor
        segmentation_service = st.session_state.segmentation_service
        storage = st.session_state.storage

        # Create new session
        session_id = create_new_session()
        storage.create_session(session_id)

        # Get frame range from session state
        metadata = get_state("video_metadata")
        start_frame = get_state("start_frame", 0)
        end_frame = get_state("end_frame", metadata["frame_count"] if metadata else None)

        # Create and save VideoSession object
        now = datetime.now()
        video_session = VideoSession(
            id=UUID(session_id),
            filename=video_path.name,
            filepath=video_path,
            upload_timestamp=now,
            status=SessionStatus.PROCESSING,
            metadata=metadata,
            processing_progress=0.0,
            created_at=now,
            updated_at=now,
        )
        storage.save_video_session(video_session)

        # Validate frame range
        if end_frame is None or end_frame > metadata["frame_count"]:
            end_frame = metadata["frame_count"]

        total_frames_to_process = end_frame - start_frame
        set_state("total_frames", total_frames_to_process)
        set_state("processed_start_frame", start_frame)
        set_state("processed_end_frame", end_frame)

        st.success(f"✅ Processing frames {start_frame} to {end_frame - 1} ({total_frames_to_process} frames)")

        # Process frames with progress bar
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        segmentation_results = {}

        for idx, frame in enumerate(video_processor.extract_frame_range(video_path, start_frame, end_frame)):
            # Calculate actual frame index
            frame_idx = start_frame + idx

            # Update progress
            progress = (idx + 1) / total_frames_to_process
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_idx} ({idx + 1}/{total_frames_to_process})")
            update_progress(progress)

            # Save frame
            storage.save_frame(session_id, frame_idx, frame)

            # Segment frame
            result = segmentation_service.segment_frame(frame)

            # Create and save SegmentationFrame to storage (needed for VLM labeling)
            seg_frame = create_segmentation_frame_from_result(
                session_id=UUID(session_id),
                frame_idx=frame_idx,
                result=result,
                metadata=metadata,
                storage=storage,
            )
            storage.save_segmentation_frame(session_id, seg_frame)

            # Store result in session state (for backward compatibility)
            segmentation_results[frame_idx] = {
                "masks": result.masks,
                "confidences": result.confidences,
                "labels": result.class_labels,
                "bboxes": result.bboxes,
            }

        # Update VideoSession status to COMPLETED
        video_session.status = SessionStatus.COMPLETED
        video_session.processing_progress = 1.0
        video_session.updated_at = datetime.now()
        storage.save_video_session(video_session)

        # Save results to state
        set_state("segmentation_results", segmentation_results)
        set_state("processing", False)
        set_state("video_uploaded", True)

        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Processing complete! Processed {total_frames_to_process} frames (frames {start_frame}-{end_frame - 1}).")

    except Exception as e:
        # Update session status to FAILED
        try:
            video_session.status = SessionStatus.FAILED
            video_session.error_message = str(e)
            video_session.updated_at = datetime.now()
            storage.save_video_session(video_session)
        except:
            pass  # Ignore errors in error handler

        set_error(f"Error processing video: {str(e)}")
        st.error(f"❌ {get_state('error_message')}")


def render_main_content(viz_settings: dict, confidence_threshold: float):
    """Render main content area.

    Args:
        viz_settings: Visualization settings
        confidence_threshold: Confidence threshold for filtering
    """
    # Video upload section
    video_path = render_video_upload()

    if video_path is not None and not get_state("video_uploaded"):
        # Extract metadata if not already extracted
        metadata = get_state("video_metadata")
        if not metadata:
            with st.spinner("📊 Extracting video metadata..."):
                video_processor = st.session_state.video_processor
                metadata = video_processor.extract_metadata(video_path)
                set_state("video_metadata", metadata.model_dump() if hasattr(metadata, 'model_dump') else metadata.__dict__)
                st.success(f"✅ Video loaded: {metadata.frame_count} frames")
                st.rerun()

        # Show frame range inputs if video metadata is available
        if metadata:
            st.subheader("⚙️ Processing Options")

            col1, col2 = st.columns(2)

            with col1:
                start_frame = st.number_input(
                    "Start Frame",
                    min_value=0,
                    max_value=metadata["frame_count"] - 1,
                    value=0,
                    step=1,
                    help="First frame to process (0-indexed)",
                )

            with col2:
                end_frame = st.number_input(
                    "End Frame",
                    min_value=start_frame + 1,
                    max_value=metadata["frame_count"],
                    value=min(start_frame + 100, metadata["frame_count"]),
                    step=1,
                    help="Last frame to process (exclusive)",
                )

            # Calculate and display frame range info
            num_frames = end_frame - start_frame
            duration = num_frames / metadata["fps"]
            st.info(f"📊 Will process {num_frames} frames (~{duration:.1f} seconds)")

            # Store frame range in session state
            set_state("start_frame", start_frame)
            set_state("end_frame", end_frame)

            st.divider()

        if st.button("🚀 Start Processing", type="primary"):
            process_video(video_path)

    # Show video info if metadata available
    metadata = get_state("video_metadata")
    if metadata:
        render_video_info(metadata)
        st.divider()

    # Show results if video is processed
    if get_state("video_uploaded"):
        # Frame navigation
        start_frame = get_state("processed_start_frame", 0)
        end_frame = get_state("processed_end_frame", 1)
        current_frame = get_state("current_frame_index", start_frame)

        # Ensure current frame is within processed range
        if current_frame < start_frame or current_frame >= end_frame:
            current_frame = start_frame
            set_state("current_frame_index", current_frame)

        frame_index = render_frame_navigation_range(current_frame, start_frame, end_frame)
        set_state("current_frame_index", frame_index)

        st.divider()

        # Load and display frame
        session_id = get_state("session_id")
        storage = st.session_state.storage

        try:
            frame = storage.load_frame(session_id, frame_index)

            # Load segmentation frame from storage (includes VLM labels)
            try:
                seg_frame = storage.load_segmentation_frame(session_id, frame_index)

                # Build frame_result from stored segmentation frame
                frame_result = {
                    "masks": np.array([]),  # Masks not needed for visualization
                    "confidences": np.array([mask.confidence for mask in seg_frame.masks]),
                    "labels": [mask.class_label for mask in seg_frame.masks],
                    "bboxes": [mask.bbox for mask in seg_frame.masks],
                    "vlm_labels": [mask.semantic_label for mask in seg_frame.masks],
                    "vlm_sources": [mask.semantic_label_source for mask in seg_frame.masks],
                }
            except Exception:
                # Fallback to state if frame not in storage
                segmentation_results = get_state("segmentation_results", {})
                frame_result = segmentation_results.get(frame_index)
                if frame_result:
                    frame_result["vlm_labels"] = [None] * len(frame_result.get("labels", []))
                    frame_result["vlm_sources"] = [None] * len(frame_result.get("labels", []))

            # Load VLM labels (uncertain regions) for pattern detection
            try:
                uncertain_regions = storage.load_uncertain_regions_by_frame(session_id, frame_index)
            except Exception:
                uncertain_regions = []

            # Filter by confidence if results exist
            if frame_result and len(frame_result["confidences"]) > 0:
                mask_indices = frame_result["confidences"] >= confidence_threshold
                if mask_indices.any():
                    frame_result = {
                        "masks": frame_result.get("masks", np.array([]))[mask_indices] if len(frame_result.get("masks", [])) > 0 else np.array([]),
                        "confidences": frame_result["confidences"][mask_indices],
                        "labels": [
                            label
                            for i, label in enumerate(frame_result["labels"])
                            if mask_indices[i]
                        ],
                        "bboxes": [
                            bbox
                            for i, bbox in enumerate(frame_result["bboxes"])
                            if mask_indices[i]
                        ],
                        "vlm_labels": [
                            vlm_label
                            for i, vlm_label in enumerate(frame_result.get("vlm_labels", []))
                            if mask_indices[i]
                        ],
                        "vlm_sources": [
                            vlm_source
                            for i, vlm_source in enumerate(frame_result.get("vlm_sources", []))
                            if mask_indices[i]
                        ],
                    }

            # Render segmentation view with VLM labels
            render_segmentation_view(
                frame,
                frame_result,
                show_masks=viz_settings["show_masks"],
                show_bboxes=viz_settings["show_bboxes"],
                show_labels=viz_settings["show_labels"],
                alpha=viz_settings["alpha"],
                uncertain_regions=uncertain_regions,
            )

        except Exception as e:
            st.error(f"❌ Error loading frame: {str(e)}")

        # VLM Labeling Section
        st.divider()

        # Check if VLM service is available
        vlm_service = st.session_state.get("vlm_service")
        semantic_labeling_service = st.session_state.get("semantic_labeling_service")

        if vlm_service and semantic_labeling_service:
            # Batch Semantic Labeling Workflow
            try:
                # Load VideoSession to get video path
                video_session = storage.load_video_session(session_id)
                video_path = video_session.filepath

                render_batch_semantic_labeling_workflow(
                    session_id=session_id,
                    video_path=video_path,
                    semantic_labeling_service=semantic_labeling_service,
                    storage_service=storage,
                )
            except Exception as e:
                st.error(f"❌ Error rendering batch semantic labeling: {str(e)}")

            st.divider()

            # Manual VLM Labeling Workflow (for uncertain regions)
            try:
                region_ids = storage.list_uncertain_regions(session_id)
                uncertain_regions = [
                    storage.load_uncertain_region(session_id, region_id)
                    for region_id in region_ids
                ]

                # Render manual VLM labeling workflow
                render_vlm_labeling_workflow(
                    uncertain_regions=uncertain_regions,
                    session_id=session_id,
                    vlm_service=vlm_service,
                    storage_service=storage,
                )

            except Exception as e:
                st.error(f"❌ Error loading uncertain regions: {str(e)}")
        else:
            st.warning(
                "⚠️ VLM service not available. Set OPENAI_API_KEY environment variable to enable VLM-assisted labeling."
            )


def main():
    """Main application entry point."""
    # Initialize app
    initialize_app()

    # Render header
    render_header()

    # Render sidebar and get settings
    viz_settings, confidence_threshold = render_sidebar()

    # Render main content
    render_main_content(viz_settings, confidence_threshold)

    # Show processing indicator
    if get_state("processing"):
        st.info("⏳ Processing video... Please wait.")

    # Show error if any
    error_message = get_state("error_message")
    if error_message:
        st.error(f"❌ {error_message}")


if __name__ == "__main__":
    main()
