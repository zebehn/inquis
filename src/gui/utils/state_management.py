"""Streamlit session state management utilities."""

import streamlit as st
from typing import Any, Optional
from uuid import uuid4


def init_session_state() -> None:
    """Initialize session state variables if they don't exist."""
    defaults = {
        "session_id": None,
        "video_uploaded": False,
        "video_path": None,
        "video_metadata": None,
        "processing": False,
        "processing_progress": 0.0,
        "current_frame_index": 0,
        "total_frames": 0,
        "segmentation_results": {},
        "error_message": None,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def get_state(key: str, default: Any = None) -> Any:
    """Get value from session state.

    Args:
        key: State key
        default: Default value if key doesn't exist

    Returns:
        State value or default
    """
    return st.session_state.get(key, default)


def set_state(key: str, value: Any) -> None:
    """Set value in session state.

    Args:
        key: State key
        value: Value to set
    """
    st.session_state[key] = value


def create_new_session() -> str:
    """Create a new session ID.

    Returns:
        New session ID
    """
    session_id = str(uuid4())
    set_state("session_id", session_id)
    return session_id


def reset_session() -> None:
    """Reset all session state to defaults."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()


def update_progress(progress: float) -> None:
    """Update processing progress.

    Args:
        progress: Progress value (0.0 to 1.0)
    """
    set_state("processing_progress", progress)


def set_error(message: str) -> None:
    """Set error message.

    Args:
        message: Error message
    """
    set_state("error_message", message)
    set_state("processing", False)


def clear_error() -> None:
    """Clear error message."""
    set_state("error_message", None)
