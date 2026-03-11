"""
UI access control helpers based on authenticated user profile.
"""

import streamlit as st

from core.user_profile import get_user_profile
from ui.session import get_authenticated_user


def _resolve_session_user_profile() -> str:
    """
    Resolve and cache user profile in session state.
    """
    username = get_authenticated_user(st.session_state.get("authenticated_user", ""))
    st.session_state.authenticated_user = username

    if (
        "user_profile" not in st.session_state
        or st.session_state.get("user_profile_username") != username
    ):
        st.session_state.user_profile = get_user_profile(username)
        st.session_state.user_profile_username = username

    return str(st.session_state.get("user_profile", "USER")).upper()


def is_admin_user() -> bool:
    """
    Return True when current authenticated user has ADMIN profile.
    """
    return _resolve_session_user_profile() == "ADMIN"


def hide_admin_only_pages_in_sidebar() -> None:
    """
    Hide admin-only pages from Streamlit sidebar navigation for non-admin users.
    """
    if is_admin_user():
        return

    st.markdown(
        """
        <style>
        section[data-testid="stSidebarNav"] li:has(a[href*="loader_ui"]),
        section[data-testid="stSidebarNav"] li:has(a[href*="post_answer_eval_ui"]) {
            display: none !important;
        }
        section[data-testid="stSidebarNav"] a[href*="loader_ui"],
        section[data-testid="stSidebarNav"] a[href*="post_answer_eval_ui"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def require_admin_page_access() -> None:
    """
    Stop page execution when user is not ADMIN.
    """
    if is_admin_user():
        return
    st.error("Access denied. This page is available only for ADMIN users.")
    st.stop()
