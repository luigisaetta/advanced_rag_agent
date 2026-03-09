"""
File name: pages/post_answer_eval_ui.py
Author: Luigi Saetta
Last modified: 06-03-2026
Python Version: 3.11
License: MIT
Description: Streamlit page to inspect POST_ANSWER_EVALUATION records.
"""

from datetime import date
import json

import pandas as pd
import streamlit as st

from config import MODEL_LIST
from core.post_answer_feedback import PostAnswerFeedback
from core.utils import get_console_logger

logger = get_console_logger()

ROOT_CAUSE_OPTIONS = ["ALL", "NO_ISSUE", "RETRIEVAL", "RERANK", "GENERATION"]
LLM_MODEL_OPTIONS = ["ALL"] + list(dict.fromkeys(MODEL_LIST))


def _truncate(text: str, max_len: int = 160) -> str:
    """
    Truncate long text for compact table rendering.
    """
    value = str(text or "").strip().replace("\n", " ")
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _safe_date(value):
    """
    Return date object when available.
    """
    if value is None:
        return None
    if hasattr(value, "date"):
        try:
            return value.date()
        except Exception:
            return None
    return value if isinstance(value, date) else None


def _to_table_df(records: list[dict]) -> pd.DataFrame:
    """
    Normalize records for table display.
    """
    rows = []
    for item in records:
        created_at = item.get("created_at")
        rows.append(
            {
                "id": item.get("id"),
                "created_at": (
                    created_at.strftime("%Y-%m-%d %H:%M:%S")
                    if hasattr(created_at, "strftime")
                    else str(created_at or "")
                ),
                "root_cause": item.get("root_cause", ""),
                "confidence": item.get("confidence"),
                "llm_model_id": item.get("llm_model_id", ""),
                "question_preview": _truncate(item.get("question", ""), max_len=140),
                "reason_preview": _truncate(item.get("reason", ""), max_len=180),
            }
        )
    return pd.DataFrame(rows)


def _root_cause_stats(records: list[dict]) -> dict[str, int]:
    """
    Compute counts for key root-cause categories on filtered records.
    """
    counts = {"NO_ISSUE": 0, "RERANKER": 0, "RETRIEVAL": 0, "GENERATION": 0}
    for item in records:
        value = str(item.get("root_cause") or "").strip().upper()
        if value == "NO_ISSUE":
            counts["NO_ISSUE"] += 1
        elif value in {"RERANK", "RERANKER"}:
            counts["RERANKER"] += 1
        elif value in {"RETRIEVAL", "RETRIEVASL"}:
            counts["RETRIEVAL"] += 1
        elif value == "GENERATION":
            counts["GENERATION"] += 1
    return counts


st.set_page_config(page_title="Post Answer Evaluation", layout="wide")
st.title("Post Answer Evaluation")
st.caption("Browse evaluator outcomes with filters and detailed view per record.")

col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
with col1:
    max_rows = st.number_input(
        "Max rows", min_value=10, max_value=2000, value=200, step=10
    )
with col2:
    root_cause_filter = st.selectbox("Root cause", ROOT_CAUSE_OPTIONS, index=0)
with col3:
    use_date_from = st.checkbox("Use from date", value=False)
    date_from = st.date_input(
        "From date", value=date.today(), disabled=not use_date_from
    )
with col4:
    use_date_to = st.checkbox("Use to date", value=False)
    date_to = st.date_input("To date", value=date.today(), disabled=not use_date_to)
with col5:
    llm_model_id_filter = st.selectbox("LLM model id", LLM_MODEL_OPTIONS, index=0)

filters_col1, filters_col2 = st.columns([1, 5])
with filters_col1:
    refresh = st.button("Refresh")
with filters_col2:
    st.caption("Tip: leave dates empty to query all history.")

if refresh or "post_answer_eval_records" not in st.session_state:
    try:
        filters = {
            "max_rows": int(max_rows),
            "root_cause": None if root_cause_filter == "ALL" else root_cause_filter,
            "date_from": _safe_date(date_from) if use_date_from else None,
            "date_to": _safe_date(date_to) if use_date_to else None,
            "llm_model_id": (
                None if llm_model_id_filter == "ALL" else llm_model_id_filter
            ),
        }
        records = PostAnswerFeedback().list_feedback(**filters)
        st.session_state.post_answer_eval_records = records
        logger.info(
            "Loaded post-answer evaluation rows: count=%d root_cause=%s date_from=%s date_to=%s llm_model_id=%s",
            len(records),
            filters["root_cause"],
            filters["date_from"],
            filters["date_to"],
            filters["llm_model_id"],
        )
    except Exception as exc:
        logger.error("Failed to load post-answer evaluation rows: %s", exc)
        st.error(str(exc))
        st.stop()

records = st.session_state.get("post_answer_eval_records", [])

if not records:
    st.info("No records found for the selected filters.")
    st.stop()

df = _to_table_df(records)
st.dataframe(df, use_container_width=True, height=480, hide_index=True)

stats = _root_cause_stats(records)
st.markdown("**Statistics (current filters)**")
stat_cols = st.columns(5)
stat_cols[0].metric("Total cases", len(records))
stat_cols[1].metric("NO_ISSUE", stats["NO_ISSUE"])
stat_cols[2].metric("RERANKER", stats["RERANKER"])
stat_cols[3].metric("RETRIEVAL", stats["RETRIEVAL"])
stat_cols[4].metric("GENERATION", stats["GENERATION"])

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download table as CSV",
    data=csv_bytes,
    file_name="post_answer_evaluation.csv",
    mime="text/csv",
)

id_choices = [int(item.get("id")) for item in records]
selected_id = st.selectbox("Record details (ID)", id_choices, index=0)
selected = next((item for item in records if int(item.get("id")) == selected_id), None)

if selected:
    st.subheader("Details")
    det_col1, det_col2 = st.columns([1, 1])
    with det_col1:
        st.markdown(f"**Created at:** {selected.get('created_at')}")
        st.markdown(f"**Root cause:** {selected.get('root_cause')}")
        st.markdown(f"**Confidence:** {selected.get('confidence')}")
    with det_col2:
        st.markdown(f"**ID:** {selected.get('id')}")
        cfg = selected.get("config_json", {})
        model_id = (
            (cfg.get("llm_model_id") or cfg.get("model_id", ""))
            if isinstance(cfg, dict)
            else ""
        )
        st.markdown(f"**Model:** {model_id}")

    with st.expander("Question", expanded=False):
        st.write(selected.get("question", ""))
    with st.expander("Reason", expanded=True):
        st.write(selected.get("reason", ""))
    with st.expander("Config JSON", expanded=False):
        st.code(json.dumps(selected.get("config_json", {}), indent=2), language="json")
