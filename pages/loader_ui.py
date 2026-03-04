"""
File name: pages/loader_ui.py
Author: Luigi Saetta
Last modified: 25-02-2026
Python Version: 3.11
License: MIT
Description: Streamlit utility page to upload documents and inspect loaded documents per collection.
"""

import os
import tempfile

import pandas as pd
import streamlit as st

from agent.vector_search import SemanticSearch
from config import DEBUG, COLLECTION_LIST, VLM_MODEL_ID
from core.db_utils import list_collections, list_books_in_collection
from core.session_pdf_vlm import scan_pdf_to_docs_with_vlm
from core.utils import get_console_logger

# init session
if COLLECTION_LIST:
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = COLLECTION_LIST[0]
else:
    st.error("No collections available.")

if "show_documents" not in st.session_state:
    st.session_state.show_documents = False


header_area = st.container()
table_area = st.container()

with header_area:
    st.header("Loading Utility")

logger = get_console_logger()


#
# Supporting functions
#
def list_books(_collection_name):
    """
    return the list of books in the given collection
    """
    _books_list = list_books_in_collection(collection_name=_collection_name)

    # reorder
    return sorted(_books_list)


def show_documents_in_collection(_collection_name):
    """
    show the documents in the given collection
    """
    with st.spinner():
        _books_list = list_books(_collection_name)

        books_names = [item[0] for item in _books_list]
        books_chunks = [item[1] for item in _books_list]

        # convert in a Pandas DataFrame for Visualization
        df_list = pd.DataFrame({"Document": books_names, "Num. chunks": books_chunks})
        # index starting by 1
        df_list.index = range(1, len(df_list) + 1)
        # visualize
        with table_area:
            st.table(df_list)


def _document_exists_in_collection(_collection_name: str, document_name: str) -> bool:
    """
    Return True when a document with the same source name already exists.
    """
    existing = {row[0] for row in list_books(_collection_name)}
    return document_name in existing


def on_selection_change():
    """
    React to the selection of the collection
    """
    selected = st.session_state["name_selected"]

    logger.info("Collection list selected: %s", selected)


st.session_state.collection_name = st.sidebar.selectbox(
    "Collection name",
    list_collections(),
    key="name_selected",
    on_change=on_selection_change,
)

# replaced with a button
show_doc = st.sidebar.button("Show documents")

uploaded_file = st.sidebar.file_uploader("Upload a file", type=["pdf"])

# added a button for loading
load_file = st.sidebar.button("Load file")

if show_doc:
    show_documents_in_collection(st.session_state.collection_name)

if uploaded_file is not None and load_file:
    # identify file
    only_name = os.path.basename(uploaded_file.name)
    if DEBUG:
        logger.info("Uploaded file: %s", only_name)

    progress_text = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0)

    # check that the file is not already in the collection
    if _document_exists_in_collection(st.session_state.collection_name, only_name):
        progress_bar.empty()
        progress_text.empty()
        st.error(f"{only_name} already in collection")
    else:
        logger.info("Loading %s ...", only_name)

        tmp_path = ""
        try:
            progress_text.info("Saving uploaded file...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            progress_bar.progress(10)

            progress_text.info("Scanning PDF pages with VLM...")

            def _on_page_progress(current_page: int, total_pages: int):
                """
                Update progress while scanning PDF pages.
                """
                progress_text.info(
                    f"Scanning PDF pages with VLM... ({current_page}/{total_pages})"
                )
                if total_pages > 0:
                    pct = 10 + int((current_page / total_pages) * 70)
                    progress_bar.progress(min(pct, 80))

            docs, page_count = scan_pdf_to_docs_with_vlm(
                pdf_path=tmp_path,
                vlm_model_id=VLM_MODEL_ID,
                max_pages=-1,
                source_name=only_name,
                on_progress=_on_page_progress,
                metadata_retrieval_type=None,
            )

            progress_text.info("Indexing chunks in Oracle Vector Store...")
            progress_bar.progress(90)

            if docs:
                SemanticSearch().add_documents(
                    docs, collection_name=st.session_state.collection_name
                )

            progress_bar.progress(100)
            progress_text.success(
                f"Document loaded ({page_count} pages, {len(docs)} chunks)."
            )
            st.success(f"{only_name} loaded in {st.session_state.collection_name}")

        except (ValueError, OSError, RuntimeError) as exc:
            logger.error("Error while loading document: %s", exc)
            progress_text.error("Document loading failed.")
            st.error(str(exc))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
