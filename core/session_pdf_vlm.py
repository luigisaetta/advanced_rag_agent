"""
Session PDF scanning via VLM OCR for the main Streamlit UI.
"""

import base64
from io import BytesIO
from typing import Callable, List, Optional, Tuple

import pypdfium2 as pdfium
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from core.chunk_index_utils import get_recursive_text_splitter
from core.oci_models import get_llm
from core.utils import get_console_logger, remove_path_from_ref
from config import SESSION_PDF_MAX_PAGES

logger = get_console_logger()


def _extract_text_from_vlm_response(content) -> str:
    """
    Normalize LangChain model response content to text.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def _page_to_data_url(page) -> str:
    """
    Render one PDF page and return a PNG data URL.
    """
    bitmap = page.render(scale=2.0)
    image = bitmap.to_pil()
    buf = BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _ocr_page_with_vlm(llm, page_data_url: str) -> str:
    """
    OCR one page image using a VLM.
    """
    prompt = (
        "Extract all readable text from this document page. "
        "Keep original language. "
        "Return only the extracted text, no explanations."
    )

    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": page_data_url}},
            ]
        )
    ]

    response = llm.invoke(messages)
    return _extract_text_from_vlm_response(response.content)


def scan_pdf_to_docs_with_vlm(
    pdf_path: str,
    vlm_model_id: str,
    max_pages: int = SESSION_PDF_MAX_PAGES,
    source_name: Optional[str] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
    metadata_retrieval_type: Optional[str] = "session_pdf_vlm",
    split_per_page: bool = True,
) -> Tuple[List[Document], int]:
    """
    Scan a PDF with VLM OCR and return split LangChain documents plus total page count.
    """
    pdf = pdfium.PdfDocument(pdf_path)
    total_pages = len(pdf)

    if 0 < max_pages < total_pages:
        raise ValueError(
            f"PDF has {total_pages} pages; maximum allowed is {max_pages} pages."
        )

    llm = get_llm(model_id=vlm_model_id)
    text_splitter = None if split_per_page else get_recursive_text_splitter()
    effective_source_name = remove_path_from_ref(source_name or pdf_path)
    # Use uploaded filename (when provided) so references/citations show real doc name.
    doc_title = effective_source_name.rsplit(".", 1)[0]
    chunk_header = f"# Doc. title: {doc_title}\n"

    docs: List[Document] = []

    for idx in range(total_pages):
        page = pdf[idx]
        page_data_url = _page_to_data_url(page)
        page_text = _ocr_page_with_vlm(llm, page_data_url)

        if page_text:
            splits = (
                [page_text.strip()]
                if split_per_page
                else text_splitter.split_text(page_text)
            )
            for chunk in splits:
                if not chunk:
                    continue
                docs.append(
                    Document(
                        page_content=chunk_header + chunk,
                        metadata=(
                            {
                                "source": effective_source_name,
                                "page_label": str(idx + 1),
                                "retrieval_type": metadata_retrieval_type,
                            }
                            if metadata_retrieval_type
                            else {
                                "source": effective_source_name,
                                "page_label": str(idx + 1),
                            }
                        ),
                    )
                )

        if on_progress:
            on_progress(idx + 1, total_pages)

    logger.info(
        "Session VLM scan completed for %s: pages=%d chunks=%d model=%s",
        effective_source_name,
        total_pages,
        len(docs),
        vlm_model_id,
    )
    return docs, total_pages
