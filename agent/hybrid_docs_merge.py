"""
File name: hybrid_docs_merge.py
Author: Luigi Saetta
Last modified: 02-03-2026
Python Version: 3.11

Description:
    This module merges KB and session-PDF candidates in the HYBRID subgraph.
    It deduplicates by normalized content.

Usage:
    Import this module into other scripts to use its functions.
    Example:
        from agent.hybrid_docs_merge import HybridDocsMerge

License:
    This code is released under the MIT License.
"""

from langchain_core.runnables import Runnable

from agent.agent_state import State


class HybridDocsMerge(Runnable):
    """
    Merge `retriever_docs` (KB side) with `session_retriever_docs` (session side),
    deduplicating by normalized content.
    """

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Helper for normalize text."""
        return " ".join((text or "").split()).strip().lower()

    def invoke(self, input: State, config=None, **kwargs):
        """Invoke."""
        kb_docs = list(input.get("retriever_docs", []))
        session_docs = list(input.get("session_retriever_docs", []))
        error = input.get("error")

        merged = list(kb_docs)
        seen = {
            self._normalize_text(doc.get("page_content", ""))
            for doc in merged
            if self._normalize_text(doc.get("page_content", ""))
        }

        for doc in session_docs:
            content = self._normalize_text(doc.get("page_content", ""))
            if not content or content in seen:
                continue
            merged.append(doc)
            seen.add(content)

        return {"retriever_docs": merged, "error": error}
