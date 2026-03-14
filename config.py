"""
File name: config.py
Author: Luigi Saetta
Last modified: 13-03-2026
Python Version: 3.11

Description:
    This module provides general configurations


Usage:
    Import this module into other scripts to use its functions.
    Example:
        import config

License:
    This code is released under the MIT License.

Notes:
    This is a part of a demo showing how to implement an advanced
    RAG solution as a LangGraph agent.

Warnings:
    This module is in development, may change in future versions.
"""

import os

from prompt_profiles import DEFAULT_PROMPT_PROFILE

# ---------------------------------------------------------------------------
# Core Runtime
# ---------------------------------------------------------------------------
# Global debug switch used by logs and diagnostic behavior.
DEBUG = False

# Authentication mode used by the app integrations.
AUTH = "API_KEY"


# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------
# Embedding model used for vectorization/retrieval.
EMBED_MODEL_ID = "cohere.embed-v4.0"
# EMBED_MODEL_ID = "cohere.embed-multilingual-image-v3.0"

# Default generation model.
LLM_MODEL_ID = "openai.gpt-oss-120b"
# Dedicated model for intent classification routing.
INTENT_MODEL_ID = "openai.gpt-oss-120b"
# Dedicated model for reranker.
RERANKER_MODEL_ID = "openai.gpt-oss-120b"
# VLM used to OCR scanned PDFs uploaded in-session from UI.
VLM_MODEL_ID = "openai.gpt-5.2"
# Generation parameters.
TEMPERATURE = 0.0
MAX_TOKENS = 8000
# for Cohere
# MAX_TOKENS = 4000
# Retry budget for transient failures (rate limits, temporary safety blocks).
LLM_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# OCI Endpoints
# ---------------------------------------------------------------------------
# Separate OCI regions for LLM and embeddings.
LLM_REGION = "eu-frankfurt-1"
EMBED_REGION = "us-chicago-1"
# Backward-compatible alias used by legacy references.
REGION = LLM_REGION
COMPARTMENT_ID = "ocid1.compartment.oc1..aaaaaaaaushuwb2evpuf7rcpl4r7ugmqoe7ekmaiik3ra3m7gec3d234eknq"
LLM_SERVICE_ENDPOINT = (
    f"https://inference.generativeai.{LLM_REGION}.oci.oraclecloud.com"
)
EMBED_SERVICE_ENDPOINT = (
    f"https://inference.generativeai.{EMBED_REGION}.oci.oraclecloud.com"
)
# Backward-compatible alias for legacy imports.
SERVICE_ENDPOINT = LLM_SERVICE_ENDPOINT


# ---------------------------------------------------------------------------
# UI/UX Defaults
# ---------------------------------------------------------------------------
# Answer language (fixed, no UI override):
# allowed values: "same as the question", "en", "fr", "it", "es".
MAIN_LANGUAGE = "same as the question"
# Prompt profile name; prompt-domain profiles are defined in prompt_profiles.py.
PROMPT_PROFILE = os.getenv("PROMPT_PROFILE", DEFAULT_PROMPT_PROFILE)
# Enables user feedback widgets/flows in UI.
ENABLE_USER_FEEDBACK = True
# History management: set -1 to disable trimming.
# Since history stores human/ai pairs, an even number is recommended.
MAX_MSGS_IN_HISTORY = 10
# Max number of pages for in-memory session PDF scan.
SESSION_PDF_MAX_PAGES = 30


# ---------------------------------------------------------------------------
# Available Models
# ---------------------------------------------------------------------------
# Model choices shown in UI selector.
if LLM_REGION == "us-chicago-1":
    MODEL_LIST = [
        "openai.gpt-oss-120b",
        "openai.gpt-5.2",
        "openai.gpt-5.4",
        "google.gemini-2.5-pro",
        "meta.llama-3.3-70b-instruct",
        "cohere.command-a-03-2025",
    ]
else:
    MODEL_LIST = [
        "openai.gpt-oss-120b",
        "openai.gpt-5.2",
        "openai.gpt-5.4",
        "google.gemini-2.5-pro",
        "meta.llama-3.3-70b-instruct",
        "cohere.command-a-03-2025",
    ]


# ---------------------------------------------------------------------------
# Retrieval and Ranking
# ---------------------------------------------------------------------------
# Semantic retrieval depth.
TOP_K = 10
# Enable/disable hybrid retrieval (BM25 + semantic).
ENABLE_HYBRID_SEARCH = True
HYBRID_TOP_K = TOP_K
# Conservative number of in-memory session chunks added when intent is HYBRID.
HYBRID_SESSION_TOP_K = 8
# Minimum number of session_pdf chunks preserved in HYBRID final context.
HYBRID_MIN_SESSION_DOCS = 6
# Number of session chunks used to build a KB-focused query in HYBRID.
HYBRID_QUERY_EXPANSION_TOP_K = 3
# Max chars from session chunks passed to query-expansion prompt.
HYBRID_QUERY_EXPANSION_MAX_CHARS = 3500


# ---------------------------------------------------------------------------
# Advanced Analysis
# ---------------------------------------------------------------------------
# Planner cap: max actions (retrievals/generations) for complex multi-step tasks.
ADVANCED_ANALYSIS_MAX_ACTIONS = 5
# Execution settings.
ADVANCED_ANALYSIS_KB_TOP_K = 6
ADVANCED_ANALYSIS_STEP_MAX_WORDS = 450
# Optional post-synthesis risk-validation step.
ADVANCED_ANALYSIS_ENABLE_RISK_VALIDATION = False
ADVANCED_ANALYSIS_RISK_VALIDATION_KB_TOP_K = 4


# ---------------------------------------------------------------------------
# Post-Answer Evaluation
# ---------------------------------------------------------------------------
# Log-only evaluator for hybrid retrieval runs without session PDF.
POST_ANSWER_EVALUATION_ENABLED = True
POST_ANSWER_EVALUATION_MODEL_ID = "openai.gpt-5.4"
POST_ANSWER_EVALUATION_MAX_CHARS = 12000


# ---------------------------------------------------------------------------
# Knowledge Base and Cache
# ---------------------------------------------------------------------------
# BM25 cache warms up from all collections in this list.
COLLECTION_LIST = ["COLL01", "CONTRATTI", "BOOKS", "ARERA"]
DEFAULT_COLLECTION = "COLL01"
# Optional persistence path for serialized BM25 cache data.
BM25_CACHE_DIR = os.getenv("BM25_CACHE_DIR", "bm25_cache")


# ---------------------------------------------------------------------------
# Observability (APM)
# ---------------------------------------------------------------------------
ENABLE_TRACING = True
AGENT_NAME = "OCI_ADVANCED_RAG_AGENT"

# APM endpoint in OCI.
# APM_BASE_URL = "https://aaaadec2jjn3maaaaaaaaach4e.apm-agt.eu-frankfurt-1.oci.oraclecloud.com/20200101"
APM_BASE_URL = "https://aaaadhetxjknmaaaaaaaaac7wy.apm-agt.eu-frankfurt-1.oci.oraclecloud.com/20200101"
APM_CONTENT_TYPE = "application/json"


# ---------------------------------------------------------------------------
# Ingestion / Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 100


# ---------------------------------------------------------------------------
# Citation Service
# ---------------------------------------------------------------------------
CITATION_SERVER_PORT = int(os.getenv("CITATION_SERVER_PORT", "8008"))
CITATION_BASE_URL = os.getenv("CITATION_BASE_URL", "/citations/")
