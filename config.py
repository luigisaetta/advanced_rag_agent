"""
File name: config.py
Author: Luigi Saetta
Last modified: 24-02-2026
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


DEBUG = False

# type of auth
AUTH = "API_KEY"

# embeddings
EMBED_MODEL_ID = "cohere.embed-v4.0"
# EMBED_MODEL_ID = "cohere.embed-multilingual-image-v3.0"

# this one needs to specify the dimension, default is 1536
# EMBED_MODEL_ID = "cohere.embed-v4.0"

# LLM
# this is the default model
LLM_MODEL_ID = "openai.gpt-oss-120b"
# dedicated model for intent classification routing
INTENT_MODEL_ID = "openai.gpt-oss-120b"
# dedicated model for reranker
RERANKER_MODEL_ID = "openai.gpt-oss-120b"
# VLM used to OCR scanned PDFs uploaded in-session from UI
VLM_MODEL_ID = "openai.gpt-5.2"
# deterministic
TEMPERATURE = 0.0
# increased to support hybrid search with reranker
MAX_TOKENS = 8000
# transient failures (e.g., safety false positives / rate limits)
LLM_MAX_RETRIES = 2

# max number of pages for in-memory session PDF scan
SESSION_PDF_MAX_PAGES = 30


# OCI general
# Separate OCI regions for LLM and embeddings.
# Requested setup:
# - LLM in Frankfurt
# - Embeddings in Chicago
LLM_REGION = "eu-frankfurt-1"
EMBED_REGION = "us-chicago-1"
# Backward-compatible alias used by some UI/read-only references.
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

# answer language (fixed, no UI override)
# allowed values: "same as the question", "en", "fr", "it", "es"
MAIN_LANGUAGE = "same as the question"

if LLM_REGION == "us-chicago-1":
    MODEL_LIST = [
        "openai.gpt-oss-120b",
        "openai.gpt-5.2",
        "google.gemini-2.5-pro",
        "meta.llama-3.3-70b-instruct",
        "cohere.command-a-03-2025",
    ]
else:
    MODEL_LIST = [
        "openai.gpt-oss-120b",
        "openai.gpt-5.2",
        "google.gemini-2.5-pro",
        "meta.llama-3.3-70b-instruct",
        "cohere.command-a-03-2025",
    ]


ENABLE_USER_FEEDBACK = True

# semantic search
TOP_K = 10
# to enable/disable hybrid search (BM25 + semantic)
ENABLE_HYBRID_SEARCH = True
HYBRID_TOP_K = TOP_K
# conservative number of in-memory session chunks added when intent is HYBRID
HYBRID_SESSION_TOP_K = 8
# minimum number of session_pdf chunks preserved in HYBRID final context
HYBRID_MIN_SESSION_DOCS = 6
# number of session chunks used to build a KB-focused query in HYBRID
HYBRID_QUERY_EXPANSION_TOP_K = 3
# max chars from session chunks passed to query-expansion prompt
HYBRID_QUERY_EXPANSION_MAX_CHARS = 3500

# advanced analysis planner: max number of actions (retrievals or generations)
# that the planner can suggest to the agent when the user request is complex and requires multiple steps of reasoning
ADVANCED_ANALYSIS_MAX_ACTIONS = 5
# advanced analysis execution settings
ADVANCED_ANALYSIS_KB_TOP_K = 6
ADVANCED_ANALYSIS_STEP_MAX_WORDS = 450

# BM25 cache warms up from all the collections in this list
COLLECTION_LIST = ["COLL01", "CONTRATTI", "BOOKS"]
DEFAULT_COLLECTION = "COLL01"
# optional persistence path for serialized BM25 cache file
# default points to a repository-local folder (can be overridden via env var)
BM25_CACHE_DIR = os.getenv("BM25_CACHE_DIR", "bm25_cache")


# history management (put -1 if you want to disable trimming)
# consider that we have pair (human, ai) so use an even (ex: 6) value
MAX_MSGS_IN_HISTORY = 10

# reranking enabled or disabled from UI

# Integration with APM
ENABLE_TRACING = True
AGENT_NAME = "OCI_CUSTOM_RAG_AGENT"

# lsaetta-apm compartment
# APM_BASE_URL = "https://aaaadec2jjn3maaaaaaaaach4e.apm-agt.eu-frankfurt-1.oci.oraclecloud.com/20200101"
# sviluppoteng
APM_BASE_URL = "https://aaaadhetxjknmaaaaaaaaac7wy.apm-agt.eu-frankfurt-1.oci.oraclecloud.com/20200101"
APM_CONTENT_TYPE = "application/json"

# for loading
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 100

# section for citation server
CITATION_SERVER_PORT = int(os.getenv("CITATION_SERVER_PORT", "8008"))
CITATION_BASE_URL = os.getenv(
    "CITATION_BASE_URL", f"http://127.0.0.1:{CITATION_SERVER_PORT}/"
)
