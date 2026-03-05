"""
File name: prompt_profiles.py
Author: Luigi Saetta
Last modified: 05-03-2026
Python Version: 3.11

Description:
    This module provides centralized prompt domain profiles.


Usage:
    Import this module into other scripts to use its functions.
    Example:
        from prompt_profiles import PROMPT_PROFILES, DEFAULT_PROMPT_PROFILE

License:
    This code is released under the MIT License.

Notes:
    This is a part of a demo showing how to implement an advanced
    RAG solution as a LangGraph agent.

Warnings:
    This module is in development, may change in future versions.
"""

DEFAULT_PROMPT_PROFILE = "general"

PROMPT_PROFILES = {
    "general": {
        "name": "General Purpose",
        "instructions": (
            "Operate as a domain-agnostic assistant. "
            "Do not assume any specific industry unless explicitly provided "
            "by the user request or retrieved context."
        ),
    },
    "gas_projects": {
        "name": "Gas Projects & Operations",
        "instructions": (
            "When relevant, prefer terminology and reasoning typical of gas "
            "infrastructure, operations, safety, compliance, and technical "
            "documentation. If evidence does not support this domain, stay neutral."
        ),
    },
    "contracts_compliance": {
        "name": "Contracts & Compliance",
        "instructions": (
            "When relevant, prioritize contract interpretation, obligations, "
            "risk exposure, compliance checks, and traceable evidence. "
            "Avoid legal assumptions not present in provided context."
        ),
    },
    "ai_expert": {
        "name": "AI Expert",
        "instructions": (
            "Adopt an AI-engineering expert perspective. Prioritize precise "
            "technical language, explicit assumptions, model/system limitations, "
            "evaluation criteria, and implementation tradeoffs. "
            "When possible, provide actionable and testable guidance."
        ),
    },
}
