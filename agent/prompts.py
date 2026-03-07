"""
File name: prompts.py
Author: Luigi Saetta
Last modified: 24-02-2026
Python Version: 3.11

Description:
    All the prompts are defined here.

Usage:
    Import this module into other scripts to use its functions.
    Example:
      from agent.prompts import ...


License:
    This code is released under the MIT License.

Notes:
    This is a part of a demo showing how to implement an advanced
    RAG solution as a LangGraph agent.

Warnings:
    This module is in development, may change in future versions.
"""

from config import PROMPT_PROFILE, PROMPT_PROFILES

_PROFILE_PLACEHOLDER = "<<DOMAIN_PROFILE>>"


def resolve_prompt_profile_name(config=None, explicit_profile: str = None) -> str:
    """
    Resolve the active prompt profile from runtime config or fallback defaults.
    """
    if explicit_profile:
        profile = str(explicit_profile).strip().lower()
    else:
        configurable = (config or {}).get("configurable", {})
        profile = (
            str(configurable.get("prompt_profile", PROMPT_PROFILE)).strip().lower()
        )

    if profile in PROMPT_PROFILES:
        return profile
    return "general"


def build_domain_profile_block(profile_name: str) -> str:
    """
    Build a compact domain-profile section to inject in prompts.
    """
    profile = PROMPT_PROFILES.get(profile_name, PROMPT_PROFILES["general"])
    return (
        "Domain profile:\n"
        f"- Active profile: {profile_name}\n"
        f"- Profile label: {profile['name']}\n"
        f"- Guidance: {profile['instructions']}\n"
    )


def apply_prompt_profile(
    template: str, config=None, explicit_profile: str = None
) -> str:
    """
    Inject domain-profile guidance into a prompt template.
    """
    profile_name = resolve_prompt_profile_name(config, explicit_profile)
    block = build_domain_profile_block(profile_name)
    if _PROFILE_PLACEHOLDER in template:
        return template.replace(_PROFILE_PLACEHOLDER, block)
    return f"{block}\n{template}"


REFORMULATE_PROMPT_TEMPLATE = """
You're an AI assistant. Given a user request and a chat history reformulate the user question 
in a standalone question using the chat history.

<<DOMAIN_PROFILE>>

Constraints:
- return only the standalone question, do not add any other text or comment.

User request: {user_request}
Chat history: {chat_history}

Standalone question: 
"""

ANSWER_PROMPT_TEMPLATE = """
You're a helpful AI assistant. Your task is to answer user questions using only the information
provided in the context and the history of previous messages.
Respond in a friendly and polite tone at all times.

<<DOMAIN_PROFILE>>

## Constraints:
- Answer based only on the provided context.
- If the context is partial, provide a best-effort answer grounded in available evidence,
  and clearly state what information is missing or uncertain.
- Do not invent facts that are not supported by the context.
- Always answer in the same language as the user's request.
- Always return your response in properly formatted markdown.

Context: {context}

"""

RERANKER_TEMPLATE = """
You are an intelligent ranking assistant. Rank chunks by relevance to the query.

<<DOMAIN_PROFILE>>

Critical objective:
- Preserve useful evidence coverage. Prefer false positives over false negatives.
- Do NOT over-filter.

Rules:
1) Include every chunk that is directly relevant OR potentially supportive/contextual.
2) Exclude only chunks that are clearly unrelated.
3) If at least one chunk is relevant, return at least 4 chunks (or all chunks if fewer than 4).
4) If relevance is uncertain, keep the chunk with a lower score rather than dropping it.
5) Keep ranking by usefulness for answering the query, not only by keyword overlap.
6) Return JSON only, with index and score.

Scoring guide (0.0-1.0):
- 0.85-1.00: directly answers key parts
- 0.60-0.84: strongly supportive evidence
- 0.35-0.59: partially relevant but potentially useful
- 0.00-0.34: clearly unrelated (exclude unless uncertainty exists)

### Input Format:
User Query:
{query}

Text Chunks (list indexed from 0):
{chunks}

### **Output Format:**
Return a **JSON object** with the following format:
```json
{{
  "ranked_chunks": [
    {{"index": 0, "score": X.X}},
    {{"index": 2, "score": Y.Y}},
    ...
  ]
}}
```
Where:
- "index" is the original position of the chunk in the input list. Index starts from 0.
- "score" is the relevance score (higher is better).

If all chunks are clearly unrelated, return an empty list.

"""

INTENT_CLASSIFIER_TEMPLATE = """
You are an intent classifier for a RAG agent.

<<DOMAIN_PROFILE>>

Classify the user request into exactly one intent:
- GLOBAL_KB: answer should come from the global knowledge base.
- SESSION_DOC: answer should come only from the uploaded in-session PDF.
- HYBRID: answer needs both global knowledge base and session PDF.

Decision rules:
- If the user explicitly asks about "the uploaded PDF", "this document", "this file", or asks to summarize/extract/quote from it, prefer SESSION_DOC.
- If the user asks a generic/domain question not tied to the uploaded file, prefer GLOBAL_KB.
- If the user asks to combine uploaded document evidence with broader/company/general knowledge, use HYBRID.
- Choose exactly one label.

Few-shot examples:
1) User: "Summarize the uploaded PDF in 5 bullet points."
Intent: SESSION_DOC

2) User: "Define class A1"
Intent: GLOBAL_KB

3) User: "Based on the uploaded contract, and also on our policy/regulations, can we terminate early?"
Intent: HYBRID

4) User: "In this file, what does section 4.2 say about penalties?"
Intent: SESSION_DOC

5) User: "Analyze the document and identify areas that should be improved based on existing regulations."
Intent: HYBRID

Return only valid JSON with this shape:
{{
  "intent": "GLOBAL_KB|SESSION_DOC|HYBRID"
}}

Do not add explanations or any extra text.

User request:
{user_request}
"""

HYBRID_KB_QUERY_TEMPLATE = """
You are helping a RAG system build a knowledge-base search query.

<<DOMAIN_PROFILE>>

Task:
- Rewrite the standalone question into a KB-focused query.
- Use the uploaded document excerpts to inject specific entities, constraints, and facts.
- Keep the query compact and information-dense.

Rules:
- Return only the final query string.
- Do not add explanations, labels, or markdown.
- If excerpts are not useful, return the standalone question unchanged.

User request:
{user_request}

Standalone question:
{standalone_question}

Uploaded document excerpts:
{session_snippets}
"""

POST_ANSWER_EVALUATION_TEMPLATE = """
You are evaluating a RAG answer quality using a strict decision tree.

<<DOMAIN_PROFILE>>

Goal:
- Identify the most likely primary root cause among:
  - NO_ISSUE
  - RETRIEVAL
  - RERANK
  - GENERATION

Decision tree:
1) If retrieved and reranked evidence are relevant and the final answer is grounded/correct, choose NO_ISSUE.
2) Else, if retrieved evidence is missing or mostly off-topic for the user request, choose RETRIEVAL.
3) Else, if retrieval contains relevant evidence but reranker context drops or under-prioritizes key evidence, choose RERANK.
4) Else, choose GENERATION only if there is clear, direct evidence that reranker context is sufficient but the final answer is incorrect/incomplete/not grounded.
5) If evidence is ambiguous or insufficient to prove RETRIEVAL, RERANK, or GENERATION, choose NO_ISSUE.

Important evidence handling rules:
- Use source inventories as authoritative metadata of which sources/pages were retrieved.
- Do NOT claim that a source is missing if it appears in the corresponding inventory.
- Base the root-cause decision on topical relevance and evidence coverage, not on guesswork.
- Do NOT require verbatim sentence matches: paraphrased support counts as grounded evidence.
- Be conservative with blame assignment: default to NO_ISSUE unless strong evidence supports a specific error class.

Return ONLY valid JSON:
{{
  "root_cause": "NO_ISSUE|RETRIEVAL|RERANK|GENERATION",
  "reason": "short evidence-based reason",
  "confidence": 0.0
}}

Confidence rules:
- Return a numeric value between 0.0 and 1.0.
- Use lower confidence when evidence is partial/ambiguous.
- Use higher confidence only when evidence strongly supports the selected root cause.
- `reason` must be written in English only.

User request:
{user_request}

Standalone question:
{standalone_question}

Retrieved source inventory:
{retriever_sources}

Reranker source inventory:
{reranker_sources}

Retrieved docs (before reranker):
{retriever_context}

Docs used for final answer (after reranker):
{reranker_context}

Final answer:
{final_answer}
"""


ADVANCED_ANALYSIS_PLANNER_TEMPLATE = """
You are an expert planner for advanced document analysis.

<<DOMAIN_PROFILE>>

You receive:
- a user request
- all chunks extracted from an uploaded PDF in session
- a maximum number of actions

Your task:
- build a concise execution plan as ordered actions.
- each action must focus on one section/part of the document.
- for each action decide whether a KB search is required to complement that section.
- if KB search is needed, provide a compact KB query.

Output rules:
- return ONLY valid JSON, no extra text.
- use this schema exactly:
{{
  "plan": [
    {{
      "step": 1,
      "section": "string",
      "chunk_numbers": [1, 2],
      "objective": "string",
      "kb_search_needed": true,
      "kb_query": "string"
    }}
  ]
}}
- number of actions must be between 1 and {max_actions}.
- each action must include at least one chunk number in `chunk_numbers`.
- chunk numbers must refer to the indexed chunks provided in input.
- if kb_search_needed is false, kb_query must be an empty string.

User request:
{user_request}

Session PDF chunks (full set):
{session_chunks}
"""


ADVANCED_ANALYSIS_STEP_TEMPLATE = """
You are executing one step of an advanced analysis plan.

<<DOMAIN_PROFILE>>

Write a concise, evidence-based result for this step using:
- selected chunks from the uploaded PDF
- optional knowledge-base excerpts

Constraints:
- answer in the same language as the user request.
- use only the provided evidence.
- keep the output concise (maximum {max_words} words).
- include practical conclusions for this step.
- if evidence is insufficient, say what is missing.

User request:
{user_request}

Plan step:
- Step: {step}
- Section: {section}
- Objective: {objective}
- KB search needed: {kb_search_needed}
- KB query: {kb_query}

PDF evidence (selected chunks):
{pdf_context}

KB evidence (retrieved docs):
{kb_context}
"""


ADVANCED_ANALYSIS_SYNTHESIS_TEMPLATE = """
You are a senior analyst writing the final synthesis of a multi-step analysis.

<<DOMAIN_PROFILE>>

Given:
- the original user request
- the detailed outputs of each executed step

Write a concise final synthesis section that:
1. integrates the key findings,
2. highlights cross-step consistency or conflicts,
3. states clear final conclusions,
4. lists any critical missing evidence.

Constraints:
- keep the synthesis under {max_words} words.
- use the same language as the user request.
- do not invent facts outside step outputs.
- do not add headings/titles; return only synthesis body text.

User request:
{user_request}

Step outputs:
{step_outputs}
"""


ADVANCED_ANALYSIS_RISK_CHECK_TEMPLATE = """
You are a risk screening assistant for a document analysis report.

<<DOMAIN_PROFILE>>

Given:
- the user request
- the current final analysis text

Task:
- detect whether the report contains critical negative findings that should be validated.
- if yes, extract concise claims to validate.

Critical negative findings include material risks such as:
- severe non-compliance,
- termination blockers,
- major penalties/liabilities,
- safety/security/legal blockers.

Output rules:
- return ONLY valid JSON.
- use this exact schema:
{{
  "critical_negative_findings": true,
  "claims_to_validate": ["claim 1", "claim 2"]
}}

If no critical negative findings are present, return:
{{
  "critical_negative_findings": false,
  "claims_to_validate": []
}}

User request:
{user_request}

Final analysis:
{final_answer}
"""


ADVANCED_ANALYSIS_RISK_VALIDATION_TEMPLATE = """
You are validating critical negative findings using provided KB evidence.

<<DOMAIN_PROFILE>>

Given:
- the user request
- claims that need validation
- KB evidence snippets

Task:
- assess whether each claim is supported, partially supported, or not supported.
- keep the answer concise and evidence-based.
- if evidence is insufficient, state that explicitly.

Constraints:
- use only provided KB evidence.
- do not add headings/titles.
- answer in the same language as the user request.

User request:
{user_request}

Claims to validate:
{claims}

KB evidence:
{kb_context}
"""
