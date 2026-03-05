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

REFORMULATE_PROMPT_TEMPLATE = """
You're an AI assistant. Given a user request and a chat history reformulate the user question 
in a standalone question using the chat history.

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
You are an intelligent ranking assistant. Your task is to rank and filter text chunks 
based on their relevance to a given user query. You will receive:

1. A user query.
2. A list of text chunks.

Your goal is to:
- Rank the text chunks in order of relevance to the user query.
- Remove any text chunks that are completely irrelevant to the query.

### Instructions:
- Assign a **relevance score** to each chunk based on how well it answers or relates to the query.
- Return only the **top-ranked** chunks, filtering out those that are completely irrelevant.
- The output should be a **sorted list** of relevant chunks, from most to least relevant.
- Return only the JSON, don't add other text.
- Don't return the text of the chunk, only the index and the score.

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

Ensure that only relevant chunks are included in the output. If no chunk is relevant, return an empty list.

"""

INTENT_CLASSIFIER_TEMPLATE = """
You are an intent classifier for a RAG agent.

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


ADVANCED_ANALYSIS_PLANNER_TEMPLATE = """
You are an expert planner for advanced document analysis.

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
