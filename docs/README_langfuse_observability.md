# Langfuse Observability Setup

This guide explains all configuration required to enable observability with Langfuse for this project, including Docker deployment.

## Scope

Applies to:

1. Local runtime (Streamlit / API launched from project root)
2. Docker deployment (`deployment/docker`)

## 1) Required Project Settings

Langfuse integration is enabled through:

1. Public runtime config in `config.py`
2. Private secrets in `config_private.py`
3. Python dependency `langfuse`

### 1.1 Public Config (`config.py`)

File:
`/Users/lsaetta/Progetti/advanced_rag_agent/config.py`

Required keys:

1. `ENABLE_TRACING = True`
2. `LANGFUSE_HOST = "https://<your-langfuse-host>"`

Current placeholder host in repo:
`https://langfuse.example.internal`

### 1.2 Private Config (`config_private.py`)

File:
`/Users/lsaetta/Progetti/advanced_rag_agent/config_private.py`

Required keys:

1. `LANGFUSE_PUBLIC_KEY = "pk-lf-..."`
2. `LANGFUSE_SECRET_KEY = "sk-lf-..."`

Template source:
`config_private_template.py`

Important:
`config_private.py` is gitignored and must be created/maintained per environment.

### 1.3 Dependencies

Langfuse SDK is required in runtime dependencies:

1. `requirements.txt` includes `langfuse>=3.0.0`
2. `deployment/docker/requirements.runtime.txt` includes `langfuse>=3.0.0`

Install locally:

```bash
pip install -r requirements.txt
```

## 2) How Tracing Is Wired

Integration module:
`core/observability.py`

Notes:

1. Existing `zipkin_span(...)` decorators are preserved through a compatibility adapter.
2. If Langfuse config is missing or tracing is disabled, observability becomes no-op (does not break workflow).
3. Metadata annotations are emitted from UI orchestration and core agent nodes (intent, retrieval counts, rerank stats, errors, workflow status).

## 3) Local Runtime Checklist

1. Ensure `config.py` has `ENABLE_TRACING=True`
2. Set real `LANGFUSE_HOST` in `config.py`
3. Ensure `config_private.py` contains real `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`
4. Install dependencies (`pip install -r requirements.txt`)
5. Start app as usual and run one query
6. Verify trace/observations in your Langfuse project

## 4) Docker Deployment Configuration

## 4.1 Single-source env file

Use:
`deployment/docker/setup.env` (or start from `setup.env.example`)

Required entries:

1. `CFG_ENABLE_TRACING=True`
2. `CFG_LANGFUSE_HOST=https://<your-langfuse-host>`
3. `PRIV_LANGFUSE_PUBLIC_KEY=pk-lf-...`
4. `PRIV_LANGFUSE_SECRET_KEY=sk-lf-...`

Example template is already present in:
`deployment/docker/setup.env.example`

## 4.2 Apply setup values to config files

Run from project root:

```bash
python3 deployment/docker/configure.py --env-file deployment/docker/setup.env --write
```

What this updates:

1. `config.py` (`LANGFUSE_HOST`, `ENABLE_TRACING`, other CFG_* keys)
2. `config_private.py` (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, other PRIV_* keys)
3. `deployment/docker/docker-compose.yml` host path mounts if configured

## 4.3 Compose behavior

Current Docker setup mounts `config_private.py` into container at:
`/app/config_private.py`

This means Langfuse keys are provided through mounted config file. No additional dedicated compose env vars for Langfuse are required.

## 4.4 Build/Run

From `deployment/docker`:

```bash
docker compose build
docker compose up -d
```

Then verify traces in Langfuse after at least one user request.

## 5) Verification and Troubleshooting

If no traces appear:

1. Confirm `ENABLE_TRACING=True` in the active `config.py`
2. Confirm `LANGFUSE_HOST` points to reachable Langfuse instance URL
3. Confirm `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set in active `config_private.py`
4. Confirm `langfuse` package is installed in runtime environment
5. Check app logs for startup/runtime warnings

If running in Docker:

1. Confirm the expected `config_private.py` is the one mounted in compose
2. Re-run `deployment/docker/configure.py --write` after editing `setup.env`
3. Recreate containers after config changes

