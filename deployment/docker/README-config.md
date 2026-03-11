# Docker Configuration Guide

This document is dedicated to the configuration required before running the Docker deployment.

Project root used in this guide:
`/home/ubuntu/custom_rag_agent`

This is an example project root and can be different on your machine. If different, replace this prefix in all paths below.
You can create this project root by cloning the repository:
`git clone https://github.com/luigisaetta/custom_rag_agent /home/ubuntu/custom_rag_agent`

## Constraints and Assumptions

1. Oracle database is an Autonomous Database (ADB).
2. The schema owner has already been created in the database.
3. The ADB wallet has been downloaded and is available on the host.
4. The guide assumes Linux paths/commands; it can also be used on macOS with equivalent host paths and tools.

## Scope

Deployment components in `deployment/docker`:
1. `custom-rag-agent-ui` (Streamlit app + Langgraph agent)
2. `citation_image_server` (static image server)
3. `bm25_mcp_server` (FastMCP server for BM25 cache/search)
4. `nginx_streamlit` (reverse proxy + Basic Auth)

Authentication/Profile notes:
1. Nginx Basic Auth username is forwarded to Streamlit via `X-Forwarded-User`.
2. App profile resolution uses Oracle table `USER_PROFILE` (`ADMIN` / `USER`).
3. Admin-only pages are `loader_ui` and `post_answer_eval_ui`.

## Configuration Files Checklist

### 0) `config.py` (project root, public runtime settings)

Path:
`/home/ubuntu/custom_rag_agent/config.py`

Must verify:
1. OCI regions/endpoints are set as intended:
   - `LLM_REGION` / `LLM_SERVICE_ENDPOINT`
   - `EMBED_REGION` / `EMBED_SERVICE_ENDPOINT`
2. Embedding model settings are coherent with the embedding region:
   - `EMBED_MODEL_TYPE`
   - `EMBED_MODEL_ID`

### 1) `config_private.py` (project root)

Path:
`/home/ubuntu/custom_rag_agent/config_private.py`

Purpose:
Stores private database and APM credentials used by the application.

How to prepare:
1. Create it from template if missing:
   `cp config_private_template.py config_private.py`
2. Configure these fields to connect to Oracle DB:
   - `VECTOR_DB_USER`
   - `VECTOR_DB_PWD`
   - `VECTOR_WALLET_PWD`
   - `VECTOR_DSN`
   - `VECTOR_WALLET_DIR` (host path if using direct path logic)
   - `APM_PUBLIC_KEY` (if you want to enable APM tracing)

Notes:
1. In Docker Compose, this file is mounted read-only into container at `/app/config_private.py`.
2. The wallet path used by `CONNECT_ARGS` must be consistent with the mounted wallet directory.

### 2) `deployment/docker/docker-compose.yml`

Path:
`/home/ubuntu/custom_rag_agent/deployment/docker/docker-compose.yml`

Purpose:
Defines services, mounted files, exposed ports, and runtime environment variables.

Must review and configure:
1. Citation server mount:
   - `/home/ubuntu/pages:/data/citations:ro`
   - Replace left side with your host citation images directory.
2. Wallet mount:
   - `/home/ubuntu/wallet:/app/wallet_atp:ro`
   - Replace left side with your host Oracle wallet directory.
3. Private config mount:
   - `../../config_private.py:/app/config_private.py:ro`
   - Ensure the source file exists.
4. OCI config mount:
   - `${HOME}/.oci:/root/.oci:ro`
   - Ensure `${HOME}/.oci` exists and contains valid OCI files.
5. Environment variables:
   - `VECTOR_WALLET_DIR=/app/wallet_atp`
   - `OCI_CONFIG_FILE=/root/.oci/config`
   - `OCI_CLI_PROFILE=DEFAULT` (or your desired profile)
   - `CITATION_BASE_URL=${CITATION_BASE_URL:-http://127.0.0.1:8008/}`
6. Port bindings:
   - Nginx UI: `"8501:80"` (host `8501`)
   - Citation server: `"${CITATION_SERVER_BIND_IP:-0.0.0.0}:${CITATION_SERVER_PORT:-8008}:8008"`
   - BM25 MCP server: `"${BM25_MCP_BIND_IP:-0.0.0.0}:${BM25_MCP_PORT:-8010}:8010"`
7. BM25 MCP environment values:
   - `BM25_CACHE_DIR` (container path for persisted cache, e.g. `/app/bm25_cache`)
   - `BM25_PREWARM_ENABLED` (default `true`)
   - `BM25_PREWARM_COLLECTIONS` (CSV, default from `COLLECTION_LIST`)
   - `BM25_TEXT_COLUMN` (default `TEXT`)
   - `BM25_BATCH_SIZE` (default `40`)
8. BM25 cache mounts:
   - `../../bm25_cache/ui:/app/bm25_cache` for `custom-rag-agent-ui`
   - `../../bm25_cache/mcp:/app/bm25_cache` for `bm25_mcp_server`
9. BM25 MCP build artifacts:
   - `deployment/docker/Dockerfile.mcp`
   - `deployment/docker/requirements.mcp.txt`

Optional host env variables:
1. `CITATION_BASE_URL`
2. `CITATION_SERVER_BIND_IP`
3. `CITATION_SERVER_PORT`
4. `BM25_MCP_BIND_IP`
5. `BM25_MCP_PORT`
6. `BM25_PREWARM_ENABLED`
7. `BM25_PREWARM_COLLECTIONS`
8. `BM25_TEXT_COLUMN`
9. `BM25_BATCH_SIZE`
10. `BM25_CACHE_DIR`

### 3) OCI config files on host (`${HOME}/.oci`)

Required files:
1. `${HOME}/.oci/config`
2. Private key file referenced by `key_file` in OCI config

Must verify:
1. Profile exists (`DEFAULT` unless `OCI_CLI_PROFILE` changed)
2. `key_file` points to a path available inside mounted `/root/.oci/...`
3. `tenancy`, `user`, `fingerprint`, `region` are valid

### 4) `deployment/docker/nginx/.htpasswd`

Path:
`/home/ubuntu/custom_rag_agent/deployment/docker/nginx/.htpasswd`

Purpose:
Basic Auth users/passwords for Nginx.

Must do:
1. Create file before starting stack.
2. Add at least one user.

Example using Docker:
```bash
mkdir -p deployment/docker/nginx
docker run --rm --entrypoint htpasswd httpd:2.4-alpine -Bbn user1 'password1' > deployment/docker/nginx/.htpasswd
```

### 5) `deployment/docker/nginx/streamlit.conf`

Path:
`/home/ubuntu/custom_rag_agent/deployment/docker/nginx/streamlit.conf`

Purpose:
Nginx reverse proxy rules for Streamlit and Basic Auth enforcement.

What to configure (only if needed):
1. `server_name` (default `_`)
2. Auth settings:
   - `auth_basic`
   - `auth_basic_user_file /etc/nginx/.htpasswd`
3. Upstream target:
   - `proxy_pass http://custom-rag-agent-ui:8501;`
4. Forwarded authenticated user header:
   - `proxy_set_header X-Forwarded-User $remote_user;`

### 6) `deployment/sql/001_user_profile.sql`

Path:
`/home/ubuntu/custom_rag_agent/deployment/sql/001_user_profile.sql`

Purpose:
Bootstrap `USER_PROFILE` table and initial role mapping.

Must do:
1. Execute this script before validating role-based page visibility.
2. Ensure each Basic Auth username has a row in `USER_PROFILE`.
3. Seed includes user `luigi` with profile `ADMIN`.

## Runtime Configuration Mapping

Container runtime paths expected by app:
1. `/app/config_private.py` from host `config_private.py`
2. `/app/wallet_atp` from host wallet directory
3. `/root/.oci` from host `${HOME}/.oci`
4. `/data/citations` from host citation pages directory
5. `/app/bm25_cache` from host `../../bm25_cache/ui` (`custom-rag-agent-ui`)
6. `/app/bm25_cache` from host `../../bm25_cache/mcp` (`bm25_mcp_server`)
7. BM25 MCP server uses the same `/app/config_private.py`, `/app/wallet_atp`, and `/root/.oci` mounts

## Pre-Start Validation

Run these checks from project root (`/home/ubuntu/custom_rag_agent` in this guide):
```bash
test -f config_private.py && echo "OK config_private.py"
test -d "$HOME/.oci" && echo "OK .oci dir"
test -f "$HOME/.oci/config" && echo "OK oci config"
test -f deployment/docker/nginx/.htpasswd && echo "OK .htpasswd"
test -f deployment/docker/Dockerfile.mcp && echo "OK Dockerfile.mcp"
test -f deployment/docker/requirements.mcp.txt && echo "OK requirements.mcp.txt"
test -f deployment/sql/001_user_profile.sql && echo "OK user profile SQL"
test -d bm25_cache/ui && echo "OK bm25_cache/ui"
test -d bm25_cache/mcp && echo "OK bm25_cache/mcp"
```

Then verify host mounts in compose:
```bash
# Linux example paths
rg -n "/home/ubuntu|wallet_atp|/data/citations|config_private.py|\\.oci|bm25_cache" deployment/docker/docker-compose.yml

# macOS example paths
rg -n "/Users/lsaetta|wallet_atp|/data/citations|config_private.py|\\.oci|bm25_cache" deployment/docker/docker-compose.yml

# Fallback (works without ripgrep on Linux and macOS)
grep -nE "/home/ubuntu|/Users/|wallet_atp|/data/citations|config_private.py|\\.oci|bm25_cache" deployment/docker/docker-compose.yml
```

If `rg` is not installed:
1. Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ripgrep`
2. macOS (Homebrew): `brew install ripgrep`

## Start Commands

```bash
docker compose -f deployment/docker/docker-compose.yml up -d --build
docker compose -f deployment/docker/docker-compose.yml ps
docker compose -f deployment/docker/docker-compose.yml logs -f
```

MCP-only local startup test:
```bash
docker compose -f deployment/docker/docker-compose.yml up -d --build bm25_mcp_server
docker compose -f deployment/docker/docker-compose.yml ps bm25_mcp_server
docker compose -f deployment/docker/docker-compose.yml logs -f bm25_mcp_server
```

## Typical Misconfigurations

1. Wrong host wallet path
   - Symptom: DB connection/wallet errors.
2. Missing or invalid OCI key path
   - Symptom: OCI auth errors.
3. Missing `.htpasswd`
   - Symptom: Nginx container fails to start.
4. Wrong citation image root
   - Symptom: citation links return 404.
5. Wrong `CITATION_BASE_URL`
   - Symptom: generated citation links do not resolve from browser.
6. Missing BM25 cache directories (`bm25_cache/ui`, `bm25_cache/mcp`)
   - Symptom: cache persistence is not retained as expected.
7. Missing/incorrect `USER_PROFILE` mapping
   - Symptom: authenticated user gets default `USER` profile and cannot access admin-only pages.
