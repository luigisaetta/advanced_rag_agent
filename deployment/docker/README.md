# Docker deployment (UI + Citation Image Server + BM25 MCP + Nginx)

For Ubuntu-specific setup, see [README-ubuntu.md](./README-ubuntu.md).
For a complete configuration checklist (files and required keys/paths), see [README-config.md](./README-config.md).
Recommended deployment method: Docker Compose (`deployment/docker/docker-compose.yml`).

This setup runs:
1. Streamlit UI (`assistant_ui.py`)
2. A Python static web server for citation images
3. BM25 MCP server (`fastmcp`) with startup prewarm cache
4. Nginx reverse proxy in front of Streamlit

## Deployment settings

`deployment/docker/docker-compose.yml` configures:

1. `config_private.py` mounted to `/app/config_private.py`
2. Oracle wallet mounted to `/app/wallet_atp`
3. OCI config mounted to `/root/.oci`
4. Citation image root mounted from `/Users/lsaetta/Progetti/work-iren/pages` to `/data/citations`
5. BM25 MCP service exposed on `${BM25_MCP_PORT:-8010}`
6. Persistent BM25 cache host folders:
   - `../../bm25_cache/ui` for UI service
   - `../../bm25_cache/mcp` for MCP service

Current UI service settings:

```yaml
environment:
  - VECTOR_WALLET_DIR=/app/wallet_atp
  - OCI_CONFIG_FILE=/root/.oci/config
  - OCI_CLI_PROFILE=DEFAULT
  - CITATION_BASE_URL=${CITATION_BASE_URL:-http://127.0.0.1:8008/}
  - BM25_CACHE_DIR=/app/bm25_cache
expose:
  - "8501"
volumes:
  - ../../config_private.py:/app/config_private.py:ro
  - /Users/lsaetta/Progetti/work-iren/wallet:/app/wallet_atp:ro
  - ${HOME}/.oci:/root/.oci:ro
  - ../../bm25_cache/ui:/app/bm25_cache
```

Runtime note:
1. OCI endpoints are resolved by `config.py`.
2. Current public config supports separate regions for LLM and embeddings:
   - `LLM_REGION` / `LLM_SERVICE_ENDPOINT`
   - `EMBED_REGION` / `EMBED_SERVICE_ENDPOINT`

Citation image server settings:

```yaml
ports:
  - "${CITATION_SERVER_BIND_IP:-0.0.0.0}:${CITATION_SERVER_PORT:-8008}:8008"
volumes:
  - /Users/lsaetta/Progetti/work-iren/pages:/data/citations:ro
```

BM25 MCP server settings:

```yaml
ports:
  - "${BM25_MCP_BIND_IP:-0.0.0.0}:${BM25_MCP_PORT:-8010}:8010"
environment:
  - MCP_HOST=0.0.0.0
  - MCP_PORT=8010
  - BM25_CACHE_DIR=/app/bm25_cache
  - BM25_PREWARM_ENABLED=${BM25_PREWARM_ENABLED:-true}
  - BM25_PREWARM_COLLECTIONS=${BM25_PREWARM_COLLECTIONS:-}
  - BM25_TEXT_COLUMN=${BM25_TEXT_COLUMN:-TEXT}
  - BM25_BATCH_SIZE=${BM25_BATCH_SIZE:-40}
volumes:
  - ../../config_private.py:/app/config_private.py:ro
  - /Users/lsaetta/Progetti/work-iren/wallet:/app/wallet_atp:ro
  - ${HOME}/.oci:/root/.oci:ro
  - ../../bm25_cache/mcp:/app/bm25_cache
```

Nginx reverse proxy settings:

```yaml
ports:
  - "8501:80"
volumes:
  - ./nginx/streamlit.conf:/etc/nginx/conf.d/default.conf:ro
  - ./nginx/.htpasswd:/etc/nginx/.htpasswd:ro
```

## Basic Auth setup (Nginx)

Create the password file on the host at:

```text
deployment/docker/nginx/.htpasswd
```

Cross-platform (works on macOS and Linux, requires Docker):

```bash
mkdir -p deployment/docker/nginx
docker run --rm --entrypoint htpasswd httpd:2.4-alpine -Bbn user1 'password1' > deployment/docker/nginx/.htpasswd
docker run --rm --entrypoint htpasswd httpd:2.4-alpine -Bbn user2 'password2' >> deployment/docker/nginx/.htpasswd
```

Native commands (optional):

macOS (Homebrew):

```bash
brew install httpd
mkdir -p deployment/docker/nginx
htpasswd -Bbc deployment/docker/nginx/.htpasswd user1
htpasswd -Bb deployment/docker/nginx/.htpasswd user2
```

Linux (Debian/Ubuntu):

```bash
sudo apt-get update && sudo apt-get install -y apache2-utils
mkdir -p deployment/docker/nginx
htpasswd -Bbc deployment/docker/nginx/.htpasswd user1
htpasswd -Bb deployment/docker/nginx/.htpasswd user2
```

After creating/updating users:

```bash
docker compose -f deployment/docker/docker-compose.yml up -d
docker compose -f deployment/docker/docker-compose.yml restart nginx_streamlit
```

Important checks:

1. `config_private.py` exists in project root.
2. Wallet host path exists and contains `tnsnames.ora`, `sqlnet.ora`, wallet files.
3. `${HOME}/.oci/config` exists and the selected profile is valid.
4. In OCI config, `key_file` must resolve inside the mounted `/root/.oci/...` path in container.
5. `/Users/lsaetta/Progetti/work-iren/pages` contains citation subfolders and `pageNNNN.png` files.
6. `deployment/docker/nginx/.htpasswd` exists before starting Nginx.
7. BM25 MCP container startup logs show successful prewarm (or explicit prewarm errors).
8. `bm25_cache/ui` and `bm25_cache/mcp` directories exist in project root.

## Build

Run from the project root:

```bash
docker build -f deployment/docker/Dockerfile -t custom-rag-agent-ui:latest .
docker build -f deployment/docker/Dockerfile.mcp -t custom-rag-agent-mcp:latest .
```

## Local MCP startup test

Start only the MCP service:

```bash
docker compose -f deployment/docker/docker-compose.yml up -d --build bm25_mcp_server
docker compose -f deployment/docker/docker-compose.yml ps bm25_mcp_server
docker compose -f deployment/docker/docker-compose.yml logs -f bm25_mcp_server
```

Expected in logs:
1. `BM25 MCP startup`
2. Prewarm details (`warmed` collections and possible errors)

## Start (Recommended: Compose)

```bash
docker compose -f deployment/docker/docker-compose.yml up -d --build
```

UI URL (through Nginx):

```text
http://localhost:8501
```

## Stop

```bash
docker compose -f deployment/docker/docker-compose.yml down
```

## Logs

```bash
docker compose -f deployment/docker/docker-compose.yml logs -f
```

## Alternative: docker run (UI only, no Nginx)

```bash
docker run -d \
  --name custom-rag-agent-ui \
  -p 8501:8501 \
  -e VECTOR_WALLET_DIR=/app/wallet_atp \
  -e OCI_CONFIG_FILE=/root/.oci/config \
  -e OCI_CLI_PROFILE=DEFAULT \
  -v "$(pwd)/config_private.py:/app/config_private.py:ro" \
  -v /Users/lsaetta/Progetti/work-iren/wallet:/app/wallet_atp:ro \
  -v "${HOME}/.oci:/root/.oci:ro" \
  custom-rag-agent-ui:latest
```
