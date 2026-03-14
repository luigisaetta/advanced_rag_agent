# Docker Deployment on Ubuntu

This guide explains how to run this project on Ubuntu with Docker Compose, including:
1. Streamlit app
2. Citation image server
3. BM25 MCP server
4. Nginx reverse proxy with Basic Auth

Current deployment behavior:
1. Nginx Basic Auth username is forwarded to Streamlit as `X-Forwarded-User`.
2. The app resolves user profile from Oracle table `USER_PROFILE` (`ADMIN` or `USER`).
3. Streamlit pages `loader_ui` and `post_answer_eval_ui` are visible only to `ADMIN`.

## 1) Prerequisites

Install Docker and Compose plugin:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Optional: run Docker without `sudo`:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## 2) Clone project and single-file setup

```bash
git clone <YOUR_REPO_URL> advanced_rag_agent
cd advanced_rag_agent
```

Create a single setup file:

```bash
cp deployment/docker/setup.env.example deployment/docker/setup.env
```

Edit only:

```text
deployment/docker/setup.env
```

Then run the configurator in dry-run mode:

```bash
python3 deployment/docker/configure.py --env-file deployment/docker/setup.env
```

Apply changes:

```bash
python3 deployment/docker/configure.py --env-file deployment/docker/setup.env --write
```

The script updates:
1. `config.py`
2. `config_private.py`
3. `deployment/docker/docker-compose.yml`

Useful options:

```bash
# show all options
python3 deployment/docker/configure.py --help

# update only specific targets
python3 deployment/docker/configure.py --env-file deployment/docker/setup.env --targets config private --write

# skip path existence checks
python3 deployment/docker/configure.py --env-file deployment/docker/setup.env --write --no-validate-paths

# write without backups
python3 deployment/docker/configure.py --env-file deployment/docker/setup.env --write --no-backup
```

## 3) Verify OCI configuration on Ubuntu host

Ensure OCI files are available on host:

```text
${HOME}/.oci/config
${HOME}/.oci/<private_key_file>
```

In `${HOME}/.oci/config`, verify:
1. profile exists (`[DEFAULT]` unless you changed compose)
2. `key_file` points to a file inside `${HOME}/.oci`
3. tenancy, user, fingerprint, region are valid

## 4) Prepare citation directory structure

Expected structure:

```text
<citation_root>/<document_name_without_pdf>/page0001.png
```

Example:

```text
/home/ubuntu/work-iren/pages/MyDoc/page0007.png
```

## 5) Configure Basic Auth users (Nginx)

Create password file on host:

```bash
mkdir -p deployment/docker/nginx
```

Option A (native Ubuntu tool):

```bash
sudo apt-get update && sudo apt-get install -y apache2-utils
htpasswd -Bbc deployment/docker/nginx/.htpasswd user1
htpasswd -Bb deployment/docker/nginx/.htpasswd user2
```

Option B (no host package, Docker-only):

```bash
docker run --rm --entrypoint htpasswd httpd:2.4-alpine -Bbn user1 'password1' > deployment/docker/nginx/.htpasswd
docker run --rm --entrypoint htpasswd httpd:2.4-alpine -Bbn user2 'password2' >> deployment/docker/nginx/.htpasswd
```

## 6) Build and start

From project root:

```bash
docker compose -f deployment/docker/docker-compose.yml up -d --build
```

## 7) Bootstrap user profiles in Oracle

Before validating role-based access in UI, create/seed `USER_PROFILE`:

```bash
# example with SQL*Plus (replace credentials/DSN)
sqlplus <DB_USER>/<DB_PASSWORD>@<DB_DSN> @deployment/sql/001_user_profile.sql
```

The script:
1. Creates table `USER_PROFILE` if missing.
2. Seeds user `luigi` as `ADMIN`.

If your Basic Auth username is different, add/update it in table:

```sql
MERGE INTO USER_PROFILE t
USING (SELECT 'your_username' username, 'USER' profile_code, 1 enabled FROM dual) s
ON (UPPER(t.username) = UPPER(s.username))
WHEN MATCHED THEN
  UPDATE SET t.profile_code=s.profile_code, t.enabled=s.enabled, t.updated_at=SYSTIMESTAMP
WHEN NOT MATCHED THEN
  INSERT (username, profile_code, enabled, updated_at)
  VALUES (s.username, s.profile_code, s.enabled, SYSTIMESTAMP);
COMMIT;
```

Check status:

```bash
docker compose -f deployment/docker/docker-compose.yml ps
```

Follow logs:

```bash
docker compose -f deployment/docker/docker-compose.yml logs -f
```

MCP-only local startup test:

```bash
docker compose -f deployment/docker/docker-compose.yml up -d --build bm25_mcp_server
docker compose -f deployment/docker/docker-compose.yml ps bm25_mcp_server
docker compose -f deployment/docker/docker-compose.yml logs -f bm25_mcp_server
```

Expected log lines include `BM25 MCP startup` and prewarm status.

## 8) Network and firewall

By default:
1. UI is exposed through Nginx on `8501/tcp`
2. BM25 MCP server is exposed on `8010/tcp` (if kept enabled in compose)
3. Citation server is internal-only and is proxied by Nginx under `/citations/`

If using firewall:

```bash
sudo iptables -I INPUT -p tcp -s 0.0.0.0/0 --dport 8501 -j ACCEPT
sudo iptables -I INPUT -p tcp -s 0.0.0.0/0 --dport 8010 -j ACCEPT
sudo service netfilter-persistent save
```

If you want MCP server private, remove its `ports:` mapping from compose and keep only internal access.

## 9) Validation checklist

1. Open `http://<UBUNTU_HOST_IP>:8501`
2. Nginx prompts for username/password
3. After login, Streamlit UI loads
4. In app logs you see: `Authenticated user=<username> profile=<ADMIN|USER> thread_id=<...>`
5. `loader_ui` and `post_answer_eval_ui` are visible only for `ADMIN`
6. Citation image links open correctly
7. `docker compose ... ps` shows all services as `Up`

## 10) Common issues

1. `.htpasswd is a directory`
   - fix:
   ```bash
   rm -rf deployment/docker/nginx/.htpasswd
   ```
   then recreate as a file
2. OCI auth failures
   - check `${HOME}/.oci/config` and key path mapping
3. Wallet/DB connection errors
   - verify wallet mount path in compose and files in wallet directory
4. 502 from Nginx
   - check app logs:
   ```bash
   docker compose -f deployment/docker/docker-compose.yml logs -f nginx_streamlit custom-rag-agent-ui
   ```
5. User authenticated but role-based pages not visible as expected
   - verify `USER_PROFILE` content for the same Basic Auth username
   - verify `X-Forwarded-User` forwarding is present in `deployment/docker/nginx/streamlit.conf`

## 12) Stop and cleanup

Stop stack:

```bash
docker compose -f deployment/docker/docker-compose.yml down
```

Stop and remove images too:

```bash
docker compose -f deployment/docker/docker-compose.yml down --rmi local
```
