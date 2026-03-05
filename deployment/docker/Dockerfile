FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime deps for PDF/doc loading and HTTPS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install only the runtime dependencies required by the app.
COPY deployment/docker/requirements.runtime.txt /tmp/requirements.runtime.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/requirements.runtime.txt

# Copy application code
COPY agent /app/agent
COPY core /app/core
COPY ui /app/ui
COPY pages /app/pages
COPY assistant_ui.py /app/assistant_ui.py
COPY config.py /app/config.py
COPY prompt_profiles.py /app/prompt_profiles.py
COPY config_private_template.py /app/config_private_template.py

EXPOSE 8501

CMD ["streamlit", "run", "assistant_ui.py", "--server.address=0.0.0.0", "--server.port=8501"]
