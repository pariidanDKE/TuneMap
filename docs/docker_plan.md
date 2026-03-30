# Docker Compose Plan — Apple Music Knowledge Graph

## Architecture

Three services — only the Streamlit app needs a custom Dockerfile.

```
docker-compose.yml
├── neo4j      → official image (neo4j:5)
├── vllm       → official image (vllm/vllm-openai:latest)  [GPU only]
└── app        → custom image built from ./Dockerfile
```

---

## Files to create

```
AppleMusciKG/
├── Dockerfile
├── docker-compose.yml          ← full stack (neo4j + vllm + app)
├── docker-compose.no-gpu.yml   ← lightweight (neo4j + app, OpenAI/Claude API)
├── .env.example
└── requirements.txt            ← if not already present
```

---

## Dockerfile (app only)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py query_engine.py ./
COPY data_processing/ ./data_processing/
COPY lib/ ./lib/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
```

---

## docker-compose.yml (full GPU stack)

```yaml
services:

  neo4j:
    image: neo4j:5
    restart: unless-stopped
    ports:
      - "7474:7474"   # browser UI
      - "7687:7687"   # bolt
    volumes:
      - neo4j_data:/data
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}
      - NEO4J_PLUGINS=["apoc"]
    healthcheck:
      test: ["CMD", "neo4j", "status"]
      interval: 10s
      timeout: 5s
      retries: 10

  vllm:
    image: vllm/vllm-openai:latest
    restart: unless-stopped
    runtime: nvidia                         # requires nvidia-container-toolkit on host
    ports:
      - "8000:8000"
    volumes:
      - model_cache:/root/.cache/huggingface
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    command: >
      --model QuantTrio/Qwen3.5-9B-AWQ
      --served-model-name qwen3.5-9b-awq
      --quantization awq_marlin
      --enable-auto-tool-choice
      --tool-call-parser qwen3_coder
      --reasoning-parser qwen3
      --gpu-memory-utilization 0.90
      --max-num-seqs 1
      --port 8000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 20
      start_period: 120s              # model load takes time

  app:
    build: .
    restart: unless-stopped
    ports:
      - "8501:8501"
    depends_on:
      neo4j:
        condition: service_healthy
      vllm:
        condition: service_healthy
    env_file: .env
    environment:
      - NEO4J_URI=bolt://neo4j:7687   # service name, not localhost
      - VLLM_BASE_URL=http://vllm:8000/v1

volumes:
  neo4j_data:
  model_cache:
```

---

## docker-compose.no-gpu.yml (OpenAI or Claude, no vLLM)

For users without a GPU, or for cloud deployment (Railway, Render, etc.).
Drop the vllm service entirely — set `LLM_PROVIDER` to `openai` or `claude`.

```yaml
services:

  neo4j:
    image: neo4j:5
    restart: unless-stopped
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}
      - NEO4J_PLUGINS=["apoc"]
    healthcheck:
      test: ["CMD", "neo4j", "status"]
      interval: 10s
      timeout: 5s
      retries: 10

  app:
    build: .
    restart: unless-stopped
    ports:
      - "8501:8501"
    depends_on:
      neo4j:
        condition: service_healthy
    env_file: .env
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - LLM_PROVIDER=${LLM_PROVIDER:-openai}   # openai | claude

volumes:
  neo4j_data:
```

Run with:
```bash
docker compose -f docker-compose.no-gpu.yml up
```

---

## .env.example

```bash
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme

# vLLM (local GPU stack only)
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=qwen3.5-9b-awq
HF_TOKEN=hf_...                      # HuggingFace token for model download

# LLM Provider (for no-gpu stack or UI switcher)
# Options: vllm | openai | claude
LLM_PROVIDER=vllm
LLM_MODEL=                           # leave blank to use provider default
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Gotchas & Notes

| Issue | Detail |
|---|---|
| **GPU host requirement** | Full stack needs NVIDIA GPU + `nvidia-container-toolkit` installed on the host machine. Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html |
| **vLLM startup time** | First run downloads ~5GB model from HuggingFace. Subsequent runs use the `model_cache` volume. The `start_period: 120s` healthcheck delay accounts for this. |
| **Neo4j data** | The `neo4j_data` volume holds your ingested Knowledge Graph. For others to use it they must run the ingestion pipeline (`ingest_graph.py`, `ingest_lyrics.py`) against their own `Library.xml`. |
| **Service networking** | Inside Docker Compose, services talk to each other by service name — `bolt://neo4j:7687`, `http://vllm:8000/v1`. The `environment` overrides in the `app` service handle this automatically. |
| **APOC plugin** | The query engine uses APOC for schema introspection. The `NEO4J_PLUGINS=["apoc"]` env var installs it automatically on the official neo4j image. |
| **Streamlit file watcher** | `--server.headless=true` disables the file watcher in the container (no browser to open). |

---

## Sharing with others — recommended flow

```
1. git clone <repo>
2. cp .env.example .env   →  fill in NEO4J_PASSWORD + API key
3. Drop Library.xml into Data/
4. python data_processing/parse_library.py
5. docker compose up neo4j          →  wait for healthy
6. python data_processing/ingest_graph.py
7. python data_processing/ingest_lyrics.py
8. docker compose up                →  full stack
9. open http://localhost:8501
```

Or without a GPU:
```
1–3. same as above (set LLM_PROVIDER=openai, add OPENAI_API_KEY)
4–7. same ingestion steps
8. docker compose -f docker-compose.no-gpu.yml up
```
