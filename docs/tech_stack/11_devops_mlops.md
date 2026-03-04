# 11. DevOps & MLOps: Docker, CI/CD, MLflow & LLM Monitoring

> Interview preparation guide for Backend AI Engineers.
> Covers containerization, pipelines, experiment tracking, and production monitoring.

---

## Table of Contents

1. [Docker](#1-docker)
2. [CI/CD](#2-cicd)
3. [MLOps Overview](#3-mlops-overview)
4. [MLflow](#4-mlflow)
5. [Model Versioning](#5-model-versioning)
6. [Monitoring LLM Applications](#6-monitoring-llm-applications)
7. [Infrastructure for AI](#7-infrastructure-for-ai)
8. [Data Pipeline Management](#8-data-pipeline-management)
9. [Security in MLOps](#9-security-in-mlops)
10. [Q&A Section](#10-qa-section)

---

## 1. Docker

### 1.1 What is Docker?

Docker is a platform for building, shipping, and running applications inside **containers** --
lightweight, isolated environments that share the host OS kernel. Unlike virtual machines,
containers do not bundle a full guest operating system, making them faster and more
resource-efficient.

```
VMs:                              Containers:
┌─────┐ ┌─────┐ ┌─────┐          ┌─────┐ ┌─────┐ ┌─────┐
│App A│ │App B│ │App C│          │App A│ │App B│ │App C│
├─────┤ ├─────┤ ├─────┤          ├─────┴─┴─────┴─┴─────┤
│OS   │ │OS   │ │OS   │          │    Docker Engine      │
├─────┤ ├─────┤ ├─────┤          ├───────────────────────┤
│  Hypervisor          │          │      Host OS           │
├───────────────────────┤          ├───────────────────────┤
│    Host Hardware      │          │    Host Hardware       │
└───────────────────────┘          └───────────────────────┘

VMs:  Heavy (~GB), slow boot (minutes), full OS isolation
Containers: Light (~MB), fast boot (seconds), shared kernel
```

| Feature       | VM                     | Container              |
|---------------|------------------------|------------------------|
| Size          | Gigabytes              | Megabytes              |
| Boot time     | Minutes                | Seconds                |
| Isolation     | Full hardware-level    | Process-level          |
| Performance   | Near-native with overhead | Near-native          |
| Portability   | Hypervisor-dependent   | Runs anywhere Docker runs |
| Use case      | Full OS needed         | Microservices, CI/CD   |

### 1.2 Dockerfile Instructions

A `Dockerfile` is a text file with instructions to build an image layer by layer.

| Instruction  | Purpose                                       | Example                                      |
|-------------|-----------------------------------------------|----------------------------------------------|
| `FROM`      | Base image                                    | `FROM python:3.11-slim`                      |
| `RUN`       | Execute command during build                  | `RUN pip install flask`                      |
| `COPY`      | Copy files from host to image                 | `COPY . /app`                                |
| `ADD`       | Like COPY but can extract tars and fetch URLs | `ADD archive.tar.gz /app`                    |
| `WORKDIR`   | Set working directory                         | `WORKDIR /app`                               |
| `ENV`       | Set environment variable                      | `ENV PYTHONUNBUFFERED=1`                     |
| `EXPOSE`    | Document which port the container listens on  | `EXPOSE 8000`                                |
| `CMD`       | Default command (overridable)                 | `CMD ["python", "main.py"]`                  |
| `ENTRYPOINT`| Fixed command (args appended)                 | `ENTRYPOINT ["uvicorn"]`                     |
| `ARG`       | Build-time variable                           | `ARG MODEL_VERSION=v1`                       |
| `VOLUME`    | Mount point for external storage              | `VOLUME /data`                               |

**CMD vs ENTRYPOINT:**

```dockerfile
# CMD -- overridable at runtime
CMD ["python", "main.py"]
# docker run myimage              -> python main.py
# docker run myimage bash         -> bash  (overridden)

# ENTRYPOINT -- fixed command, CMD becomes default args
ENTRYPOINT ["uvicorn"]
CMD ["main:app", "--host", "0.0.0.0"]
# docker run myimage              -> uvicorn main:app --host 0.0.0.0
# docker run myimage other:app    -> uvicorn other:app  (args replaced)
```

### 1.3 Multi-Stage Builds

Multi-stage builds use multiple `FROM` statements. Only the final stage goes into the
production image. This keeps images small by discarding build tools and intermediate files.

```dockerfile
# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build-only dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: Production ----
FROM python:3.11-slim

WORKDIR /app

# Copy only installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why multi-stage?**

```
Without multi-stage:          With multi-stage:
┌──────────────────┐          ┌──────────────────┐
│ gcc, g++ (200MB) │          │                  │
│ pip cache (150MB)│          │ Python runtime   │
│ Python runtime   │    =>    │ App dependencies │
│ App dependencies │          │ App code         │
│ App code         │          │                  │
│ Total: ~1.2 GB   │          │ Total: ~250 MB   │
└──────────────────┘          └──────────────────┘
```

### 1.4 Docker Compose

Docker Compose defines multi-container applications in a single YAML file.

```yaml
# docker-compose.yml -- FastAPI + Qdrant + PostgreSQL
version: "3.9"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/app
      - QDRANT_URL=http://qdrant:6333
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./app:/app/app          # hot-reload in dev
    depends_on:
      postgres:
        condition: service_healthy
      qdrant:
        condition: service_started
    networks:
      - backend

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: app
    volumes:
      - pg_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 5s
      timeout: 3s
      retries: 5
    networks:
      - backend

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    networks:
      - backend

volumes:
  pg_data:
  qdrant_data:

networks:
  backend:
    driver: bridge
```

Key commands:

```bash
docker compose up -d            # Start all services in background
docker compose down             # Stop and remove containers
docker compose logs -f api      # Follow logs for api service
docker compose exec api bash    # Shell into running container
docker compose build --no-cache # Rebuild without cache
docker compose ps               # List running services
```

### 1.5 Volumes and Networking

**Volumes** persist data beyond the container lifecycle:

```
Named Volume:      Bind Mount:         tmpfs:
┌──────────┐      ┌──────────┐        ┌──────────┐
│Container │      │Container │        │Container │
│ /data ───┼──┐   │ /app ────┼──┐     │ /tmp ────┼──┐
└──────────┘  │   └──────────┘  │     └──────────┘  │
              ▼                 ▼                    ▼
      Docker-managed       Host directory        RAM only
      /var/lib/docker/     ./local/path          (no disk)
      volumes/xyz/
```

```bash
# Named volume
docker run -v model_cache:/root/.cache/huggingface myapp

# Bind mount (host path)
docker run -v $(pwd)/data:/app/data myapp

# tmpfs (in-memory, e.g. for secrets)
docker run --tmpfs /run/secrets myapp
```

**Networking** -- containers communicate via service names:

```bash
# Default bridge -- manual linking
docker network create mynet
docker run --network mynet --name db postgres
docker run --network mynet myapp    # can reach "db" by hostname

# In Compose, services auto-resolve by service name
# api can connect to postgres:5432 and qdrant:6333
```

### 1.6 Best Practices for Python / AI Projects

```dockerfile
# 1. Use slim base images
FROM python:3.11-slim          # NOT python:3.11 (saves ~600MB)

# 2. Pin versions
FROM python:3.11.7-slim

# 3. Avoid running as root
RUN useradd -m appuser
USER appuser

# 4. Combine RUN commands to reduce layers
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# 5. Copy requirements first (leverage cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .                       # Code changes don't bust pip cache

# 6. Use .dockerignore
# 7. Set PYTHONUNBUFFERED for real-time logs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 8. Health checks
HEALTHCHECK --interval=30s --timeout=5s \
    CMD curl -f http://localhost:8000/health || exit 1
```

### 1.7 .dockerignore

```
# .dockerignore
__pycache__/
*.pyc
*.pyo
.git/
.github/
.env
.env.*
*.egg-info/
dist/
build/
.venv/
venv/
node_modules/
.mypy_cache/
.pytest_cache/
.ruff_cache/
*.log
models/              # Large model files -- use volumes instead
data/                # Large datasets
*.ipynb_checkpoints/
Dockerfile
docker-compose*.yml
README.md
docs/
tests/
```

---

## 2. CI/CD

### 2.1 What is CI/CD?

```
CI (Continuous Integration)              CD (Continuous Delivery/Deployment)
─────────────────────────────            ──────────────────────────────────
Developers merge code frequently         Automatically deploy to production
into a shared branch. Each merge         (Deployment) or make release-ready
triggers automated builds and tests.     (Delivery) after passing all checks.

┌──────────────────────────────────────────────────────────────────────┐
│                        CI/CD Pipeline                                │
│                                                                      │
│  Code Push ──> Build ──> Test ──> Lint ──> Security ──> Deploy       │
│      │          │         │        │         │            │           │
│     Git       Docker    pytest   ruff    Trivy/Bandit  K8s/Cloud     │
│                                                                      │
│  ◄──── Continuous Integration ────►◄── Continuous Deployment ──►     │
└──────────────────────────────────────────────────────────────────────┘
```

**Key principles:**
- Every commit triggers the pipeline
- Fast feedback (fail early)
- Automated testing at every stage
- Infrastructure as code
- Immutable deployments (build once, deploy anywhere)

### 2.2 GitHub Actions

GitHub Actions is the most widely used CI/CD platform for open-source and many commercial
projects. Workflows are defined in `.github/workflows/*.yml`.

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.11"
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # ── Job 1: Lint and Type Check ──
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install ruff mypy

      - name: Run ruff linter
        run: ruff check .

      - name: Run ruff formatter check
        run: ruff format --check .

      - name: Run mypy
        run: mypy app/ --ignore-missing-imports

  # ── Job 2: Test ──
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test_db
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: "pip"

      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt

      - name: Run tests with coverage
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/test_db
        run: |
          pytest tests/ -v --cov=app --cov-report=xml --cov-report=term

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: coverage.xml

  # ── Job 3: Security Scan ──
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Bandit (Python security)
        run: |
          pip install bandit
          bandit -r app/ -ll

      - name: Run Trivy (dependency scan)
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: fs
          scan-ref: .

  # ── Job 4: Build and Push Docker Image ──
  build:
    needs: [lint, test, security]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}

  # ── Job 5: Deploy ──
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production

    steps:
      - name: Deploy to production
        run: |
          echo "Deploying ${{ github.sha }} to production..."
          # kubectl set image deployment/api api=$REGISTRY/$IMAGE_NAME:${{ github.sha }}
          # or: ssh deploy@server 'docker pull ... && docker compose up -d'
```

### 2.3 GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - lint
  - test
  - build
  - deploy

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.pip-cache"

cache:
  paths:
    - .pip-cache/

lint:
  stage: lint
  image: python:3.11-slim
  script:
    - pip install ruff
    - ruff check .
    - ruff format --check .

test:
  stage: test
  image: python:3.11-slim
  services:
    - postgres:16-alpine
  variables:
    POSTGRES_DB: test_db
    POSTGRES_USER: test
    POSTGRES_PASSWORD: test
    DATABASE_URL: postgresql://test:test@postgres:5432/test_db
  script:
    - pip install -r requirements.txt -r requirements-dev.txt
    - pytest tests/ -v --cov=app
  coverage: '/^TOTAL.*\s+(\d+%)$/'

build:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main

deploy:
  stage: deploy
  script:
    - echo "Deploy to production"
  environment:
    name: production
  when: manual
  only:
    - main
```

### 2.4 Testing Strategies

```
Testing Pyramid:

           /\
          /  \          E2E Tests (few, slow, expensive)
         / E2E\         - Full user workflows
        /──────\        - Browser or API-level
       /        \
      /Integration\     Integration Tests (moderate)
     /─────────────\    - DB queries, API calls, external services
    /               \
   /   Unit Tests    \  Unit Tests (many, fast, cheap)
  /───────────────────\ - Individual functions, classes
```

```python
# tests/unit/test_chunker.py -- Unit test
def test_chunk_text_splits_correctly():
    text = "word " * 100
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)

# tests/integration/test_api.py -- Integration test
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_search_endpoint(async_client: AsyncClient, seeded_db):
    response = await async_client.post(
        "/api/v1/search",
        json={"query": "machine learning basics", "top_k": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) <= 5
    assert all("score" in r for r in data["results"])

# tests/e2e/test_rag_pipeline.py -- End-to-end test
@pytest.mark.e2e
async def test_full_rag_pipeline():
    # Ingest document
    ingest_resp = await client.post("/ingest", files={"file": pdf_file})
    assert ingest_resp.status_code == 200

    # Wait for processing
    job_id = ingest_resp.json()["job_id"]
    await wait_for_job(job_id)

    # Query
    query_resp = await client.post("/query", json={"question": "What is RAG?"})
    assert query_resp.status_code == 200
    assert "retrieval" in query_resp.json()["answer"].lower()
```

### 2.5 Environment Management

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│    Dev      │────>│  Staging    │────>│ Production  │
│             │     │             │     │             │
│ Local Docker│     │ Cloud replica│    │ Full infra  │
│ Mock APIs   │     │ Real APIs   │     │ Real APIs   │
│ Debug on    │     │ Debug off   │     │ Debug off   │
│ Seed data   │     │ Subset data │     │ Real data   │
└────────────┘     └────────────┘     └────────────┘
     auto               auto              manual/auto
    deploy             deploy              deploy
```

```python
# config.py -- Environment-aware configuration
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    environment: str = "development"
    debug: bool = False
    database_url: str
    openai_api_key: str
    qdrant_url: str = "http://localhost:6333"
    log_level: str = "INFO"

    # Production overrides
    max_workers: int = 4
    rate_limit: int = 100  # requests per minute

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

### 2.6 Secrets Management

```yaml
# GitHub Actions -- use repository secrets
steps:
  - name: Deploy
    env:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      DATABASE_URL: ${{ secrets.DATABASE_URL }}
    run: ./deploy.sh
```

```python
# In application code -- never hardcode secrets
import os

# Good
api_key = os.environ["OPENAI_API_KEY"]

# Better -- with validation
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str  # Raises error if missing
```

**Secrets hierarchy (most to least recommended):**

1. Cloud secret managers (AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault)
2. CI/CD platform secrets (GitHub Secrets, GitLab Variables)
3. Environment variables injected at deploy time
4. `.env` files (local dev only, NEVER committed)

### 2.7 Deployment and Rollback Strategies

```
Blue-Green Deployment:
┌──────────────┐         ┌──────────────┐
│  Blue (v1)   │◄── LB   │  Green (v2)  │   1. Deploy v2 to Green
│  (current)   │         │  (new)       │   2. Test Green
│              │         │              │   3. Switch LB to Green
└──────────────┘         └──────────────┘   4. Blue becomes standby

Canary Deployment:
                    ┌──────────────┐
              90% ──│   v1 (old)   │
Users ── LB ──┤     └──────────────┘
              10% ──┌──────────────┐     Gradually shift traffic:
                    │   v2 (new)   │     10% -> 25% -> 50% -> 100%
                    └──────────────┘

Rolling Update:
Pod1 [v1] -> [v2]                        Replace one instance at a time
Pod2 [v1] ────────> [v2]                 No downtime
Pod3 [v1] ──────────────> [v2]           Slower rollout
```

**Rollback example:**

```bash
# Docker Compose -- rollback to previous image tag
docker compose pull                          # pull latest
docker compose up -d                         # deploy
# If something is wrong:
docker compose down
docker compose -f docker-compose.yml up -d   # use pinned tag

# Kubernetes rollback
kubectl rollout undo deployment/api
kubectl rollout status deployment/api

# AWS ECS
aws ecs update-service --cluster prod \
    --service api \
    --task-definition api-task:42  # previous revision number
```

---

## 3. MLOps Overview

### 3.1 What is MLOps?

MLOps (Machine Learning Operations) applies DevOps principles to the ML lifecycle. It bridges
the gap between model development and production deployment by automating training,
validation, deployment, and monitoring.

```
Traditional Software:       ML Systems:
Code ──> Build ──> Deploy   Code ──> Build ──> Deploy
                               +        +        +
                            Data ──> Train ──> Monitor
                                      +          +
                                    Model ──> Retrain
```

### 3.2 ML Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ML Lifecycle                                  │
│                                                                      │
│  Data      Feature       Train      Evaluate    Deploy    Monitor   │
│  Prep  --> Engineering -> Model  --> Model   --> Model --> Model    │
│   │          │             │          │           │         │        │
│   ▼          ▼             ▼          ▼           ▼         ▼        │
│  DVC     Feature        MLflow     Metrics     Docker   Prometheus  │
│  Great   Store          W&B        Custom      K8s      Grafana     │
│  Expect. Feast          Optuna     Eval        vLLM     Langfuse    │
│                                                                      │
│  ◄─────────── Experiment Loop ───────────►◄── Production Loop ──►   │
│                                                                      │
│                    ▲                              │                   │
│                    └── Retrain on drift/decay ────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 MLOps Maturity Levels

| Level | Name             | Description                                              |
|-------|------------------|----------------------------------------------------------|
| 0     | No MLOps         | Manual everything. Jupyter notebooks. No versioning.     |
| 1     | DevOps, no MLOps | CI/CD for code, but models are manually trained/deployed.|
| 2     | Automated Training | Automated training pipelines. Experiment tracking.     |
| 3     | Automated Deploy | Model registry. Auto-deploy validated models. A/B tests. |
| 4     | Full MLOps       | Auto-retrain on drift. Continuous monitoring. Feature store.|

### 3.4 Key Challenges

**Reproducibility:**
- Same data + same code + same config must produce the same model
- Track random seeds, library versions, hardware specs
- Version data alongside code

**Versioning:**
- Code (Git), Data (DVC), Models (MLflow Registry), Configs (Git)

**Drift:**
```
Data Drift:                     Concept Drift:
Distribution of inputs          Relationship between input
changes over time               and output changes

Training data:  [■■■■■□□□]     Training: rain -> umbrella
Production:     [□□□■■■■■]     Production: rain -> no umbrella
                                (people stopped caring)
```

---

## 4. MLflow

### 4.1 What is MLflow?

MLflow is an open-source platform for managing the ML lifecycle. It has four main components:

```
┌─────────────────────────────────────────────────────┐
│                    MLflow                             │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ Tracking │  │  Models   │  │  Model Registry   │  │
│  │          │  │          │  │                   │  │
│  │ Log:     │  │ Package  │  │ Stage:            │  │
│  │ -params  │  │ models   │  │ None -> Staging   │  │
│  │ -metrics │  │ in a     │  │   -> Production   │  │
│  │ -artifacts│ │ standard │  │   -> Archived     │  │
│  │ -code    │  │ format   │  │                   │  │
│  └──────────┘  └──────────┘  └───────────────────┘  │
│                                                       │
│  ┌──────────────────────────────────────────────────┐ │
│  │              Projects                             │ │
│  │  Define reproducible runs with MLproject file     │ │
│  └──────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### 4.2 MLflow Tracking

```python
import mlflow
from mlflow.models import infer_signature

# Configure tracking server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("rag-pipeline-v2")

# Start a run
with mlflow.start_run(run_name="chunk-512-top5"):
    # Log parameters
    mlflow.log_param("chunk_size", 512)
    mlflow.log_param("chunk_overlap", 50)
    mlflow.log_param("embedding_model", "text-embedding-3-small")
    mlflow.log_param("top_k", 5)
    mlflow.log_param("reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    mlflow.log_param("llm_model", "gpt-4o-mini")
    mlflow.log_param("temperature", 0.1)

    # Run experiment
    results = evaluate_rag_pipeline(
        test_queries=test_set,
        chunk_size=512,
        top_k=5,
    )

    # Log metrics
    mlflow.log_metric("precision_at_5", results["precision"])
    mlflow.log_metric("recall_at_5", results["recall"])
    mlflow.log_metric("mrr", results["mrr"])
    mlflow.log_metric("avg_latency_ms", results["avg_latency"])
    mlflow.log_metric("avg_tokens_used", results["avg_tokens"])
    mlflow.log_metric("faithfulness", results["faithfulness"])
    mlflow.log_metric("answer_relevancy", results["relevancy"])

    # Log artifacts
    mlflow.log_artifact("config.yaml")
    mlflow.log_artifact("eval_results.json")

    # Log the model
    mlflow.pyfunc.log_model(
        artifact_path="rag_model",
        python_model=rag_pipeline,
        signature=infer_signature(sample_input, sample_output),
    )

    # Set tags for easy filtering
    mlflow.set_tag("team", "search")
    mlflow.set_tag("dataset_version", "v3.2")
```

### 4.3 Comparing Experiments

```python
import mlflow

# Search runs
runs = mlflow.search_runs(
    experiment_names=["rag-pipeline-v2"],
    filter_string="metrics.mrr > 0.7 AND params.embedding_model = 'text-embedding-3-small'",
    order_by=["metrics.mrr DESC"],
    max_results=10,
)

print(runs[["params.chunk_size", "params.top_k", "metrics.mrr", "metrics.precision_at_5"]])

# Output:
#   params.chunk_size  params.top_k  metrics.mrr  metrics.precision_at_5
# 0              512             5        0.847                    0.82
# 1              256            10        0.831                    0.79
# 2             1024             5        0.798                    0.75
```

### 4.4 Model Registry Workflow

```
Developer trains model
        │
        ▼
┌──────────────┐    Register    ┌───────────────────────────────┐
│ MLflow Run   │ ──────────────>│     Model Registry             │
│ (experiment) │                │                                │
└──────────────┘                │  Version 1  [Archived]        │
                                │  Version 2  [Staging]    ◄─── Test
                                │  Version 3  [Production] ◄─── Approve
                                └───────────────────────────────┘
```

```python
import mlflow

# Register a model from a run
model_uri = f"runs:/{run_id}/rag_model"
model_details = mlflow.register_model(
    model_uri=model_uri,
    name="rag-pipeline",
)

# Transition stages
client = mlflow.tracking.MlflowClient()

# Move to staging
client.transition_model_version_stage(
    name="rag-pipeline",
    version=model_details.version,
    stage="Staging",
)

# After validation, promote to production
client.transition_model_version_stage(
    name="rag-pipeline",
    version=model_details.version,
    stage="Production",
)

# Load the production model
model = mlflow.pyfunc.load_model("models:/rag-pipeline/Production")
prediction = model.predict(input_data)
```

### 4.5 MLflow for LLM Tracking

```python
import mlflow

# Enable auto-logging for LLM calls
mlflow.openai.autolog()

# Or manually log LLM interactions
with mlflow.start_run():
    mlflow.log_param("model", "gpt-4o")
    mlflow.log_param("system_prompt_version", "v2.1")

    # Log a trace for a single LLM call
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
    )

    # Log token usage
    mlflow.log_metric("prompt_tokens", response.usage.prompt_tokens)
    mlflow.log_metric("completion_tokens", response.usage.completion_tokens)
    mlflow.log_metric("total_tokens", response.usage.total_tokens)

    # Log cost
    cost = calculate_cost(response.usage, model="gpt-4o")
    mlflow.log_metric("cost_usd", cost)

    # Log the full interaction as artifact
    mlflow.log_dict(
        {
            "input": user_query,
            "output": response.choices[0].message.content,
            "model": "gpt-4o",
            "tokens": response.usage.total_tokens,
        },
        "llm_interaction.json",
    )
```

### 4.6 MLflow Deployment with Docker

```dockerfile
# Serve an MLflow model via REST API
FROM python:3.11-slim

RUN pip install mlflow[extras]==2.12.1

# Copy model artifact
COPY ./model /opt/model

EXPOSE 5001

CMD ["mlflow", "models", "serve", \
     "--model-uri", "/opt/model", \
     "--host", "0.0.0.0", \
     "--port", "5001", \
     "--no-conda"]
```

---

## 5. Model Versioning

### 5.1 Why Version Models?

- **Reproducibility** -- recreate any past result
- **Rollback** -- revert to a working model if a new one degrades
- **Compliance** -- audit trail of what was deployed and when
- **Comparison** -- evaluate improvements across versions

### 5.2 DVC (Data Version Control)

DVC extends Git to handle large files (datasets, models) by storing pointers in Git and
actual data in remote storage (S3, GCS, Azure Blob, etc.).

```
Git repo:                    Remote storage (S3):
├── data/                    ├── ab/
│   └── train.csv.dvc  ─────│── cd1234...  (actual data)
├── models/                  ├── ef/
│   └── model.pkl.dvc  ─────│── gh5678...  (actual model)
├── dvc.yaml                 └── ...
├── dvc.lock
└── .dvc/
    └── config
```

```bash
# Initialize DVC in an existing Git repo
dvc init

# Track a large file
dvc add data/training_data.parquet
git add data/training_data.parquet.dvc data/.gitignore
git commit -m "Track training data with DVC"

# Configure remote storage
dvc remote add -d myremote s3://my-bucket/dvc-store
git add .dvc/config
git commit -m "Configure DVC remote"

# Push data to remote
dvc push

# Pull data on another machine
git clone <repo-url>
dvc pull

# Switch to a different data version
git checkout v1.0
dvc checkout
```

**DVC Pipelines** (`dvc.yaml`):

```yaml
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - src/prepare.py
      - data/raw/
    outs:
      - data/processed/

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/
    params:
      - train.epochs
      - train.learning_rate
    outs:
      - models/model.pkl
    metrics:
      - metrics.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - models/model.pkl
      - data/test/
    metrics:
      - eval_metrics.json:
          cache: false
```

```bash
dvc repro        # Reproduce pipeline (only reruns changed stages)
dvc metrics show # Show metrics across branches/commits
dvc plots diff   # Compare plots between experiments
```

### 5.3 Versioning Strategy Comparison

| Tool              | Tracks       | Storage             | Best For                    |
|-------------------|-------------|---------------------|-----------------------------|
| Git               | Code, config | Git repo            | Source code, small configs   |
| Git LFS           | Large files  | LFS server          | Models < 2GB, binary assets |
| DVC               | Data, models | S3/GCS/Azure/local  | Large datasets, pipelines   |
| MLflow Registry   | Models       | MLflow server/S3    | Model lifecycle management  |
| Hugging Face Hub  | Models       | HF servers          | Sharing pretrained models   |
| W&B Artifacts     | Any artifact | W&B cloud           | Experiment + artifact combo  |

### 5.4 Git LFS

```bash
# Install Git LFS
git lfs install

# Track file patterns
git lfs track "*.pkl"
git lfs track "*.onnx"
git lfs track "*.bin"

# This creates/updates .gitattributes
cat .gitattributes
# *.pkl filter=lfs diff=lfs merge=lfs -text
# *.onnx filter=lfs diff=lfs merge=lfs -text

git add .gitattributes
git add models/classifier.pkl
git commit -m "Add model via Git LFS"
git push  # Large file goes to LFS storage, pointer to Git
```

---

## 6. Monitoring LLM Applications

### 6.1 What to Monitor

```
┌──────────────────────────────────────────────────────────────┐
│              LLM Application Monitoring Layers                │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Infrastructure:  CPU, memory, GPU, disk, network      │   │
│  │  Tools: Prometheus + Grafana, Datadog, CloudWatch      │   │
│  └────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Application:  Latency, throughput, error rates, HTTP  │   │
│  │  Tools: Prometheus, OpenTelemetry, Sentry              │   │
│  └────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  LLM-Specific:  Token usage, costs, quality, traces   │   │
│  │  Tools: Langfuse, LangSmith, MLflow, custom metrics    │   │
│  └────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Business:  User satisfaction, task completion, ROI     │   │
│  │  Tools: Analytics, A/B testing, user feedback           │   │
│  └────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

**Key metrics to track:**

| Category     | Metric                      | Why It Matters                    |
|-------------|----------------------------|-----------------------------------|
| Latency     | Time to first token (TTFT)  | User-perceived responsiveness     |
| Latency     | Total response time         | End-to-end performance            |
| Latency     | Tokens per second           | Throughput capacity               |
| Cost        | Tokens per request          | Budget control                    |
| Cost        | Cost per query              | Unit economics                    |
| Quality     | Faithfulness score          | Hallucination detection           |
| Quality     | Answer relevancy            | Response usefulness               |
| Quality     | User feedback (thumbs)      | Direct user satisfaction          |
| Errors      | LLM API error rate          | Reliability                       |
| Errors      | Timeout rate                | Availability                      |
| Drift       | Embedding distribution shift| Input distribution changes        |
| Drift       | Query topic distribution    | Usage pattern changes             |

### 6.2 Prometheus + Grafana Setup

```python
# metrics.py -- Custom Prometheus metrics for an LLM API
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    generate_latest, CONTENT_TYPE_LATEST,
)

# Counters
llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM API requests",
    ["model", "endpoint", "status"],
)

# Histograms (for latency distribution)
llm_latency_seconds = Histogram(
    "llm_latency_seconds",
    "LLM request latency in seconds",
    ["model", "endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# Gauges
active_requests = Gauge(
    "llm_active_requests",
    "Currently active LLM requests",
)

# Token tracking
tokens_used = Counter(
    "llm_tokens_total",
    "Total tokens consumed",
    ["model", "type"],  # type: prompt, completion
)

cost_usd = Counter(
    "llm_cost_usd_total",
    "Total cost in USD",
    ["model"],
)
```

```python
# middleware.py -- FastAPI middleware for automatic metrics
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        active_requests.inc()
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            status = "success"
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.perf_counter() - start_time
            active_requests.dec()
            llm_requests_total.labels(
                model="gpt-4o",
                endpoint=request.url.path,
                status=status,
            ).inc()
            llm_latency_seconds.labels(
                model="gpt-4o",
                endpoint=request.url.path,
            ).observe(duration)

        return response
```

```yaml
# docker-compose.monitoring.yml
version: "3.9"

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus

volumes:
  prometheus_data:
  grafana_data:
```

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "fastapi-llm"
    static_configs:
      - targets: ["api:8000"]
    metrics_path: /metrics
```

### 6.3 Langfuse (Open-Source LLM Observability)

```python
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse(
    public_key="pk-...",
    secret_key="sk-...",
    host="http://localhost:3000",  # self-hosted
)

@observe()                      # Automatic tracing
def rag_pipeline(query: str) -> str:
    # Retrieval step -- automatically traced
    docs = retrieve_documents(query)

    # Generation step -- automatically traced
    answer = generate_answer(query, docs)

    # Score the trace
    langfuse_context.score_current_trace(
        name="user_feedback",
        value=1,  # or 0 for negative
    )

    return answer

@observe()
def retrieve_documents(query: str) -> list[dict]:
    embedding = embed(query)
    results = vector_db.search(embedding, limit=5)

    # Log retrieval quality
    langfuse_context.update_current_observation(
        metadata={"num_results": len(results)},
    )
    return results

@observe()
def generate_answer(query: str, docs: list[dict]) -> str:
    context = "\n".join(d["text"] for d in docs)
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Context: {context}"},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content
```

### 6.4 OpenTelemetry Tracing

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Setup
provider = TracerProvider()
exporter = OTLPSpanExporter(endpoint="http://jaeger:4317")
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("rag-service")

async def handle_query(query: str):
    with tracer.start_as_current_span("rag_query") as span:
        span.set_attribute("query.text", query)
        span.set_attribute("query.length", len(query))

        # Retrieval span
        with tracer.start_as_current_span("retrieval") as ret_span:
            docs = await retrieve(query)
            ret_span.set_attribute("docs.count", len(docs))

        # Generation span
        with tracer.start_as_current_span("generation") as gen_span:
            answer = await generate(query, docs)
            gen_span.set_attribute("tokens.total", answer.usage.total_tokens)

        span.set_attribute("response.length", len(answer.text))
        return answer
```

Resulting trace visualization:

```
rag_query (350ms)
├── retrieval (50ms)
│   ├── embed_query (10ms)
│   └── vector_search (40ms)
└── generation (300ms)
    ├── build_prompt (2ms)
    └── llm_call (298ms)
```

### 6.5 Alerting Strategies

```yaml
# Prometheus alerting rules (alerts.yml)
groups:
  - name: llm_alerts
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(llm_latency_seconds_bucket[5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency exceeds 10 seconds"

      - alert: HighErrorRate
        expr: rate(llm_requests_total{status="error"}[5m]) / rate(llm_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Error rate exceeds 5%"

      - alert: HighCost
        expr: increase(llm_cost_usd_total[1h]) > 100
        labels:
          severity: warning
        annotations:
          summary: "Hourly LLM cost exceeds $100"

      - alert: QualityDegradation
        expr: avg_over_time(llm_faithfulness_score[1h]) < 0.7
        for: 30m
        labels:
          severity: critical
        annotations:
          summary: "Average faithfulness score dropped below 0.7"
```

### 6.6 Cost Monitoring

```python
# cost_tracker.py
from dataclasses import dataclass

@dataclass
class ModelPricing:
    """Pricing per 1M tokens (USD)."""
    prompt: float
    completion: float

PRICING = {
    "gpt-4o": ModelPricing(prompt=2.50, completion=10.00),
    "gpt-4o-mini": ModelPricing(prompt=0.15, completion=0.60),
    "gpt-4.1": ModelPricing(prompt=2.00, completion=8.00),
    "gpt-4.1-mini": ModelPricing(prompt=0.40, completion=1.60),
    "claude-sonnet-4-20250514": ModelPricing(prompt=3.00, completion=15.00),
}

def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    pricing = PRICING[model]
    cost = (
        (prompt_tokens / 1_000_000) * pricing.prompt
        + (completion_tokens / 1_000_000) * pricing.completion
    )
    return round(cost, 6)

# Usage with metrics
def track_llm_call(model: str, usage):
    cost = calculate_cost(model, usage.prompt_tokens, usage.completion_tokens)
    tokens_used.labels(model=model, type="prompt").inc(usage.prompt_tokens)
    tokens_used.labels(model=model, type="completion").inc(usage.completion_tokens)
    cost_usd.labels(model=model).inc(cost)
    return cost
```

---

## 7. Infrastructure for AI

### 7.1 GPU Management

```dockerfile
# Dockerfile for GPU workloads
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

CMD ["python3", "serve.py"]
```

```yaml
# docker-compose with GPU support
services:
  inference:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1        # or "all"
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
```

```bash
# Run with GPU
docker run --gpus all myimage
docker run --gpus '"device=0,1"' myimage

# Check GPU inside container
nvidia-smi
```

### 7.2 Kubernetes Basics for ML

```yaml
# k8s deployment for a model serving API
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-api
  labels:
    app: llm-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-api
  template:
    metadata:
      labels:
        app: llm-api
    spec:
      containers:
        - name: api
          image: ghcr.io/myorg/llm-api:v1.2.0
          ports:
            - containerPort: 8000
          resources:
            requests:
              memory: "2Gi"
              cpu: "1"
            limits:
              memory: "4Gi"
              cpu: "2"
              nvidia.com/gpu: 1    # GPU request
          env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: llm-secrets
                  key: openai-api-key
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: llm-api-service
spec:
  selector:
    app: llm-api
  ports:
    - port: 80
      targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "50"
```

### 7.3 Model Serving Frameworks

```
┌──────────────────────────────────────────────────────────────────┐
│                    Model Serving Options                          │
│                                                                    │
│  Framework     Best For                 Key Feature                │
│  ─────────     ────────                 ───────────                │
│  vLLM          LLM inference            PagedAttention, fast       │
│  TGI           HF models in production  Built-in batching          │
│  Triton        Multi-framework serving  GPU scheduling             │
│  TorchServe    PyTorch models           Native PyTorch support     │
│  Ollama        Local LLM dev/testing    Easy setup                 │
│  Ray Serve     Complex pipelines        Scaling, composition       │
└──────────────────────────────────────────────────────────────────┘
```

**vLLM example:**

```bash
# Serve a model with vLLM
pip install vllm

# Start the server (OpenAI-compatible API)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
```

```python
# Client code -- uses standard OpenAI SDK
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Explain RAG in 3 sentences."}],
    max_tokens=200,
)
```

### 7.4 Caching Layers

```python
import hashlib
import json
import redis

redis_client = redis.Redis(host="localhost", port=6379, db=0)

def cached_llm_call(
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    ttl: int = 3600,
) -> str:
    """Cache LLM responses. Only effective when temperature=0."""
    if temperature > 0:
        # Non-deterministic outputs -- skip cache
        return call_llm(model, messages, temperature)

    # Build cache key from inputs
    cache_key = hashlib.sha256(
        json.dumps({"model": model, "messages": messages}, sort_keys=True).encode()
    ).hexdigest()

    # Check cache
    cached = redis_client.get(f"llm:{cache_key}")
    if cached:
        return cached.decode("utf-8")

    # Call LLM
    response = call_llm(model, messages, temperature)

    # Cache response
    redis_client.setex(f"llm:{cache_key}", ttl, response)
    return response
```

```
Request Flow with Cache:

User Query ──> Hash ──> Redis Lookup
                            │
                     ┌──────┴──────┐
                     │ Cache Hit?  │
                     └──────┬──────┘
                       Yes/    \No
                      /         \
              Return cached    Call LLM API
              response         Store in Redis
                               Return response
```

---

## 8. Data Pipeline Management

### 8.1 ETL/ELT for ML

```
ETL (Extract, Transform, Load):
Source ──> Transform ──> Load to Target
           (clean,        (warehouse,
            enrich)        feature store)

ELT (Extract, Load, Transform):
Source ──> Load to Target ──> Transform
           (raw data)         (in the warehouse)

For ML pipelines, ELT is often preferred because:
- Raw data is preserved for experimentation
- Transformations can be rerun with different parameters
- Feature engineering happens closer to training
```

```python
# Simplified data pipeline for ML
from pathlib import Path
import pandas as pd
from datetime import datetime

class DataPipeline:
    """ETL pipeline with validation and versioning."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

    def extract(self, source: str) -> pd.DataFrame:
        """Extract data from source."""
        if source.endswith(".csv"):
            return pd.read_csv(source)
        elif source.endswith(".parquet"):
            return pd.read_parquet(source)
        elif source.startswith("postgresql://"):
            return pd.read_sql("SELECT * FROM documents", source)
        else:
            raise ValueError(f"Unsupported source: {source}")

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate data quality."""
        checks = {
            "no_empty_df": len(df) > 0,
            "no_null_ids": df["id"].notna().all(),
            "no_duplicate_ids": df["id"].is_unique,
            "text_not_empty": (df["text"].str.len() > 0).all(),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"Validation failed: {failed}")
        return True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and transform data."""
        df = df.drop_duplicates(subset=["id"])
        df["text"] = df["text"].str.strip()
        df["text_length"] = df["text"].str.len()
        df["processed_at"] = datetime.utcnow().isoformat()
        return df

    def load(self, df: pd.DataFrame, version: str) -> Path:
        """Save processed data with version."""
        output_path = self.processed_dir / f"data_{version}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        return output_path

    def run(self, source: str, version: str) -> Path:
        """Run the full pipeline."""
        df = self.extract(source)
        self.validate(df)
        df = self.transform(df)
        self.validate(df)  # validate again after transform
        return self.load(df, version)
```

### 8.2 Data Validation with Great Expectations

```python
import great_expectations as gx

context = gx.get_context()

# Define expectations for training data
suite = context.add_expectation_suite("training_data_suite")

# Column expectations
suite.add_expectation(
    gx.expectations.ExpectColumnToExist(column="text")
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(column="text")
)
suite.add_expectation(
    gx.expectations.ExpectColumnValueLengthsToBeBetween(
        column="text", min_value=10, max_value=10000
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeUnique(column="id")
)

# Table expectations
suite.add_expectation(
    gx.expectations.ExpectTableRowCountToBeBetween(
        min_value=1000, max_value=1000000
    )
)
```

### 8.3 Feature Stores

```
┌────────────────────────────────────────────────────────┐
│                    Feature Store                        │
│                                                          │
│  Offline Store (batch)         Online Store (real-time) │
│  ┌──────────────────┐         ┌──────────────────┐     │
│  │ Parquet / BigQuery│         │ Redis / DynamoDB │     │
│  │                  │         │                  │     │
│  │ Historical data  │  sync   │ Latest features  │     │
│  │ for training     │ ──────> │ for inference    │     │
│  └──────────────────┘         └──────────────────┘     │
│                                                          │
│  Feature definitions: shared across training & serving  │
└────────────────────────────────────────────────────────┘
```

```python
# Feast example -- feature_store.yaml
"""
project: ml_project
registry: data/registry.db
provider: local
online_store:
  type: redis
  connection_string: localhost:6379
offline_store:
  type: file
"""

# feature definitions (features.py)
from feast import Entity, FeatureView, Field
from feast.types import Float32, String, Int64
from feast.infra.offline_stores.file_source import FileSource

user = Entity(name="user_id", join_keys=["user_id"])

user_features = FeatureView(
    name="user_features",
    entities=[user],
    schema=[
        Field(name="total_queries", dtype=Int64),
        Field(name="avg_query_length", dtype=Float32),
        Field(name="preferred_topic", dtype=String),
    ],
    source=FileSource(path="data/user_features.parquet"),
    ttl=timedelta(days=1),
)
```

---

## 9. Security in MLOps

### 9.1 Security Layers

```
┌──────────────────────────────────────────────────────────┐
│                  MLOps Security Layers                     │
│                                                            │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Code Security                                      │   │
│  │  - Dependency scanning (pip-audit, safety)          │   │
│  │  - Static analysis (bandit, semgrep)                │   │
│  │  - Secret detection (detect-secrets, trufflehog)    │   │
│  └────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Container Security                                 │   │
│  │  - Image scanning (Trivy, Snyk)                     │   │
│  │  - Non-root users                                   │   │
│  │  - Minimal base images                              │   │
│  │  - No secrets in images                             │   │
│  └────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Model Security                                     │   │
│  │  - Prompt injection protection                      │   │
│  │  - Input/output validation                          │   │
│  │  - Model access control                             │   │
│  │  - Adversarial input detection                      │   │
│  └────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Data Privacy                                       │   │
│  │  - PII detection and masking                        │   │
│  │  - Data encryption at rest and in transit           │   │
│  │  - Access control and audit logging                 │   │
│  │  - GDPR/CCPA compliance                             │   │
│  └────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### 9.2 Container Security

```dockerfile
# Secure Dockerfile practices

# 1. Use specific, minimal base image
FROM python:3.11.7-slim-bookworm

# 2. Non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app appuser

# 3. No secrets in build
# BAD:  ENV API_KEY=sk-12345
# GOOD: Read from runtime environment or secrets manager

# 4. Copy only needed files
COPY --chown=appuser:appgroup requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appgroup app/ ./app/

# 5. Switch to non-root
USER appuser

# 6. Read-only filesystem where possible
# docker run --read-only --tmpfs /tmp myimage
```

```bash
# Scan image for vulnerabilities
trivy image myapp:latest

# Scan for high/critical only
trivy image --severity HIGH,CRITICAL myapp:latest

# Scan filesystem
trivy fs --security-checks vuln,secret,config .
```

### 9.3 Secret Management

```python
# secrets_manager.py -- Abstraction over secret backends
import os
from abc import ABC, abstractmethod

class SecretManager(ABC):
    @abstractmethod
    def get_secret(self, key: str) -> str:
        ...

class EnvSecretManager(SecretManager):
    """Read from environment variables (simplest)."""
    def get_secret(self, key: str) -> str:
        value = os.environ.get(key)
        if value is None:
            raise ValueError(f"Secret {key} not found in environment")
        return value

class AWSSecretManager(SecretManager):
    """Read from AWS Secrets Manager."""
    def __init__(self):
        import boto3
        self.client = boto3.client("secretsmanager")

    def get_secret(self, key: str) -> str:
        response = self.client.get_secret_value(SecretId=key)
        return response["SecretString"]

class VaultSecretManager(SecretManager):
    """Read from HashiCorp Vault."""
    def __init__(self, vault_addr: str, token: str):
        import hvac
        self.client = hvac.Client(url=vault_addr, token=token)

    def get_secret(self, key: str) -> str:
        secret = self.client.secrets.kv.v2.read_secret_version(path=key)
        return secret["data"]["data"]["value"]

# Usage
def get_secret_manager() -> SecretManager:
    env = os.getenv("ENVIRONMENT", "development")
    if env == "production":
        return AWSSecretManager()
    return EnvSecretManager()

secrets = get_secret_manager()
api_key = secrets.get_secret("OPENAI_API_KEY")
```

### 9.4 Prompt Injection Protection

```python
# input_validation.py
import re

class InputValidator:
    """Validate and sanitize user inputs before sending to LLM."""

    INJECTION_PATTERNS = [
        r"ignore\s+(previous|above|all)\s+instructions",
        r"disregard\s+.*instructions",
        r"you\s+are\s+now\s+",
        r"new\s+instruction[s]?\s*:",
        r"system\s*prompt\s*:",
        r"<\s*system\s*>",
    ]

    MAX_INPUT_LENGTH = 10_000

    @classmethod
    def validate(cls, user_input: str) -> tuple[bool, str]:
        """Returns (is_valid, cleaned_input_or_error_message)."""
        if len(user_input) > cls.MAX_INPUT_LENGTH:
            return False, f"Input exceeds max length of {cls.MAX_INPUT_LENGTH}"

        lower_input = user_input.lower()
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, lower_input):
                return False, "Input contains potentially malicious content"

        return True, user_input.strip()
```

### 9.5 Dependency Scanning in CI

```yaml
# In GitHub Actions workflow
- name: Python dependency audit
  run: |
    pip install pip-audit
    pip-audit --requirement requirements.txt --strict

- name: Detect hardcoded secrets
  uses: trufflesecurity/trufflehog@main
  with:
    path: .
    extra_args: --only-verified
```

---

## 10. Q&A Section

### Q1: Explain Docker containers vs VMs.

**A:** Containers and VMs both provide isolation, but at different levels.

A **VM** runs a full guest operating system on top of a hypervisor, with each VM getting its
own kernel, memory, and virtual hardware. This provides strong isolation but consumes
significant resources (gigabytes of RAM, minutes to boot).

A **container** shares the host OS kernel and isolates processes using Linux namespaces and
cgroups. Containers package only the application and its dependencies (not the OS), resulting
in images measured in megabytes and boot times in seconds.

For AI/ML workloads, containers are preferred because they are lightweight, portable, and
start quickly -- important for scaling inference services. VMs are still used when strong
security isolation is needed (multi-tenant environments) or when running different OS types.

---

### Q2: What is a multi-stage Docker build and why use it?

**A:** A multi-stage build uses multiple `FROM` statements in one Dockerfile. Each stage can
use a different base image. Only the final stage becomes the production image; intermediate
stages are discarded.

**Why use it:**
- **Smaller images** -- build tools (gcc, pip cache) stay in the builder stage
- **Security** -- fewer packages in production means a smaller attack surface
- **Separation of concerns** -- build dependencies are separate from runtime

Example: a Python AI project might use a builder stage with `gcc` to compile C extensions,
then copy only the installed packages to a slim final image. This can reduce image size from
over 1 GB to around 250 MB.

---

### Q3: How do you set up CI/CD for a Python AI project?

**A:** A typical CI/CD pipeline for a Python AI project includes:

1. **Lint** -- `ruff check` and `ruff format --check` for code quality
2. **Type check** -- `mypy` for type safety
3. **Unit tests** -- `pytest` with coverage reporting
4. **Integration tests** -- test with real databases (Postgres, Qdrant) using CI service containers
5. **Security scan** -- `bandit` for Python security, `trivy` for dependency vulnerabilities
6. **Build** -- Docker image build and push to a container registry
7. **Deploy** -- automated deployment to staging; manual approval for production

Use GitHub Actions or GitLab CI. Cache pip dependencies between runs. Run lint and tests in
parallel to speed up the pipeline. Use environment-specific secrets and deploy to staging
automatically on `develop` branch, to production on `main` with approval gates.

---

### Q4: What is MLflow and what are its components?

**A:** MLflow is an open-source platform for managing the end-to-end ML lifecycle. Its four
components are:

1. **Tracking** -- log parameters, metrics, and artifacts for each experiment run. Enables
   comparison of runs to find the best configuration.
2. **Models** -- a standard format for packaging models so they can be deployed to any
   supported serving platform (REST API, Docker, Spark, etc.).
3. **Model Registry** -- a centralized store for versioning models with lifecycle stages
   (None, Staging, Production, Archived). Enables approval workflows.
4. **Projects** -- a convention for packaging code with dependencies (conda/pip) and entry
   points so experiments can be reproduced.

For LLM/RAG work, MLflow Tracking is most commonly used to compare chunk sizes, embedding
models, retrieval parameters, and quality metrics across experiments.

---

### Q5: How do you version ML models?

**A:** Multiple tools address different aspects of ML versioning:

- **Git** for source code and configuration files
- **DVC** for large datasets and model artifacts -- stores pointers in Git, actual data in S3/GCS
- **MLflow Model Registry** for model lifecycle management -- tracks versions, assigns stages
  (staging/production), and enables approval workflows
- **Git LFS** for moderately large files (under ~2 GB) that should live in Git
- **Hugging Face Hub** for sharing pretrained models with the community

A typical workflow: code is in Git, data tracked by DVC, experiment parameters/metrics in
MLflow Tracking, and validated models registered in MLflow Model Registry. Each model version
is linked to its training run, data version, and code commit for full reproducibility.

---

### Q6: What should you monitor in an LLM application?

**A:** LLM monitoring spans four layers:

1. **Infrastructure** -- CPU, memory, GPU utilization, disk I/O (Prometheus + Grafana)
2. **Application** -- request latency, throughput, error rates, HTTP status codes
3. **LLM-specific** -- time to first token, tokens per request, cost per query, prompt/completion
   token counts, LLM API error rates and timeouts
4. **Quality** -- faithfulness scores, answer relevancy, user feedback (thumbs up/down),
   retrieval precision, hallucination rates

Critical alerts should fire on: error rate > 5%, P95 latency > 10s, hourly cost spikes,
and quality score drops. Use Langfuse or LangSmith for LLM-specific tracing and evaluation.

---

### Q7: How do you handle secrets in CI/CD?

**A:** Secrets should NEVER be hardcoded in code or Docker images. Best practices:

1. **CI/CD platform secrets** -- GitHub Secrets, GitLab CI Variables. Injected as environment
   variables at runtime, masked in logs.
2. **Cloud secret managers** -- AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault.
   Application fetches secrets at startup. Supports rotation and audit logging.
3. **Environment variables** -- set at deployment time via the orchestrator (K8s Secrets,
   Docker Compose env files).
4. **`.env` files** -- for local development only, always listed in `.gitignore`.

In a GitHub Actions workflow, reference secrets as `${{ secrets.MY_SECRET }}`. For Kubernetes,
use `secretKeyRef` to mount secrets as environment variables in pods.

---

### Q8: What is DVC and when would you use it?

**A:** DVC (Data Version Control) extends Git to handle large files (datasets, model weights)
that are too big for Git. It stores lightweight pointer files (`.dvc` files) in Git while the
actual data lives in remote storage (S3, GCS, Azure Blob, local).

**Use DVC when:**
- You have datasets larger than ~100 MB that change over time
- You need to reproduce training runs with exact data versions
- You want to define ML pipelines (data prep -> train -> evaluate) with dependency tracking
- You need to share large files with your team without bloating the Git repo

DVC's `dvc repro` command reruns only the pipeline stages whose inputs have changed, saving
compute time. `dvc metrics show` compares metrics across Git branches or tags.

---

### Q9: How do you deploy a model with Docker?

**A:** The typical approach:

1. **Build a Docker image** containing the model serving code and dependencies
2. **Include the model artifact** via COPY (small models) or download at startup from S3/MLflow (large models)
3. **Expose an HTTP API** using FastAPI, Flask, or dedicated serving frameworks (vLLM, TGI, Triton)
4. **Health checks** for readiness (model loaded) and liveness (process alive)
5. **Run behind a load balancer** with multiple replicas

For LLMs, use specialized frameworks like vLLM which provide an OpenAI-compatible API,
continuous batching, and efficient GPU memory management (PagedAttention). Deploy the vLLM
container with GPU access (`--gpus all`) and point your application at it as if it were the
OpenAI API.

---

### Q10: What is the MLflow Model Registry?

**A:** The MLflow Model Registry is a centralized model store that provides:

- **Versioning** -- every registered model gets an auto-incrementing version number
- **Stage transitions** -- models move through None -> Staging -> Production -> Archived
- **Lineage** -- each version links back to the MLflow run that produced it
- **Annotations** -- add descriptions and tags for documentation
- **Approval workflows** -- teams can require review before promoting to Production

Workflow: a data scientist trains a model, logs it with MLflow Tracking, then registers it.
A reviewer evaluates it in Staging. Once approved, it transitions to Production. The serving
infrastructure loads the latest Production model. If problems arise, the previous Production
version can be restored.

---

### Q11: How do you handle GPU requirements in Docker?

**A:** GPU access in Docker requires:

1. **NVIDIA Container Toolkit** installed on the host (provides `nvidia-container-runtime`)
2. **Base image** with CUDA -- use `nvidia/cuda:12.x.x-runtime-ubuntuXX.XX` or PyTorch/TensorFlow images with CUDA
3. **Runtime flag** -- `docker run --gpus all` or `--gpus '"device=0"'`
4. **Docker Compose** -- use `deploy.resources.reservations.devices` with `capabilities: [gpu]`

For Kubernetes, install the NVIDIA device plugin and request GPUs via
`nvidia.com/gpu: 1` in resource limits. GPU sharing (MIG, time-slicing) can be configured
to serve multiple containers from one GPU.

---

### Q12: What is blue-green deployment?

**A:** Blue-green deployment maintains two identical production environments:

- **Blue** is the current live version
- **Green** is the new version being deployed

The process: deploy the new version to Green, run smoke tests, then switch the load balancer
from Blue to Green. If something goes wrong, switch back to Blue instantly (rollback in
seconds, not minutes).

**Advantages:** zero-downtime deployments, instant rollback, full testing in a production-like
environment.

**Disadvantages:** requires double the infrastructure (temporarily), database migrations need
careful handling since both versions may access the same database.

---

### Q13: How do you roll back a failed deployment?

**A:** Rollback strategy depends on the deployment method:

- **Docker Compose** -- `docker compose` with the previous image tag
- **Kubernetes** -- `kubectl rollout undo deployment/name` reverts to the previous ReplicaSet
- **Blue-green** -- switch the load balancer back to the Blue environment
- **Canary** -- stop sending traffic to the canary and scale it down
- **AWS ECS** -- `aws ecs update-service` with the previous task definition revision

Best practices: always tag images with commit SHA (not just `latest`), keep previous image
versions available, automate rollback triggers based on error rate or latency thresholds,
and test rollback procedures regularly.

---

### Q14: What is data drift and how do you detect it?

**A:** Data drift occurs when the statistical distribution of production data diverges from
the training data. This can degrade model performance even when the model itself has not
changed.

**Types:**
- **Covariate drift** -- input distribution changes (new topics, different language)
- **Concept drift** -- the relationship between inputs and outputs changes
- **Label drift** -- the distribution of target values shifts

**Detection methods:**
- Statistical tests (KS test, PSI -- Population Stability Index)
- Monitor embedding space distributions (cosine similarity of centroids)
- Track prediction confidence distributions over time
- Compare feature value distributions between training and production data

**Response:** alert when drift exceeds a threshold, investigate root cause, retrain with
recent data if the drift is genuine rather than a data quality issue.

---

### Q15: How do you manage environments (dev/staging/prod)?

**A:** Each environment serves a different purpose:

- **Development** -- local Docker Compose, mock external APIs, seed data, debug mode
- **Staging** -- cloud-hosted replica of production, real API integrations, subset of data
- **Production** -- full infrastructure, real data, monitoring, alerting

Implementation:
- Use environment variables or config files per environment (`pydantic-settings`)
- Docker images are built once, configured per environment via env vars
- CI/CD deploys automatically to staging, requires manual approval for production
- Use feature flags to test features in staging before enabling in production
- Database migrations run in staging first, validated before production

---

### Q16: How does Docker networking work in Compose?

**A:** Docker Compose creates a default bridge network for all services in the file.
Services can reach each other by service name (DNS resolution). For example, if you have
services `api` and `postgres`, the API can connect to `postgresql://user:pass@postgres:5432/db`.

You can define custom networks to isolate groups of services. Services on different networks
cannot communicate unless explicitly connected to both. Ports exposed via `ports:` are
accessible from the host; ports only exposed via `expose:` are accessible only within the
Docker network.

---

### Q17: What is OpenTelemetry and why use it for LLM apps?

**A:** OpenTelemetry (OTel) is a vendor-neutral standard for collecting traces, metrics, and
logs. For LLM applications, it provides:

- **Distributed tracing** -- see the full request journey: API -> retrieval -> embedding -> LLM call -> response
- **Latency breakdown** -- identify bottlenecks (is retrieval or generation slow?)
- **Error correlation** -- link errors to specific pipeline stages
- **Vendor agnostic** -- export to Jaeger, Zipkin, Datadog, or any OTel-compatible backend

Each operation creates a "span" with timing, attributes, and relationships. A full request
creates a "trace" composed of nested spans, making it easy to visualize where time is spent.

---

### Q18: What is Langfuse and how does it differ from LangSmith?

**A:** Both are LLM observability platforms, but they differ in approach:

**Langfuse** (open-source):
- Self-hostable, privacy-friendly
- Traces LLM calls with latency, tokens, costs
- Supports user feedback and evaluation datasets
- Integrates with LangChain, LlamaIndex, OpenAI SDK
- Free to self-host, cloud version available

**LangSmith** (proprietary, by LangChain):
- Cloud-hosted (managed service)
- Tight integration with LangChain framework
- Built-in evaluation and testing tools
- Dataset management for few-shot prompts
- Paid product

Choose Langfuse for self-hosting requirements or framework-agnostic setups. Choose LangSmith
if you are heavily invested in LangChain and want managed infrastructure.

---

### Q19: How do you implement canary deployments for ML models?

**A:** Canary deployment gradually shifts traffic from the old model to the new one:

1. Deploy the new model alongside the existing one
2. Route a small percentage (e.g., 5-10%) of traffic to the new model
3. Monitor key metrics (latency, error rate, quality scores)
4. If metrics are healthy, gradually increase traffic (25% -> 50% -> 100%)
5. If metrics degrade, route all traffic back to the old model

Implementation with Kubernetes:

```yaml
# Use Istio or similar service mesh for traffic splitting
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: model-service
spec:
  hosts:
    - model-service
  http:
    - route:
        - destination:
            host: model-service
            subset: v1
          weight: 90
        - destination:
            host: model-service
            subset: v2
          weight: 10
```

For ML specifically, track not just infrastructure metrics but also model quality metrics
(accuracy, faithfulness) during the canary phase.

---

### Q20: What is a feature store and why use one?

**A:** A feature store is a centralized system for storing, managing, and serving ML features.
It solves the "training-serving skew" problem by ensuring the same feature computation is
used during training and inference.

**Components:**
- **Offline store** -- historical features for training (stored in Parquet, BigQuery, etc.)
- **Online store** -- latest feature values for real-time inference (stored in Redis, DynamoDB)
- **Feature definitions** -- shared code defining how features are computed
- **Feature registry** -- catalog of available features for discovery

**Why use one:**
- Avoid rewriting feature logic for training vs serving
- Share features across teams and models
- Ensure consistency between training data and inference data
- Time-travel: get features as they were at a specific point in time

Popular tools: Feast (open-source), Tecton, Databricks Feature Store.

---

### Q21: How do you handle model A/B testing in production?

**A:** Model A/B testing compares two or more models on real production traffic:

1. **Split traffic** -- route users randomly to model A or model B (use consistent hashing
   so the same user always hits the same model)
2. **Collect metrics** -- track quality metrics, latency, user satisfaction per model
3. **Statistical significance** -- run the test long enough with enough traffic to reach
   significance (typically p < 0.05)
4. **Decide** -- promote the winner, archive the loser

```python
import hashlib

def get_model_variant(user_id: str, experiment: str = "rag-v2") -> str:
    """Deterministic assignment: same user always gets same variant."""
    hash_input = f"{user_id}:{experiment}"
    hash_val = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
    return "model_b" if hash_val % 100 < 10 else "model_a"  # 10% to B
```

---

### Q22: Explain the concept of Infrastructure as Code (IaC) for ML.

**A:** IaC manages infrastructure through version-controlled configuration files rather than
manual setup. For ML workloads:

- **Terraform** -- provision cloud resources (GPU instances, S3 buckets, databases)
- **Kubernetes manifests** -- define deployments, services, autoscalers
- **Docker Compose** -- define multi-service local/staging environments
- **Helm charts** -- templated Kubernetes deployments for reuse

Benefits: reproducible environments, version history, code review for infra changes, easy
replication across regions. Every environment (dev, staging, prod) is defined in code and can
be recreated from scratch.

---

### Q23: How do you optimize Docker images for AI/ML workloads?

**A:** AI/ML images tend to be large due to frameworks and model weights. Optimization techniques:

1. **Multi-stage builds** -- separate build dependencies from runtime
2. **Slim base images** -- `python:3.11-slim` instead of `python:3.11` (saves ~600 MB)
3. **`--no-cache-dir`** -- `pip install --no-cache-dir` prevents pip from caching wheels
4. **Layer ordering** -- put rarely-changing layers (dependencies) before frequently-changing
   layers (app code)
5. **`.dockerignore`** -- exclude datasets, models, `.git/`, `__pycache__/`
6. **Download models at runtime** -- instead of baking large models into the image, download
   from S3/HF Hub at startup and cache in a volume
7. **Distroless images** -- Google distroless images have no shell or package manager

---

### Q24: What are the key differences between GitHub Actions and GitLab CI?

**A:**

| Feature         | GitHub Actions                | GitLab CI                      |
|----------------|-------------------------------|--------------------------------|
| Config file     | `.github/workflows/*.yml`     | `.gitlab-ci.yml`               |
| Runners         | GitHub-hosted or self-hosted  | GitLab-hosted or self-hosted   |
| Marketplace     | Large action marketplace      | Templates, smaller ecosystem   |
| Container registry | GHCR (ghcr.io)            | Built-in per-project           |
| Secrets         | Repository/Org secrets        | CI/CD Variables (project/group)|
| Caching         | `actions/cache`               | Built-in `cache:` keyword      |
| Services        | Service containers            | Service containers             |
| DAG support     | `needs:` keyword              | `needs:` keyword               |
| Pricing         | Free for public repos         | Free tier with limits          |

Both are fully capable. GitHub Actions has a larger community and marketplace. GitLab CI
has deeper integration with GitLab's built-in features (registry, security scanning, etc.).

---

### Q25: How do you ensure reproducibility in ML experiments?

**A:** Reproducibility requires controlling every variable:

1. **Code** -- Git commit SHA for exact code version
2. **Data** -- DVC or a data registry for exact dataset version
3. **Dependencies** -- pinned versions in `requirements.txt` (use `pip freeze`)
4. **Configuration** -- all hyperparameters logged in MLflow or config files in Git
5. **Random seeds** -- set seeds for Python, NumPy, PyTorch, etc.
6. **Hardware** -- document GPU type, CUDA version (can affect floating-point results)
7. **Environment** -- Docker image with pinned base ensures consistent OS-level deps

```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

MLflow captures most of this automatically when used correctly: parameters, metrics,
artifacts, code version (via Git integration), and the environment (conda/pip YAML).

---

### Q26: What is the difference between CMD and ENTRYPOINT in Docker?

**A:**

- **CMD** provides default arguments that can be entirely overridden at `docker run` time.
- **ENTRYPOINT** sets the fixed executable; `CMD` provides default arguments to it.

When both are used together, `ENTRYPOINT` is the command and `CMD` is appended as arguments.
When `docker run` provides arguments, they replace `CMD` but not `ENTRYPOINT`.

Use `CMD` when you want flexibility (users can run a different command). Use `ENTRYPOINT`
when the container has a fixed purpose (always runs the same binary, e.g., a model server).

---

### Q27: How do you handle database migrations in CI/CD?

**A:** Database migrations should be:

1. **Version-controlled** -- migration files in Git (Alembic for SQLAlchemy, Django migrations)
2. **Automated** -- run as a CI/CD step before deploying new code
3. **Backwards-compatible** -- new code should work with old schema during rolling deploys
4. **Tested** -- run migrations against a test database in CI

```yaml
# GitHub Actions step
- name: Run database migrations
  env:
    DATABASE_URL: ${{ secrets.STAGING_DATABASE_URL }}
  run: alembic upgrade head
```

For zero-downtime deployments, use the "expand-contract" pattern: first deploy a migration
that adds new columns/tables (expand), then deploy code that uses them, then remove old
columns in a future migration (contract).

---

### Q28: What monitoring tools would you use for a RAG application?

**A:** A RAG application needs monitoring at multiple levels:

- **Infrastructure** -- Prometheus + Grafana for CPU, memory, GPU, request counts
- **Application** -- OpenTelemetry for distributed tracing across retrieval and generation
- **LLM-specific** -- Langfuse for tracing individual LLM calls, token usage, and costs
- **Retrieval quality** -- custom metrics for retrieval precision, recall, MRR
- **Generation quality** -- automated evaluation (faithfulness, relevancy) on sampled queries
- **Cost** -- custom dashboard tracking token usage and cost by model, endpoint, and user
- **Alerting** -- PagerDuty/Slack alerts on error rate spikes, latency degradation, or cost anomalies

The key insight for RAG monitoring is that you need to track both the retrieval stage
(did we find relevant documents?) and the generation stage (did the LLM use them correctly?)
separately, because problems in either stage require different fixes.
