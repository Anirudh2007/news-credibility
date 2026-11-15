# The API image (Phase 8). The web app has its own small image: frontend/Dockerfile.
# docker-compose.yml starts both.
#
# Not baked in: models/ and data/reference/ (git-ignored, several GB). They are
# mounted at run time, so retraining needs no rebuild. Hugging Face base models
# (MiniLM encoder, cross-encoder, CLIP) download on first start into /cache/huggingface.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    NEWSCRED_DB_URL=sqlite:////db/app.db \
    NEWSCRED_FACTCHECK_CACHE=/db/factcheck.sqlite \
    NEWSCRED_WIKI_CACHE=/db/wiki.sqlite \
    NEWSCRED_NEWS_CACHE=/db/news.sqlite \
    NEWSCRED_MEMORY_PATH=/db/evidence_memory.sqlite

# libgomp1: OpenMP for LightGBM and PyTorch (OpenCV and its system libraries left with video support)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements_locked.txt ./
# 1. CPU-only PyTorch: the default Linux wheel bundles CUDA (~2.5 GB) that a CPU server never uses.
#    "torch==2.14.1" in the lock file also matches the CPU build "2.14.1+cpu".
# 2. Everything else at the versions the test suite passed with. Jupyter is left out
#    (notebooks are not served).
#    The CPU index does not mirror every pinned dependency (MarkupSafe 3.0.4 is missing), so this step
#    pins only torch/torchvision; the next step brings the rest to the locked versions from PyPI.
RUN TORCH=$(grep -i "^torch==" requirements_locked.txt | cut -d= -f3) \
    && VISION=$(grep -i "^torchvision==" requirements_locked.txt | cut -d= -f3) \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH}+cpu" "torchvision==${VISION}+cpu" \
    && grep -v -E "^jupyter" requirements.txt > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt -c requirements_locked.txt \
    && python -m spacy download en_core_web_sm

COPY src ./src
COPY api ./api
COPY tests ./tests

# Unprivileged user; the database and model-cache volumes must be writable by it.
RUN useradd --create-home app && mkdir -p /db /cache/huggingface && chown app /db /cache/huggingface
USER app

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
