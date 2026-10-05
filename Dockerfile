# Runs on anything that takes a container: Hugging Face Spaces, Cloud Run,
# Railway, Fly, or a plain VM. The port is read from $PORT at start-up and
# defaults to 7860, which is what Spaces expects.
FROM python:3.10-slim

# Spaces runs the container as uid 1000, so create that user and give it a
# home the model cache can be written to.
RUN useradd --create-home --uid 1000 user

# Which commit this image was built from. Passed by the deploy script; the
# default makes a hand-built image say so rather than claim a revision.
ARG APP_REVISION=unknown

ENV APP_REVISION=${APP_REVISION} \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies first so edits to the app don't invalidate the pip layer.
COPY --chown=user requirements.txt setup.py ./
COPY --chown=user src ./src

# Install torch from the CPU index first. The default PyPI wheel bundles the
# CUDA runtime (~2.5GB) which is dead weight on a CPU-only host; pip then sees
# the requirement as already satisfied when it reads requirements.txt.
#
# PyPI has to stay available as a fallback. The PyTorch index serves the torch
# wheel but not every pure-Python dependency of it: typing-extensions is there
# only as an sdist, and building that needs flit_core, which the index does not
# carry at all -- so --index-url alone fails to resolve. Keeping the PyTorch
# index first still selects the CPU build, because PEP 440 sorts the local
# version 2.x+cpu above a plain 2.x from PyPI.
RUN pip install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

RUN pip install --no-cache-dir -r requirements.txt

USER user

# Bake the embedding weights into the image. Without this the first request
# after every cold start waits on a ~90MB download from HuggingFace.
RUN python -c "from src.helper import download_hugging_face_embeddings; download_hugging_face_embeddings()"

COPY --chown=user . .

EXPOSE 7860

# Generous start period: the first boot loads the embedding model before the
# port opens.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT','7860')}/healthz\").read()"

# Settings live in gunicorn.conf.py, which also resolves $PORT -- so this is
# the exec form with no shell in between, and gunicorn is PID 1.
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
