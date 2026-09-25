# Runs on anything that takes a container: Hugging Face Spaces, Cloud Run,
# Railway, Fly, or a plain VM. The port is read from $PORT at start-up and
# defaults to 7860, which is what Spaces expects.
FROM python:3.10-slim

# Spaces runs the container as uid 1000, so create that user and give it a
# home the model cache can be written to.
RUN useradd --create-home --uid 1000 user

ENV HOME=/home/user \
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
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

USER user

# Bake the embedding weights into the image. Without this the first request
# after every cold start waits on a ~90MB download from HuggingFace.
RUN python -c "from src.helper import download_hugging_face_embeddings; download_hugging_face_embeddings()"

COPY --chown=user . .

EXPOSE 7860

# One worker: each would load its own copy of the embedding model. Threads
# handle concurrency instead, since requests are spent waiting on the Groq
# and Pinecone APIs. The long timeout covers slow LLM responses.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 120 app:app"]
