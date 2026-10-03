"""Gunicorn settings for the container.

In a file rather than a CMD line so each choice can carry its reason, and so
the port can be read in Python instead of needing a shell to expand ${PORT}.
"""

import os

# Spaces expects 7860; Cloud Run and most other hosts inject PORT.
bind = f"0.0.0.0:{os.environ.get('PORT', '7860')}"

# One worker: each would load its own copy of the embedding model, roughly 1GB
# of the free tier's memory. Threads cover concurrency instead, which suits a
# workload that spends almost all its time waiting on the Groq and Pinecone
# APIs rather than on the CPU.
workers = 1
threads = 4

# Long enough for a slow reasoning model. The app's own Groq timeout is 60s, so
# this is the backstop rather than the normal limit.
timeout = 120

# Both logs to stdout, which is where Spaces, Cloud Run and docker logs read.
accesslog = "-"
errorlog = "-"
