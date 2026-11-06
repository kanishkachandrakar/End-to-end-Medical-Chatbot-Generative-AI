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

# Gunicorn switches to gthread by itself when threads > 1, which means the
# threads setting above silently depends on that behaviour. Naming the class
# makes the pairing explicit: a sync worker would ignore `threads` entirely and
# serve one request at a time.
worker_class = "gthread"

# Long enough for a slow reasoning model. The app's own Groq timeout is 60s, so
# this is the backstop rather than the normal limit.
timeout = 120

# Both logs to stdout, which is where Spaces, Cloud Run and docker logs read.
accesslog = "-"
errorlog = "-"

# Restart the worker periodically. torch and sentence-transformers hold on to
# allocator arenas across requests, and on a 16GB free tier a slow climb is
# cheaper to pre-empt than to diagnose. The jitter stops a restart landing at a
# predictable moment.
#
# A recycle is not "a brief pause", which is what this comment used to say.
# Measured inside the built image:
#
#   import torch and langchain   2.8s
#   build the embedding model   24.2s
#   first embedding              1.2s
#                              -----
#                               28.1s before the worker can answer
#
# plus up to fifteen more if read_index_size has to time out. With one worker
# there is nothing else serving meanwhile, so every recycle is roughly half a
# minute of unavailability -- the same cost DEPLOY.md quotes for a cold start,
# because it is the same work.
#
# preload_app would remove it: gunicorn would build the app once in the master
# and fork, so a replacement worker inherits a loaded model instead of building
# one. It is not enabled here because the master would then hold open Pinecone
# and Groq HTTP connections across the fork, and a connection pool shared with
# a forked child is a classic source of intermittent, hard-to-attribute
# failures. That needs trying against a real deployment, not a config comment.
#
# So the number stays high rather than the interval short: at the demo's own
# cap of 30 answers a minute, a thousand requests is at least half an hour of
# saturated traffic, and most days it is never reached.
max_requests = 1000
max_requests_jitter = 100

# Give in-flight requests time to finish on a restart, which happens whenever a
# Space secret is edited. Shorter than `timeout` on purpose: a worker being
# replaced should not hold the new one off for the full two minutes.
graceful_timeout = 30
