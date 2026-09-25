# Deploying to Hugging Face Spaces

Spaces is used here because it is the only free tier with enough memory for
this app: the embedding model runs locally inside the container, which needs
roughly 1GB of RAM, and free tiers elsewhere cap at 512MB. A free Space gets
2 vCPU and 16GB.

## Before you deploy

**1. Rotate both API keys.** The Groq and Pinecone keys used during
development were committed to this repository's history and are public.
Generate new ones in the [Groq console](https://console.groq.com/keys) and the
[Pinecone console](https://app.pinecone.io), and put the new values in your
local `.env`. Deploying with the old keys publishes working credentials.

**2. Rebuild the index once.** The chunks were upserted several times with
random ids, so the index holds six copies of everything — and because
duplicates embed to identical vectors, a top-3 search returns the same chunk
three times. Delete the index and rebuild it now that ids are derived from the
chunk text:

```python
from pinecone import Pinecone
Pinecone(api_key="<your new key>").delete_index("medicalbot")
```

```bash
pip install -r requirements.txt
python store_index.py
```

Expect roughly 5,900 unique chunks. `store_index.py` recreates the index and
is safe to re-run.

## Create the Space

1. Go to <https://huggingface.co/new-space>.
2. Give it a name, choose **Docker** as the SDK and **Blank** as the template.
3. Leave the hardware on the free **CPU basic** tier.

The Space's configuration comes from the YAML block at the top of
[README.md](README.md) — `sdk: docker` and `app_port: 7860` in particular, which
must match the port the container binds.

## Add the secrets

In the Space, open **Settings → Variables and secrets** and add two *secrets*
(not variables, which are public):

| Name | Value |
|---|---|
| `PINECONE_API_KEY` | your rotated Pinecone key |
| `GROQ_API_KEY` | your rotated Groq key |

Spaces exposes secrets as environment variables, which is where `app.py` reads
them from. `.env` is never deployed.

## Deploy

```bash
./scripts/deploy_space.sh <your-username>/<your-space-name>
```

The script copies only the runtime files into a clone of the Space and pushes.
It leaves `Data/Medical_book.pdf` behind on purpose: the container never reads
it, and at 15MB it would need Git LFS on the Hub.

You will be asked for your Hugging Face credentials — use an
[access token](https://huggingface.co/settings/tokens) with write scope as the
password.

## Verify

The first build takes a few minutes; watch it under the **Logs → Build** tab.
When it finishes, check the runtime log for:

```
index 'medicalbot' holds 5900 vectors
```

If it says the index is empty, the app will answer with no retrieved context —
go back to *Rebuild the index*.

Then open the Space and ask something like *What is hypertension?*. There is
also `GET /healthz`, which returns `ok` without spending a Groq call.

## Things to expect

- **Cold starts.** A free Space idles after about 48 hours untouched and
  restarts on the next visit, which takes 30–60 seconds. The model weights are
  baked into the image, so it is the container start you are waiting on, not a
  download.
- **Editing a secret restarts the Space.** Gunicorn drains in-flight requests
  when that happens.
- **Groq rate limits** on the free tier surface as a "could not answer that
  right now" reply; the traceback is in the runtime log.

## Other hosts

The `Dockerfile` is not Spaces-specific — it reads `$PORT` and falls back to
7860, so it also runs on Cloud Run, Railway, Fly or any VM. A 512MB host needs
the local embedding model replaced with a hosted embedding API first; keep to
384 dimensions or the existing index becomes unusable.
