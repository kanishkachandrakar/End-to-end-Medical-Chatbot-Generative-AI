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

```bash
pip install -r requirements.txt
python store_index.py --recreate
```

It asks you to type the index name first, since it is deleting 5,860 vectors;
add `--yes` to skip that if you are scripting it.

`--recreate` deletes the index before rebuilding, which is what clears the
duplicates — they were written under random ids, so a plain re-run cannot
overwrite them. Expect roughly 5,900 unique chunks.

Check it first if you like: `--dry-run` reports what a rebuild would produce
without touching Pinecone, and `--limit 50` exercises the whole pipeline in
seconds.

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

Then open the Space and ask something like *What is hypertension?*.

`GET /healthz` answers without spending a Groq call, and says which of three
states the app is in:

| Response | Status | Meaning |
|---|---|---|
| `ok <rev> vectors=5860` | 200 | Working. `<rev>` is the commit the image was built from, so you can confirm a deploy took effect. |
| `degraded <rev> index-empty` | 503 | Running, but the index has no vectors — every answer will be ungrounded. Rebuild it. |
| `degraded <rev> chain-unavailable` | 503 | Running, but the retrieval chain could not be built at startup: a bad key, or Pinecone unreachable. The reason is in the runtime log. |
| no response at all | — | The container is not serving. Check the build log, then the runtime log. |

The third row is why the app starts even when its dependencies are down: it
would otherwise crash, restart, crash again, and tell you nothing. A transient
outage at boot also recovers on the next restart rather than needing a manual
one.

## Optional settings

Everything in `.env.example` below the two keys can also be set as a Space
*variable* (not a secret — variables are public, which is correct for these):

| Variable | Default | Why you might change it |
|---|---|---|
| `RATE_LIMIT_PER_MINUTE` | `30` | Questions answered per minute, across all visitors. `0` removes the cap. |
| `MAX_QUESTION_CHARS` | `500` | Longest accepted question. The input box mirrors this. |
| `GROQ_TIMEOUT` | `60` | Seconds before a stalled Groq call is abandoned. |
| `TOP_K` | `3` | Passages retrieved per question. |
| `LOG_LEVEL` | `INFO` | Set to `WARNING` to quieten the log. |

The rate limit is deliberately global rather than per visitor: behind the
Spaces proxy every request arrives from the same address, so a per-IP limit
would put all visitors in one bucket anyway, and keying on a forwarded header
would be trivially spoofed. One busy visitor can therefore make others wait.

## Things to expect

- **Cold starts.** A free Space idles after about 48 hours untouched and
  restarts on the next visit, which takes 30–60 seconds. The model weights are
  baked into the image, so it is the container start you are waiting on, not a
  download.
- **Editing a secret restarts the Space.** Gunicorn drains in-flight requests
  when that happens.
- **Groq rate limits** on the free tier surface as "I am being rate limited
  right now" with a 429; the traceback is in the runtime log.
- **The demo's own limit** is separate, and says "This demo answers a limited
  number of questions a minute". That one is `RATE_LIMIT_PER_MINUTE`, not Groq.
- **Request logs appear in the runtime log**, one line per request, so you can
  tell a question that failed from one that never arrived.

## Why the dependencies are pinned

`requirements.txt` carries version bounds rather than bare package names, and
the comments say which breakage each one is for. Two are worth knowing about,
because both produced an image that built cleanly and then would not start:

- **An upper bound missing** let `langchain` 1.0 move `langchain.chains` out
  from under `app.py`.
- **A lower bound missing** was worse. Resolving `pinecone[grpc]` constrains
  the `pinecone` version, and rather than report a conflict, pip walked
  `langchain-pinecone` backwards until something fit — settling on 0.0.1, from
  2023, with no `PineconeVectorStore` in it.

Neither is visible from a working checkout, because an existing virtualenv
already has the right versions. They only appear in a clean resolve, which is
what the image build does. If you loosen a bound, let CI build the image before
believing it.

## Other hosts

The `Dockerfile` is not Spaces-specific — it reads `$PORT` and falls back to
7860, so it also runs on Cloud Run, Railway, Fly or any VM. A 512MB host needs
the local embedding model replaced with a hosted embedding API first; keep to
384 dimensions or the existing index becomes unusable.
