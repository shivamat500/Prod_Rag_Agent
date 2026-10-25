# Gemini RAG Agent

A document Q&A chatbot: LangGraph agent, local FAISS retrieval (fastembed
embeddings), Google Gemini for answers, optional Tavily web fallback, guardrails,
per-visitor memory, rate limiting, and a small web UI. Deployed on Render.

## Run locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env            # then put your GEMINI_API_KEY in .env
# put your PDFs / .txt / .md files in ./docs  (delete docs/SAMPLE_replace_me.txt)
python ingest.py ./docs         # builds ./faiss_index

pytest                          # unit tests, no API key needed
python main.py                  # CLI chat
uvicorn app:app --reload        # web UI at http://127.0.0.1:8000
```

No API key yet? Set `OFFLINE_MODE=true` to try retrieval without any Gemini call.

## Deploy: GitHub, then Render

1. **Add your documents** to `docs/` and delete `docs/SAMPLE_replace_me.txt`.
   The index is built from this folder during every Render build, so commit the
   documents (keep them small enough for GitHub, and only if they may be public
   in your repo).
2. **Push to GitHub**
   ```bash
   git init && git add . && git commit -m "RAG agent"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
   `.env` is git-ignored, so your key is not uploaded.
3. **On render.com**: *New +* then *Blueprint*, choose the repo. Render reads
   `render.yaml`. When prompted, paste `GEMINI_API_KEY` (and optionally
   `TAVILY_API_KEY`). Deploy.
   - Manual alternative: *New +* then *Web Service*, Runtime *Python 3*,
     Build `pip install -r requirements.txt && python ingest.py ./docs`,
     Start `uvicorn app:app --host 0.0.0.0 --port $PORT`,
     Health check path `/healthz`, plus the env vars above and `PYTHON_VERSION=3.12.8`.
4. Open the `onrender.com` URL. The first build takes several minutes (it
   downloads the embedding model); the free plan sleeps when idle, so the first
   request after a pause is slow.

To change the documents later: edit `docs/`, commit, push. Render rebuilds the index.

## Notes

- **Free-tier quotas.** Each question makes 2 Gemini calls (answer + self-critique).
  Set `ENABLE_SELF_CRITIQUE=false` to make it 1.
- **Memory.** Chat history is stored per visitor in `memory_store/`, which is
  wiped on every Render deploy/restart (free plan has no persistent disk).
- **Retrieval threshold.** If good questions return "could not find", lower
  `RETRIEVAL_MIN_SCORE` (default 0.35); if unrelated chunks come back, raise it.
- **Locking versions.** Once `pip install -r requirements.txt` works for you,
  run `pip freeze > requirements.lock` and keep it for reproducible rebuilds.
