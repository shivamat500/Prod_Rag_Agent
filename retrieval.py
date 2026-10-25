"""Retrieval: the local FAISS vector store, and the optional Tavily web fallback."""
from __future__ import annotations

import os

from langchain_community.vectorstores import FAISS


class Retriever:
    """Loads the FAISS index once and returns the matching passages as text."""

    def __init__(
        self,
        embeddings,
        index_path: str = "./faiss_index",
        top_k: int = 4,
        min_score: float = 0.0,
        verbose: bool = False,
    ) -> None:
        self.embeddings = embeddings
        self.index_path = index_path
        self.top_k = top_k
        self.min_score = min_score
        self.verbose = verbose
        self._store = self._load()

    def _load(self) -> FAISS:
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                "Run `python ingest.py ./docs` first."
            )
        # normalize_L2 must match ingest.py: it makes distances behave like
        # cosine similarity, which is what retrieval_min_score is tuned for.
        return FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True,  # the index is our own file
            normalize_L2=True,
        )

    def retrieve_docs(self, query: str) -> list:
        if not query:
            return []
        scored = self._store.similarity_search_with_relevance_scores(query, k=self.top_k)
        if self.verbose:
            for doc, score in scored:
                print(f"[retrieval] score={score:.3f} {doc.page_content[:60]!r}")
        return [doc for doc, score in scored if score >= self.min_score]

    def retrieve(self, query: str) -> str:
        docs = self.retrieve_docs(query)
        return "\n\n".join(doc.page_content for doc in docs)


class WebSearchTool:
    """Tavily web search, used only when the local documents have no answer."""

    def __init__(self, enabled: bool = False, api_key: str | None = None, max_results: int = 3):
        self._client = None
        self.max_results = max_results
        if enabled and api_key:
            try:
                from tavily import TavilyClient

                self._client = TavilyClient(api_key=api_key)
            except Exception as exc:
                print(f"[retrieval] Tavily unavailable: {exc}")

    @property
    def available(self) -> bool:
        return self._client is not None

    def search(self, query: str) -> str:
        if not self.available or not query:
            return ""
        try:
            response = self._client.search(query, max_results=self.max_results)
        except Exception as exc:
            print(f"[retrieval] web search failed: {exc}")
            return ""
        return "\n\n".join(
            f"[web] {r.get('title', '')}: {r.get('content', '')}"
            for r in response.get("results", [])
        )
