"""Build the FAISS index from the documents in a folder.

Usage:  python ingest.py ./docs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Providers, Settings


class DocumentIngestor:
    """Loads documents, splits them, embeds them, and saves a FAISS index."""

    def __init__(self, settings: Settings | None = None, providers: Providers | None = None):
        self.settings = settings or Settings()
        self.providers = providers or Providers(self.settings)

    def _load(self, folder: str) -> list:
        docs: list = []
        docs.extend(
            DirectoryLoader(folder, glob="**/*.pdf", loader_cls=PyPDFLoader).load()
        )
        for pattern in ("**/*.txt", "**/*.md"):
            docs.extend(
                DirectoryLoader(
                    folder,
                    glob=pattern,
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"},
                ).load()
            )
        return docs

    def _split(self, docs: list) -> list:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        return splitter.split_documents(docs)

    def ingest(self, folder: str) -> None:
        if not Path(folder).is_dir():
            print(f"[ingest] Error: {folder} is not a valid directory.")
            sys.exit(1)

        print(f"[ingest] loading documents from {folder} ...")
        docs = self._load(folder)
        if not docs:
            print("[ingest] no .pdf, .txt or .md files found. Add some to ./docs")
            sys.exit(1)

        chunks = self._split(docs)
        print(f"[ingest] split {len(docs)} documents into {len(chunks)} chunks")

        print("[ingest] embedding chunks and building the FAISS index ...")
        # normalize_L2 must match retrieval.py (cosine-like scores).
        store = FAISS.from_documents(chunks, self.providers.embeddings, normalize_L2=True)
        store.save_local(self.settings.faiss_index_path)
        print(f"[ingest] FAISS index saved to {self.settings.faiss_index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into FAISS")
    parser.add_argument(
        "folder",
        nargs="?",
        default="./docs",
        help="folder containing documents to ingest (default: ./docs)",
    )
    args = parser.parse_args()
    DocumentIngestor().ingest(args.folder)


if __name__ == "__main__":
    main()
