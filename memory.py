"""Per-session conversation memory, persisted as one JSON file per session."""
from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    messages_from_dict,
    messages_to_dict,
)


class MemoryManager:
    def __init__(
        self,
        memory_dir: str,
        session_id: str,
        window: int = 10,
        max_messages: int = 40,
    ) -> None:
        self._dir = Path(memory_dir)
        self.session_id = session_id
        self.window = window
        self.max_messages = max_messages
        self._messages: list[BaseMessage] = []
        self.load()  # resume from disk if this session already has history

    # -- where this session lives on disk ------------------------------------
    @property
    def _path(self) -> Path:
        return self._dir / f"{self.session_id}.json"

    # -- writing --------------------------------------------------------------
    def add_user_message(self, message: str) -> None:
        self._append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self._append(AIMessage(content=message))

    def _append(self, message: BaseMessage) -> None:
        self._messages.append(message)
        self._trim()
        self.save()

    def _trim(self) -> None:
        if self.max_messages and len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]

    # -- reading --------------------------------------------------------------
    @property
    def all_messages(self) -> list[BaseMessage]:
        return list(self._messages)

    def window_messages(self) -> list[BaseMessage]:
        """The most recent messages to feed the model as chat history."""
        recent = self._messages[-self.window:] if self.window else list(self._messages)
        # Chat models expect a conversation to open with the user, so drop a
        # leading AI message left over from slicing.
        while recent and not isinstance(recent[0], HumanMessage):
            recent = recent[1:]
        return list(recent)

    # -- persistence ----------------------------------------------------------
    def save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        payload = messages_to_dict(self._messages)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)  # atomic: never leaves a half-written file

    def load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            self._messages = list(messages_from_dict(payload))
            self._trim()
        except Exception as exc:
            print(f"[memory] could not load {self._path.name}: {exc}")
            self._messages = []

    def clear(self) -> None:
        self._messages = []
        if self._path.exists():
            self._path.unlink()
