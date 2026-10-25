"""Command-line chat with the RAG agent.  Run:  python main.py"""
from __future__ import annotations

import traceback

from langchain_core.messages import HumanMessage

from agent import RagAgent
from config import Providers, Settings
from guardrails import GuardrailEngine
from memory import MemoryManager
from retrieval import Retriever, WebSearchTool
from self_critique import SelfCritic

_BANNER = """
+--------------------------------------------------------------+
| GEMINI RAG AGENT (LANGCHAIN + LANGGRAPH)                     |
| Ask anything about a document in this chatbot                |
| Type 'quit' or 'exit' to end the session                     |
+--------------------------------------------------------------+
"""


class ChatApp:
    def __init__(self, session_id: str = "default") -> None:
        self.settings = Settings()
        self.providers = Providers(self.settings)

        self.guardrails = GuardrailEngine.default(self.settings.use_guardrails_ai)
        self.memory = MemoryManager(
            session_id=session_id,
            memory_dir=self.settings.memory_dir,
            window=self.settings.memory_window,
            max_messages=self.settings.max_stored_messages,
        )
        retriever = Retriever(
            self.providers.embeddings,
            index_path=self.settings.faiss_index_path,
            top_k=self.settings.top_k,
            min_score=self.settings.retrieval_min_score,
            verbose=False,
        )
        web_tool = WebSearchTool(
            enabled=self.settings.tavily_enabled,
            api_key=self.settings.tavily_api_key,
        )

        offline = self.settings.offline_mode
        use_critic = not offline and self.settings.enable_self_critique
        critic = SelfCritic(self.providers) if use_critic else None
        if offline:
            print("[main] Offline mode is enabled. Self-critique is disabled.")
        self.graph = RagAgent(
            self.providers, retriever, web_tool, critic, offline=offline
        ).build()

    def _handle_turn(self, user_input: str) -> None:
        checked = self.guardrails.check_input(user_input)
        if not checked.allowed:
            print(f"[Guardrails] Input blocked: {checked.reason}")
            return
        if checked.reason:
            print(f"[Guardrails] {checked.reason}")
        clean_input = checked.text

        state = {
            "messages": [HumanMessage(content=clean_input)],
            "chat_history": self.memory.window_messages(),
            "context": "",
            "answer": "",
            "critique_passed": True,
        }
        try:
            result = self.graph.invoke(state)
        except Exception as exc:
            print(f"[Error] Exception during graph run: {exc}")
            traceback.print_exc()
            return

        answer = result.get("answer", "")
        checked_out = self.guardrails.check_output(answer)
        final_answer = answer if checked_out.allowed else "[Guardrail] output blocked."

        print(f"\nAgent: {final_answer}")
        self.memory.add_user_message(clean_input)
        self.memory.add_ai_message(final_answer)

    def run(self) -> None:
        print(_BANNER)
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSession ended.")
                break
            if not user_input:
                print("[main] Empty input. Please try again.")
                continue
            if user_input.lower() in {"quit", "exit"}:
                print("\n[main] Session ended.")
                break
            self._handle_turn(user_input)


def main() -> None:
    ChatApp().run()


if __name__ == "__main__":
    main()
