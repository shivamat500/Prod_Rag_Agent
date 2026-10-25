"""The LangGraph RAG agent: retrieve -> (web fallback) -> generate -> critique."""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from config import Providers
from retrieval import Retriever, WebSearchTool
from self_critique import SelfCritic


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    chat_history: list[BaseMessage]
    context: str
    answer: str
    critique_passed: bool


_SYSTEM = (
    "You are a helpful assistant. The context below was extracted from documents "
    "that have already been uploaded and indexed; treat it as the main document. "
    "Answer the question using only the provided context. If the answer is "
    "present in the context, state it clearly. If it is genuinely not in the "
    "context, say \"I could not find that in the provided documents.\" Never say "
    "that no document was provided.\n\n"
    "Context:\n{context}"
)

NOTHING_FOUND = (
    "I could not find anything about that in the indexed documents. "
    "Try rephrasing, or ask about the Reagan biography or the T001 clinical "
    "trial report."
)

_OFFLINE_PREAMBLE = (
    "Answering directly from the indexed documents (no language model is "
    "running, so these are the matching passages verbatim):"
)


class RagAgent:
    def __init__(
        self,
        providers: Providers,
        retriever: Retriever,
        web_tool: WebSearchTool,
        critic: SelfCritic | None = None,
        offline: bool = False,
    ) -> None:
        self.providers = providers
        self.retriever = retriever
        self.critic = critic
        self.web_tool = web_tool
        self.offline = offline

        if offline:
            # No LLM chain is built, so offline mode never touches Gemini.
            self._gen_chain = None
            return

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        self._gen_chain = prompt | providers.llm | StrOutputParser()

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _latest_human(state: AgentState) -> str:
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                return str(msg.content)
        return ""

    @staticmethod
    def _extractive_answer(context: str) -> str:
        """Offline answer: hand back the retrieved passages themselves.

        Without an LLM there is nothing to synthesise prose from, so the honest
        thing is to show the source text and say that is what we are doing,
        rather than dress it up as a generated answer.
        """
        context = context.strip()
        if not context:
            return NOTHING_FOUND
        return f"{_OFFLINE_PREAMBLE}\n\n{context}"

    # -- graph nodes ---------------------------------------------------------
    def retrieve_node(self, state: AgentState) -> dict:
        question = self._latest_human(state)
        return {"context": self.retriever.retrieve(question) if question else ""}

    def tool_node(self, state: AgentState) -> dict:
        question = self._latest_human(state)
        if not self.web_tool.available:
            print("[agent] no web search tool is configured - skipping the fallback")
            return {"context": state.get("context", "")}
        print("[agent] no local content - falling back to web search")
        web = self.web_tool.search(question)
        combined = (web + "\n\n" + state.get("context", "")).strip()
        return {"context": combined}

    def generate_node(self, state: AgentState) -> dict:
        context = state.get("context", "")
        if self.offline:
            return {"answer": self._extractive_answer(context)}
        if not context.strip():
            # Nothing to ground an answer on: skip the LLM call entirely.
            return {"answer": NOTHING_FOUND}
        answer = self._gen_chain.invoke(
            {
                "context": context,
                "chat_history": state.get("chat_history", []),
                "question": self._latest_human(state),
            },
            config={"callbacks": self.providers.callbacks()},
        )
        return {"answer": answer}

    def critique_node(self, state: AgentState) -> dict:
        answer = state.get("answer", "")
        context = state.get("context", "")
        if not context.strip() or answer == NOTHING_FOUND:
            return {"answer": answer, "critique_passed": True}
        final = self.critic.critique(self._latest_human(state), answer, context)
        return {"answer": final, "critique_passed": final == answer}

    def output_node(self, state: AgentState) -> dict:
        return {"messages": [AIMessage(content=state["answer"])]}

    # -- routing -------------------------------------------------------------
    def _route_after_retrieve(self, state: AgentState) -> str:
        return "generate_node" if state.get("context", "").strip() else "tool_node"

    # -- graph assembly ------------------------------------------------------
    def build(self):
        g = StateGraph(AgentState)
        g.add_node("retrieve_node", self.retrieve_node)
        g.add_node("tool_node", self.tool_node)
        g.add_node("generate_node", self.generate_node)
        g.add_node("output_node", self.output_node)

        g.add_edge(START, "retrieve_node")
        g.add_conditional_edges(
            "retrieve_node",
            self._route_after_retrieve,
            {"tool_node": "tool_node", "generate_node": "generate_node"},
        )
        g.add_edge("tool_node", "generate_node")

        if self.critic is None:
            g.add_edge("generate_node", "output_node")
        else:
            g.add_node("critique_node", self.critique_node)
            g.add_edge("generate_node", "critique_node")
            g.add_edge("critique_node", "output_node")

        g.add_edge("output_node", END)
        return g.compile()
