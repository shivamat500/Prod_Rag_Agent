"""Self-critique: a second LLM pass that scores the answer against the context
and, if the score is low, replaces it with an improved answer."""
from __future__ import annotations

import json
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Literal braces are doubled because this string is a prompt template.
_SYSTEM = """You are a strict technical reviewer.
You are given CONTEXT extracted from documents, a QUESTION and an ANSWER.
Evaluate the ANSWER only against the provided CONTEXT. Do not introduce outside
information. Check that every claim in the answer is supported by the context
and that the answer actually addresses the question.

Respond with ONLY a valid JSON object (no markdown fences) in this shape:
{{"score": <integer 1-10>, "issues": "<short description>", "improved_answer": "<a corrected answer grounded in the context, or null if the answer is already good>"}}"""


class SelfCritic:
    ACCEPT_THRESHOLD = 7

    def __init__(self, providers) -> None:
        self.providers = providers
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM),
                ("human", "CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER: {answer}"),
            ]
        )
        self._chain = prompt | providers.llm | StrOutputParser()

    @staticmethod
    def _parse(raw: str) -> dict:
        """Pull the JSON object out of the model reply, tolerating ``` fences."""
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("no JSON object in critique reply")
        return json.loads(cleaned[start : end + 1])

    def critique(self, question: str, answer: str, context: str) -> str:
        try:
            raw = self._chain.invoke(
                {"context": context, "question": question, "answer": answer},
                config={"callbacks": self.providers.callbacks()},
            )
            result = self._parse(raw)
            score = int(result.get("score", 10))
            improved = result.get("improved_answer")
        except Exception as exc:
            # The critique is a bonus: if it fails, the original answer stands.
            print(f"[self-critique] could not use critique, keeping answer: {exc}")
            return answer

        if score < self.ACCEPT_THRESHOLD and improved and str(improved).strip().lower() != "null":
            print(f"[self-critique] score {score}/10 - replacing with improved answer")
            return str(improved)
        print(f"[self-critique] score {score}/10 - accepting")
        return answer
