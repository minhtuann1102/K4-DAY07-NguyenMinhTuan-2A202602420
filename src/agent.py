from __future__ import annotations

from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with numbered chunks [1], [2], ... and sources.
        3. Enforce anti-hallucination and source traceability constraints.
        4. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        """
        Answers a user query based strictly on retrieved knowledge base context.
        """
        if self.store.get_collection_size() == 0:
            return "Không tìm thấy thông tin do cơ sở tri thức hiện đang trống."

        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy tài liệu phù hợp trong cơ sở tri thức để trả lời câu hỏi."

        context_parts = []
        for i, r in enumerate(results, 1):
            source = (
                r.get("metadata", {}).get("source")
                or r.get("metadata", {}).get("doc_id")
                or r.get("id", f"chunk_{i}")
            )
            content = r.get("content", "").strip()
            context_parts.append(f"[{i}] (Nguồn: {source})\n{content}")

        context_str = "\n\n".join(context_parts)
        prompt = (
            f"Ngữ cảnh tham chiếu:\n{context_str}\n\n"
            f"Yêu cầu: Chỉ sử dụng thông tin từ ngữ cảnh trên để trả lời. "
            f"Hãy ghi rõ nguồn tham chiếu [1], [2] tương ứng. "
            f"Nếu thông tin không xuất hiện trong ngữ cảnh, hãy nói rõ là không tìm thấy.\n\n"
            f"Câu hỏi: {question}\n"
            f"Trả lời:"
        )
        return self.llm_fn(prompt)
