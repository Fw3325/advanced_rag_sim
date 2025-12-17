import re
import ast
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from retrieval import retrieval
import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class Citation(BaseModel):
    document: str = Field(..., description="The document name including extension")
    page: int = Field(..., description="The page number from the PDF")

class LLMResponse(BaseModel):
    answer: str = Field(..., description="LLM answer text, no JSON or citations inside")
    citations: List[int] = Field(..., description="chunk ids cited in the answer")

def llm_model():
    llm = ChatOpenAI(
        temperature=0.2,
        model='gpt-4o',                  
        openai_api_key=OPENAI_API_KEY,
    )
    return llm

def generate_response(llm, vectorstore, chunks, query, kg_retriever=None, enable_kg=True, fast_mode=False):
    """
    Run the full RAG pipeline for a query and return an LLMResponse.
    Also records timing information for each major component:
    - retrieval_time: advanced retrieval (multi-query + rerank + KG + MMR)
    - generation_time: LLM generation time
    - total_time: end-to-end time inside this function
    The timing dict is attached to the returned object as `response._timing`.
    """
    import time
    total_start = time.time()

    # Reduce k in fast mode to send fewer chunks to LLM
    retrieval_k = 5 if fast_mode else 8
    mmr_top_n = 5 if fast_mode else 8

    # ── Retrieval timing ──────────────────────────────────────────────────────
    t_retrieval_start = time.time()
    retrieved_chunks = retrieval.advanced_multi_query_similarity_search(
                        llm = llm,
                        vectorstore=vectorstore,
                        chunks=chunks,
                        user_query=query,
                        k=retrieval_k,
                        window_expansion=False,  # no need for context expansion with current chunking strategy
                        rrf_top_n=10,
                        mode="hybrid",  # mode: 'dense' | 'bm25' | 'hybrid'
                        enable_mmr=not fast_mode,  # Skip MMR in fast mode
                        mmr_top_n=mmr_top_n,
                        kg_retriever=kg_retriever,
                        enable_kg=enable_kg,
                        fast_mode=fast_mode,
                        use_optimized=True
                        )
    retrieval_time = time.time() - t_retrieval_start
    
    # assign chunk_id for tracking
    chunks_with_ids = []
    for idx, chunk in enumerate(retrieved_chunks):
        chunk.metadata["chunk_id"] = idx + 1  # 1-based
        chunks_with_ids.append(chunk)
    
    # build system prompt
    system_instruction = SystemMessage(content="""
        You are a Retrieval-Augmented Generation assistant.
        Use ONLY the provided text chunks to answer the question.
        If you don't find any relevant information in the context, do not answer the user's query.
        If there is even a small bit of relevancy between the user's query and the context, 
        please provide a detailed answer based on the instructions above.

        You MUST reply strictly in JSON, following this schema:

        {
        "answer": "detailed answer here",
        "citations": [chunk_id, chunk_id]
        }

        Rules:
        - DO NOT hallucinate chunk ids. Only use from provided chunks.
        - Only quote text that exists in chunks provided.
        - If unsure, return: {"answer": "No relevant information found.", "citations": []}
        """)

    # context messages (chunk text only, metadata excluded)
    context_messages = [
        HumanMessage(content=f"[chunk_id={chunk.metadata['chunk_id']}] {chunk.page_content}")
        for chunk in chunks_with_ids
    ]
  
    # user query
    user_message = HumanMessage(content=query)

    # ── LLM generation timing ────────────────────────────────────────────────
    t_gen_start = time.time()
    raw_response = llm.invoke([system_instruction] + context_messages + [user_message]).content
    generation_time = time.time() - t_gen_start

    # parse JSON using pydantic
    response = parse_llm_response(raw_response)

    # map chunk_ids back to metadata
    id_to_chunk = {c.metadata["chunk_id"]: c for c in retrieved_chunks}

    mapped_citations = []
    for cited_id in response.citations:
        chunk = id_to_chunk.get(cited_id)
        if chunk:
            page = chunk.metadata.get("page")
            if page is None:
                page = chunk.metadata.get("page_number")
            if page is None:
                page = "unknown"

            item = {
                "document": chunk.metadata["source"],
                "page": page,
                "chunk_id": cited_id,
                "chunk_content": chunk.page_content
            }
            mapped_citations.append(item)
        #     mapped_citations.append(
        # {
        #     "document": chunk.metadata["source"],
        #     "page": chunk.metadata["page"],
        #     "chunk_id": cited_id,
        #     "chunk_content": chunk.page_content
        # }
    # )
    response.citations = mapped_citations

    # Attach timing info for downstream consumers (Streamlit, logging, etc.)
    total_time = time.time() - total_start
    timing = {
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "total_time": total_time,
        "fast_mode": fast_mode,
        "enable_kg": enable_kg,
        "retrieval_k": retrieval_k,
        "mmr_top_n": mmr_top_n,
    }
    # Attach as a non-model attribute
    setattr(response, "_timing", timing)

    # Optional console log
    print(
        f"⏱️  RAG timings — retrieval: {retrieval_time:.2f}s, "
        f"generation: {generation_time:.2f}s, total: {total_time:.2f}s"
    )

    return response

def parse_llm_response(raw_response: str) -> LLMResponse:
    """
    Parse the LLM response into the LLMResponse schema.
    Be robust to minor formatting issues like Markdown fences or extra text.
    """
    # Already parsed
    if isinstance(raw_response, LLMResponse):
        return raw_response

    # Ensure string
    if not isinstance(raw_response, str):
        raw_response = str(raw_response)

    text = raw_response.strip()

    # Remove common markdown fences, e.g. ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        # Try to strip the first and last code fences
        # e.g. ```json\n{...}\n```  -> {...}
        parts = text.split("```")
        # parts: ["", "json\n{...}\n", ""]
        if len(parts) >= 3:
            # Take the middle part and drop optional 'json' prefix
            inner = parts[1]
            if inner.lower().lstrip().startswith("json"):
                inner = inner[inner.lower().find("json") + len("json") :]
            text = inner.strip()

    # Helper: try to load JSON with optional { ... } extraction
    def _try_parse(candidate: str) -> LLMResponse:
        parsed = json.loads(candidate)
        return LLMResponse(**parsed)

    # First attempt: direct parse
    try:
        return _try_parse(text)
    except Exception:
        pass

    # Second attempt: extract the first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return _try_parse(candidate)
        except Exception:
            pass

    # Fallback: log and return safe default instead of crashing the app
    print("LLM output format error, returning fallback. Raw response:")
    print(raw_response)
    return LLMResponse(answer=text, citations=[])
