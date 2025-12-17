"""
Optimized retrieval pipeline with parallel processing, batch operations, and caching.
"""
import concurrent.futures
import time
from typing import List, Optional
from .reranker import BGEReranker
from .query_rewriter import rewrite_query
from .query_cache import get_cached_rewrite, cache_rewrite
from .bm25_retriever import BM25Retriever
from .rrf import rrf_fuse
from .mmr import mmr_select

reranker = BGEReranker()

def _parallel_dense_search(vectorstore, queries, k):
    """Parallel dense vector searches."""
    results = []
    for q in queries:
        docs = [doc for doc, _ in vectorstore.similarity_search_with_score(q, k)]
        results.append(docs)
    return results

def _parallel_bm25_search(bm25, queries, k):
    """Parallel BM25 searches."""
    results = []
    for q in queries:
        docs = bm25.search(q, k)
        results.append(docs)
    return results

def advanced_multi_query_similarity_search_optimized(
    llm, vectorstore, chunks, user_query, 
    k=8, window_expansion=False, rrf_top_n=10, mode='hybrid',
    enable_mmr=True, mmr_top_n=8, kg_retriever=None, enable_kg=True,
    fast_mode=False, max_workers=4
):
    """
    Optimized multi-query RAG with parallel processing and batch operations.
    
    Optimizations:
    1. Parallel retrieval (dense + BM25 + KG)
    2. Batch reranking (single pass instead of per sub-query)
    3. Cached embeddings for MMR
    4. Fast mode: skip reranking for speed
    5. Reduced sub-queries in fast mode
    """
    # ① Query rewriting (skip entirely in fast mode, use cache otherwise)
    start_time = time.time()
    if fast_mode:
        # Fast mode: NO query rewriting - use original query only
        sub_queries = [user_query]
        print(f"⚡ Fast mode: skipping query rewrite")
    else:
        # Check cache first
        cached = get_cached_rewrite(user_query)
        if cached:
            sub_queries = cached
            print(f"💾 Using cached query rewrites ({len(sub_queries)} sub-queries)")
        else:
            sub_queries = rewrite_query(llm, user_query)
            cache_rewrite(user_query, sub_queries)  # Cache for next time
            print(f"🧠 Generated {len(sub_queries)} sub-queries (cached for future)")
    
    query_time = time.time() - start_time
    if query_time > 0.1:
        print(f"⏱️  Query rewrite time: {query_time:.2f}s")
    
    bm25 = BM25Retriever(chunks)
    
    # ② Parallel retrieval: dense + BM25 + KG (if enabled)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        # Dense searches
        if mode in ["dense", "hybrid"]:
            futures['dense'] = executor.submit(_parallel_dense_search, vectorstore, sub_queries, k)
        
        # BM25 searches
        if mode in ["bm25", "hybrid"]:
            futures['bm25'] = executor.submit(_parallel_bm25_search, bm25, sub_queries, k)
        
        # KG search (single query, not per sub-query)
        if enable_kg and kg_retriever is not None:
            futures['kg'] = executor.submit(kg_retriever.search, user_query, k)
        
        # Wait for all retrievals
        dense_results_list = futures.get('dense', concurrent.futures.Future()).result() if 'dense' in futures else []
        bm25_results_list = futures.get('bm25', concurrent.futures.Future()).result() if 'bm25' in futures else []
        kg_results = futures.get('kg', concurrent.futures.Future()).result() if 'kg' in futures else []
    
    # ③ Merge results per sub-query using RRF
    all_candidate_chunks = []
    for i, q in enumerate(sub_queries):
        if mode == "dense":
            merged = dense_results_list[i] if i < len(dense_results_list) else []
        elif mode == "bm25":
            merged = bm25_results_list[i] if i < len(bm25_results_list) else []
        elif mode == "hybrid":
            dense_q = dense_results_list[i] if i < len(dense_results_list) else []
            bm25_q = bm25_results_list[i] if i < len(bm25_results_list) else []
            merged = rrf_fuse([dense_q, bm25_q])[:rrf_top_n] if dense_q or bm25_q else []
        else:
            merged = []
        
        if merged:
            all_candidate_chunks.extend(merged)
    
    # ④ Add KG results
    if enable_kg and kg_results:
        all_candidate_chunks.extend(kg_results)
        print(f"🔗 KG retrieved {len(kg_results)} chunks")
    
    # ⑤ Deduplicate before reranking (reduces reranker workload)
    candidate_chunks = dedup_docs(all_candidate_chunks)
    
    # ⑥ Batch reranking (single pass instead of per sub-query) - SKIP in fast mode
    if not fast_mode and len(candidate_chunks) > mmr_top_n:
        # Rerank all candidates at once with original query
        rerank_top_n = min(rrf_top_n * 2, len(candidate_chunks))
        reranked = reranker.rerank(user_query, candidate_chunks, top_n=rerank_top_n)
        candidate_chunks = reranked
        print(f"⚡ Batch reranked {len(reranked)} chunks")
    else:
        print(f"⚡ Fast mode: skipping reranking")
    
    # ⑦ MMR diversity (SKIP in fast mode to save embedding time)
    docs = candidate_chunks
    if not fast_mode and enable_mmr and len(docs) > mmr_top_n:
        # Try to reuse embeddings from vectorstore if available
        try:
            mmr_start = time.time()
            # Batch embed documents for MMR
            doc_texts = [d.page_content for d in docs]
            doc_embeddings = vectorstore.embedding_function.embed_documents(doc_texts)
            query_vec = vectorstore.embedding_function.embed_query(user_query)
            docs = mmr_select(query_vec, doc_embeddings, docs, top_k=mmr_top_n)
            mmr_time = time.time() - mmr_start
            print(f"🎯 MMR selected {len(docs)} diverse chunks ({mmr_time:.2f}s)")
        except Exception as e:
            print(f"⚠️  MMR failed, using top chunks: {e}")
            docs = docs[:mmr_top_n]
    elif fast_mode:
        # Fast mode: just take top N, no MMR
        docs = docs[:mmr_top_n]
        print(f"⚡ Fast mode: skipping MMR, using top {len(docs)} chunks")
    
    # ⑧ Window expansion (if enabled)
    if window_expansion:
        window_size = 1
        final_results, seen = [], set()
        index_map = {(c.metadata["source"], c.metadata["chunk_number"]): i for i, c in enumerate(chunks)}
        
        for ck in docs:
            idx = index_map.get((ck.metadata["source"], ck.metadata["chunk_number"]))
            if idx is None:
                continue
            
            for i in range(max(0, idx - window_size), min(len(chunks), idx + window_size + 1)):
                key = (chunks[i].metadata["source"], chunks[i].metadata["chunk_number"])
                if key not in seen:
                    final_results.append(chunks[i])
                    seen.add(key)
        return final_results
    
    return docs

def dedup_docs(docs):
    """Deduplicate documents by source and chunk_number."""
    seen = set()
    out = []
    for doc in docs:
        key = (doc.metadata.get("source"), doc.metadata.get("chunk_number"))
        if key not in seen:
            seen.add(key)
            out.append(doc)
    return out
