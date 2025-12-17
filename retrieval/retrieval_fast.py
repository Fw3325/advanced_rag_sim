"""
Ultra-fast retrieval pipeline - skips expensive operations.
Designed for maximum speed with minimal quality loss.
"""
import time
import concurrent.futures
from typing import List
from .bm25_retriever import BM25Retriever
from .rrf import rrf_fuse

def advanced_multi_query_similarity_search_fast(
    llm, vectorstore, chunks, user_query, 
    k=8, rrf_top_n=10, mode='hybrid',
    kg_retriever=None, enable_kg=True
):
    """
    Ultra-fast retrieval - NO query rewriting, NO reranking, NO MMR.
    Just direct retrieval + RRF fusion.
    
    Speed optimizations:
    1. Skip query rewriting (saves 1-2s LLM call)
    2. Skip reranking (saves 5-10s)
    3. Skip MMR (saves 1-2s embedding time)
    4. Direct retrieval only
    """
    start_time = time.time()
    
    bm25 = BM25Retriever(chunks)
    candidate_chunks = []
    
    # Direct retrieval with original query (no sub-queries)
    print(f"⚡ Fast retrieval for: {user_query[:50]}...")
    
    # Parallel retrieval: dense + BM25 + KG
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        
        # Dense search
        if mode in ["dense", "hybrid"]:
            futures['dense'] = executor.submit(
                lambda: [doc for doc, _ in vectorstore.similarity_search_with_score(user_query, k)]
            )
        
        # BM25 search
        if mode in ["bm25", "hybrid"]:
            futures['bm25'] = executor.submit(lambda: bm25.search(user_query, k))
        
        # KG search
        if enable_kg and kg_retriever is not None:
            futures['kg'] = executor.submit(kg_retriever.search, user_query, k)
        
        # Collect results
        dense_results = futures.get('dense', concurrent.futures.Future()).result() if 'dense' in futures else []
        bm25_results = futures.get('bm25', concurrent.futures.Future()).result() if 'bm25' in futures else []
        kg_results = futures.get('kg', concurrent.futures.Future()).result() if 'kg' in futures else []
    
    retrieval_time = time.time() - start_time
    print(f"⚡ Retrieval completed in {retrieval_time:.2f}s")
    
    # Merge using RRF
    all_results = []
    if dense_results:
        all_results.append(dense_results)
    if bm25_results:
        all_results.append(bm25_results)
    if kg_results:
        all_results.append(kg_results)
        print(f"🔗 KG retrieved {len(kg_results)} chunks")
    
    if all_results:
        candidate_chunks = rrf_fuse(all_results, k=rrf_top_n)
    else:
        candidate_chunks = []
    
    # Deduplicate
    seen = set()
    final_chunks = []
    for doc in candidate_chunks:
        key = (doc.metadata.get("source"), doc.metadata.get("chunk_number"))
        if key not in seen:
            seen.add(key)
            final_chunks.append(doc)
    
    total_time = time.time() - start_time
    print(f"⚡ Fast retrieval total: {total_time:.2f}s ({len(final_chunks)} chunks)")
    
    return final_chunks[:k]  # Return top k
