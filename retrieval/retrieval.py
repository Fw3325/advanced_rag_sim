# advanced retrieval methods: embedding -> reranker -> window expand
from .reranker import BGEReranker
from .query_rewriter import rewrite_query
from .bm25_retriever import BM25Retriever
from .rrf import rrf_fuse
from .mmr import mmr_select
reranker = BGEReranker()

# Import optimized version
try:
    from .retrieval_optimized import advanced_multi_query_similarity_search_optimized
except ImportError:
    advanced_multi_query_similarity_search_optimized = None

def advanced_multi_query_similarity_search(llm, vectorstore, chunks, user_query, 
                                            k=8, window_expansion=False, rrf_top_n=10, mode='hybrid',
                                            enable_mmr=True, mmr_top_n=8, kg_retriever=None, enable_kg=True,
                                            fast_mode=False, use_optimized=True):
    '''
    Multi-query RAG (Reason → Rewrite → Retrieval):

    1. Rewrite user query → multiple research sub-queries
    2. Each sub-query:
        a) embedding similarity search (recall)
        b) BM25 search (keyword matching)
        c) Knowledge Graph search (entity-based)
        d) BGE reranker (precision)
        e) neighbor expansion (context preservation)
    3. Merge & deduplicate
    
    Use optimized version if available and use_optimized=True.
    '''
    # Use optimized version if available
    if use_optimized and advanced_multi_query_similarity_search_optimized is not None:
        # In fast mode, use ultra-fast version if available
        if fast_mode:
            try:
                from .retrieval_fast import advanced_multi_query_similarity_search_fast
                return advanced_multi_query_similarity_search_fast(
                    llm, vectorstore, chunks, user_query, k, rrf_top_n, 
                    mode, kg_retriever, enable_kg
                )
            except ImportError:
                pass  # Fall back to optimized version
        
        return advanced_multi_query_similarity_search_optimized(
            llm, vectorstore, chunks, user_query, k, window_expansion, 
            rrf_top_n, mode, enable_mmr, mmr_top_n, kg_retriever, enable_kg,
            fast_mode=fast_mode
        )
    
    # Original implementation (fallback)
    #  ① rewrite query for better recall
    sub_queries = rewrite_query(llm, user_query)
    if fast_mode:
        sub_queries = [user_query] + sub_queries[:1]  # Reduce sub-queries in fast mode
    print("\n🧠 Generated sub-queries:", sub_queries)

    bm25 = BM25Retriever(chunks)
    candidate_chunks = []

    # ① KG retrieval on original query (entity-based, not per sub-query)
    kg_results = []
    if enable_kg and kg_retriever is not None:
        kg_results = kg_retriever.search(user_query, k=k)
        print(f"🔗 KG retrieved {len(kg_results)} chunks")

    for q in sub_queries:
        # similarity score: the smallest, the best as this is distance
        dense_results = [doc for doc, _ in vectorstore.similarity_search_with_score(q, k)]
        if mode == "dense":
            merged = dense_results
        elif mode == "bm25":
            merged = bm25.search(q, k)
        elif mode == "hybrid":
            bm25_results = bm25.search(q, k)
            # dense + bm25 → RRF (dedup)
            merged = rrf_fuse([dense_results, bm25_results])[:rrf_top_n]
        else:
            raise ValueError("mode must be 'dense', 'bm25', or 'hybrid'")

        # ② reranker (cross-encoder BGE large) - SKIP in fast mode
        rerank_top_n = max(3, int(rrf_top_n / 2))
        if not fast_mode:
            reranked = reranker.rerank(q, merged, top_n=rerank_top_n)
            candidate_chunks.extend(reranked)
        else:
            # Fast mode: just take top N without reranking
            candidate_chunks.extend(merged[:rerank_top_n])
    
    # ③ Merge KG results with other candidates using RRF
    if enable_kg and kg_results:
        all_candidates = [candidate_chunks, kg_results]
        candidate_chunks = rrf_fuse(all_candidates, k=rrf_top_n * 2)

    # ③ optional MMR diversity re-ranking
    docs = dedup_docs(candidate_chunks)

    if not fast_mode and enable_mmr and len(docs) > mmr_top_n:
        doc_embeddings = vectorstore.embedding_function.embed_documents([d.page_content for d in docs])
        query_vec = vectorstore.embedding_function.embed_query(user_query)

        mmr_docs = mmr_select(query_vec, doc_embeddings, docs, top_k=mmr_top_n)
    elif fast_mode:
        mmr_docs = docs[:mmr_top_n]  # Fast mode: skip MMR

    # ④ neighbor expansion (context completeness) - no need with current chunking strategy
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

    return mmr_docs if enable_mmr else docs

def dedup_docs(docs):
    seen = set()
    out = []
    for doc in docs:
        key = (doc.metadata.get("source"),doc.metadata.get("chunk_number"))
        if key not in seen:
            seen.add(key)
            out.append(doc)
    return out
