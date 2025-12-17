# reciprocal rank fusion
def rrf_fuse(list_of_lists, k=60):
    """
    Reciprocal Rank Fusion: merge rankings from multiple retrievers
    """
    rank_scores = {}

    for result_list in list_of_lists:
        for rank, doc in enumerate(result_list):
            key = (doc.metadata["source"], doc.metadata["chunk_number"])
            rank_scores[key] = rank_scores.get(key, 0) + 1/(k + rank + 1)
            
    # Return as sorted list of unique Document objects
    key_to_doc = { (doc.metadata["source"], doc.metadata["chunk_number"]): doc 
                    for r in list_of_lists for doc in r }
    final_sorted_chunks = [key_to_doc[k] for k, _ in sorted(rank_scores.items(), key=lambda x: x[1], reverse=True)]

    return final_sorted_chunks
