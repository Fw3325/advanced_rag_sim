import numpy as np

def mmr_select(query_vec, doc_vecs, docs, top_k=8, lambda_=0.7):
    doc_vecs = np.array(doc_vecs)
    selected = []
    candidates = list(range(len(docs)))

    while len(selected) < top_k and candidates:
        if not selected:
            i = int(np.argmax(np.dot(doc_vecs, query_vec)))
            selected.append(i)
            candidates.remove(i)
        else:
            sim_to_query = np.dot(doc_vecs[candidates], query_vec)
            sim_to_selected = np.max(np.dot(doc_vecs[candidates], doc_vecs[selected].T), axis=1)
            mmr = lambda_ * sim_to_query - (1 - lambda_) * sim_to_selected
            i = candidates[int(np.argmax(mmr))]
            selected.append(i)
            candidates.remove(i)

    return [docs[i] for i in selected]
