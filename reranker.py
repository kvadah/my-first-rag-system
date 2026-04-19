from sentence_transformers import CrossEncoder
reranker =CrossEncoder("BAAI/bge-reranker-large")
def rerank(documents,question,top_k=5):
  pairs = [(question, doc["text"]) for doc in documents]
  scores = reranker.predict(pairs)
  ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
  ranked_docs = [doc["text"] for _, doc in ranked[:top_k]]

  return ranked_docs