from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, AutoTokenizer,AutoModelForSeq2SeqLM
from openai import OpenAI
import os
from context_aware_chunking import advanced_smart_chunk
from pydantic import BaseModel
from fastapi import FastAPI
from query_expansion import expand_query
from reranker import rerank
from answer_generator import generate_answer

def QueryRequest(BaseModel):
   question:str

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
gen_model =AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
model = SentenceTransformer("all-mpnet-base-v2")
tfidf=TfidfVectorizer()
reranker = CrossEncoder("BAAI/bge-reranker-large")
app = FastAPI()

path = "ET_Criminal_Code.pdf"
documents=advanced_smart_chunk(path)
texts =[doc["text"] for doc in documents ]
metadatas = [doc['metadata'] for doc in documents]
embedings = model.encode(texts)
faiss.normalize_L2(embedings)
embedings= np.array(embedings).astype('float32')
dimension =embedings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embedings)
tfidf_matrix = tfidf.fit_transform(texts)


id_to_metadata = {i: metadatas[i] for i in range(len(metadatas))}

def keyword_search(query,top_k=10):
  query_vec=tfidf.transform([query])
  scores =cosine_similarity(query_vec,tfidf_matrix)[0]
  top_indices = scores.argsort()[-top_k:][::-1]
  return top_indices, scores


def vector_search(question,k=10):
  question_embeding=model.encode([question])
  question_embeding = np.array(question_embeding).astype('float32')
  faiss.normalize_L2(question_embeding)
  k=5
  distances, indices = index.search(question_embeding,k)
  return distances,indices

def is_article_query(queries):
  for query in queries:
    return 'article' in query.lower()


def filter_by_article(article_number):
  return [doc for doc in documents if article_number == doc['metadata']['article_number']]


def extract_article_number(query: str):
    match = re.search(r'article\s+(\d+)', query.lower())
    return match.group(1) if match else None
def rrf_fusion(vector_indices, keyword_indices, k=60):
    scores = {}

    for rank, idx in enumerate(vector_indices):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)

    for rank, idx in enumerate(keyword_indices):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

  
def ask_rag(question):
  quiries =expand_query(question)
  quiries =expand_query(question)
  article_number=0
  if is_article_query(quiries):
    article_number=extract_article_number(quiries[0])
    print("yes")
  if article_number:
          filtered_docs = filter_by_article(article_number)
          if filtered_docs:
              print(f"Filtered to {len(filtered_docs)} docs for Article {article_number}")
              ranked = rerank(filtered_docs, quiries[0], top_k=5)
              context = "\n\n".join([doc["text"] for doc in ranked])
              return generate_answer(context, question)
  quiries.append(question)
  all_vector_indices=[]
  all_vector_scores=[]
  all_keyword_indices=[]
  all_keyword_scores=[]
  for q in quiries:
    print(q)
    vector_scores, vector_indices = vector_search(q)
    keyword_indices, keyword_scores = keyword_search(q)
    all_vector_indices.append(vector_indices)
    all_vector_scores.append(vector_scores)
    all_keyword_indices.append(keyword_indices)
    all_keyword_scores.append(keyword_scores)

  all_vector_indices=np.concatenate(all_vector_indices)
  all_vector_scores=np.concatenate(all_vector_scores)
  all_keyword_indices=np.concatenate(all_keyword_indices)
  all_keyword_scores=np.concatenate(all_keyword_scores)
  combined_distances = {}
  alpha=0.65
  for i, score in zip(all_vector_indices, all_vector_scores):
      combined_distances[i] = alpha * score

  for i, score in zip(all_keyword_indices, all_keyword_scores): 
    if i in combined_distances:
        combined_distances[i] += (1 - alpha) * score
    else:
        combined_distances[i] = (1 - alpha) * score


  combined_distances = sorted(combined_distances.items(), key=lambda x: x[1], reverse=True)
  sorted_list = [(int(i), float(score)) for i, score in combined_distances]
  retrieved=[]
  for i,_ in sorted_list[:40]:
        retrieved.append({
            "text":texts[i],
            "metadata":id_to_metadata[i]
        })
  print("combined")
  print(combined_distances)
  if not retrieved:
    return "I don't know"
  ranked_docs=rerank(retrieved,question)
  if len(ranked_docs)==0:
    return "I don't know"
  context = "\n\n".join(ranked_docs[:3])

  
  return generate_answer(context,question)


@app.post("/ask")
def ask(request:QueryRequest):
   answer =ask_rag(request.question)
   return {"answer": answer}


