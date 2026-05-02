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



  
def ask_rag(question):
  quiries =expand_query(question)
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
  sorted_list = [(int(i), float(score)) for i, score in sorted_indices]
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


