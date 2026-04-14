from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, AutoTokenizer,AutoModelForSeq2SeqLM
from openai import OpenAI
import os
import pdfplumber

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
gen_model =AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
model = SentenceTransformer("all-mpnet-base-v2")
tfidf=TfidfVectorizer()
reranker = CrossEncoder("BAAI/bge-reranker-large")
text1 = """
Global Dynamics Tech Solutions, founded in 1998 by Dr. Helena Vance in Seattle, remains a pioneer in the integration 
of specialized cloud infrastructure and proprietary machine learning algorithms designed for the manufacturing sector.
The company's flagship product, 'NexusCore,' utilizes advanced neural networks to predict equipment failure before it 
occurs, a process known in the industry as predictive maintenance. While their primary headquarters moved to a sustainable 
campus in Austin, Texas in 2014, they maintain satellite offices in Berlin and Tokyo to manage their international logistics
and supply chain optimization software. In 2022, Global Dynamics announced a strategic partnership with the European Space 
Agency to develop radiation-hardened semiconductors for deep-space exploration probes. Their corporate culture emphasizes a 
'remote-first' approach, utilizing agile methodology and various DevOps tools like Jenkins and Kubernetes to maintain high deployment 
frequency. Despite market volatility, the firm recently reported a twenty percent increase in annual recurring revenue, largely attributed 
to their new cybersecurity suite, 'AegisShield,' which offers end-to-end encryption and biometric authentication for remote workers. 
Sustainability is also a core pillar of their operations, as they have pledged to be carbon neutral by 2030 through the use of 
solar-powered data centers and advanced water-cooling systems for their server racks.
"""

def smart_chunk(text, max_chunk_size=200):
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def extract_text_from_pdf(filepath):
  text =""
  with pdfplumber.open(filepath) as pdf:
    for page in pdf.pages:
      text+=page.extract_text()+"\n"
  
  return text
path = "nelson-mandela.pdf"
text=extract_text_from_pdf(path)
documents=smart_chunk(text)
embedings = model.encode(documents)
faiss.normalize_L2(embedings)
embedings= np.array(embedings).astype('float32')
dimension =embedings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embedings)
tfidf_matrix = tfidf.fit_transform(documents)





def keyword_search(query,top_k=5):
  query_vec=tfidf.transform([query])
  scores =cosine_similarity(query_vec,tfidf_matrix)[0]
  top_indices = scores.argsort()[-top_k:][::-1]
  return top_indices, scores


def vector_search(question,k=5):
  question_embeding=model.encode([question])
  question_embeding = np.array(question_embeding).astype('float32')
  faiss.normalize_L2(question_embeding)
  k=5
  distances, indices = index.search(question_embeding,k)
  return distances,indices

def generate_answer(context,question):
  prompt =f"""You are a Technical Assistant. Your goal is to explain concepts clearly based on the provided documentation..

    Rules:
    - Do NOT guess
    - Do NOT add external knowledge
    - If the answer is not clearly stated, say EXACTLY: "I don't know"
    - Synthesize the information into a professional summary.
    - Do NOT copy the text word-for-word.
    - Use a helpful, educational tone.
    - If the context doesn't have the answer, say "I'm sorry, I don't have enough information on that topic."
    -Answer in a complete sentence.Do not give short phrases.

    Context:
    {context}

    Question:
    {question}
    """
  inputs = tokenizer(prompt,return_tensors="pt")
  outputs= gen_model.generate(**inputs,max_length=100,temperature=0.7,repetition_penalty=1.6,top_p=0.6)
  return tokenizer.decode(outputs[0],skip_special_tokens=True)


def generate_answer1(context, question):
    prompt = f"""
You are a highly accurate technical assistant.

STRICT RULES:
- Answer ONLY using the provided context
- Do NOT add external knowledge
- If the answer is not in the context, only say: "I am sorry I don't have enough knowldge about it "
- Be clear, structured, and professional

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        messages=[
            {"role": "system", "content": "You are a precise RAG assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=200
    )

    return response.choices[0].message.content


def rewrite_question(question):
  rewrite_prompt = f"""Rewrite the question for better document retrieval.

    STRICT RULES:
    - Do NOT change the meaning
    - Do NOT add new information
    - Only clarify or expand slightly
    - Keep the same intent
    - If already clear, return it unchanged

    Question: {question}
    Rewritten:
          """
  inputs = tokenizer(rewrite_prompt,return_tensors="pt")
  outputs= gen_model.generate(**inputs,max_length=100,temperature=0.2,do_sample=False)
  return tokenizer.decode(outputs[0],skip_special_tokens=True)

def choose_better_question(question):
  rewritten=rewrite_question(question)
  question_embeding=model.encode([question])
  rewritten_embeding = model.encode([rewritten])
  question_score = cosine_similarity(rewritten_embeding,question_embeding)
  print(f"question_score {question_score}")
  print(f"question {question}")
  print(f"rewritten {rewritten}")
  if question_score>0.8:
    return rewritten
  else:
     return question



def rerank(documents,question,top_k=5):
  pairs = [(question, doc) for doc in documents]
  scores = reranker.predict(pairs)
  ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
  ranked_docs = [doc for _, doc in ranked[:top_k]]

  return ranked_docs
  
def ask_rag(question):

  question =choose_better_question(question)
  print(question)
  question=question.lower()
  question_embeding=model.encode([question])
  vector_distances,vector_indices=vector_search(question)
  vector_indices=vector_indices[0].tolist()
  vector_distances=vector_distances[0].tolist()
  keyword_indices, keyword_distances = keyword_search(question,)
  keyword_distances=keyword_distances.tolist()
  print(keyword_indices)
  combined_indices = list(set(vector_indices + keyword_indices.tolist()))
  # combined_distances =list(set(vector_distances+keyword_distances.tolist()))
  combined_distances = {}
  alpha=0.65
    # Vector scores
  for i, score in zip(vector_indices, vector_distances):
      combined_distances[i] = alpha * score

  # Keyword scores
  for i in keyword_indices:
      score = keyword_distances[i]
      if i in combined_distances:
          combined_distances[i] += (1 - alpha) * score
      else:
          combined_distances[i] = (1 - alpha) * score


  combined_distances = sorted(combined_distances.items(), key=lambda x: x[1], reverse=True)
  retrieved = [documents[i] for i,_ in combined_distances]
  print("combined")
  print(combined_indices)
  print(combined_distances)
  #print(distances[0])
  # print(indices[0])
  # for i in indices[0]:
  #   print(documents[i])
  # for i, score in score_map.items():
  #   if score > 0.3:
  #     retrieved.append(documents[i])
  # print(retrieved)
  if not retrieved:
    return "I don't know"
  ranked_docs=rerank(retrieved,question)
  if len(ranked_docs)==0:
    return "I don't know"
  context = "\n\n".join(ranked_docs[:3])

  
  return generate_answer1(context,question)
print(ask_rag("who suceeded mandela?"))



