 Ethiopian Criminal Code RAG API

A Retrieval-Augmented Generation (RAG) system designed to answer legal questions from the Ethiopian Criminal Code using hybrid search, reranking, and LLM-based reasoning.

🚀 Overview

This project implements a high-performance legal question-answering system that combines:

Semantic search (vector embeddings)
Keyword-based retrieval (TF-IDF)
Reciprocal Rank Fusion (RRF)
Cross-encoder reranking
LLM-based answer generation

It is exposed as a FastAPI service, allowing integration with web or mobile applications.

 Key Features
 Hybrid Search
Combines:
Vector search (FAISS + BGE embeddings)
Keyword search (TF-IDF)
Uses Reciprocal Rank Fusion (RRF) for robust ranking
🧾 Legal-Aware Chunking
Splits documents by:
Article structure
Sentence boundaries
Maintains metadata:
Article title
Article number
 Query Expansion
Expands user queries using LLM
Improves recall and retrieval coverage
Intelligent Query Routing

Supports multiple query types:

FACT → Direct answer from context
LIST → Extract and summarize multiple items
COUNT → (planned)
MULTI → (planned)
REASONING → (basic support)
 Article-Based Filtering
Detects queries like:
“What does Article 246 say?”
Filters documents before retrieval
Improves precision and speed
 Cross-Encoder Reranking
Uses BAAI/bge-reranker-large
Reorders retrieved chunks for maximum relevance
 Controlled LLM Generation
Strict prompts:
No hallucination
No external knowledge
Context-only answers
Supports:
Fact answers
Structured list extraction
🌐 API Deployment
Built with FastAPI
Public exposure using ngrok

Core Libraries
sentence-transformers – Embeddings
faiss – Vector similarity search
scikit-learn – TF-IDF & cosine similarity
transformers – NLP models
pypdf – PDF parsing

 Backend
FastAPI – API framework
uvicorn – ASGI server
ngrok – Public endpoint

 Models
Embeddings:
BAAI/bge-large-en-v1.5
Reranker:
BAAI/bge-reranker-large
LLM:
llama-3.3-70b-versatile (via Groq API)

🧠 Intelligence Improvements
 Multi-query decomposition
 Better query classification
 Fuzzy matching for user input
 Intent-aware routing
📊 Evaluation & Monitoring
 Logging pipeline
 Retrieval accuracy metrics
 Answer correctness evaluation
⚡ Performance
 Caching embeddings
 Response caching
 Async processing
 
 Key Concepts Used
Retrieval-Augmented Generation (RAG)
Dense + Sparse Hybrid Search
Reciprocal Rank Fusion (RRF)
Cross-Encoder Reranking
Prompt Engineering
Legal Document Structuring


👨‍💻 Author
Khalid
Computer Science Student – Addis Ababa University
Focus: AI Systems, and Backend Engineering

 License
This project is for educational and research purposes.

 Final Note
This system already implements core ideas used in production RAG systems (Perplexity, ChatGPT retrieval, etc.). With further optimization, it can evolve into a full-scale legal assistant platform.
