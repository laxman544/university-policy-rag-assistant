# Demo Script – Agentic AI RAG Assistant

## Introduction (30–40 sec)

Hello, my name is Pinnaka Khantirava Venkat Laxman Kumar.

This project is an Agentic AI RAG Assistant designed to answer questions from a university student handbook using Retrieval-Augmented Generation.

---

## Problem Statement (40–60 sec)

University handbooks are usually long and difficult to navigate manually. Students often spend a lot of time searching for policies like attendance, grading, and graduation requirements.

This project solves that problem by allowing users to ask natural language questions and receive accurate answers instantly.

---

## Project Overview (1 min)

This system follows a Retrieval-Augmented Generation pipeline:

1. The handbook PDF is loaded
2. Text is extracted from the document
3. The text is split into smaller chunks
4. Each chunk is converted into embeddings
5. The embeddings are stored in a FAISS vector database

When a user asks a question:
- The system retrieves the most relevant chunks
- These chunks are passed to a language model
- The model generates a context-aware answer

---

## Architecture Explanation (1–2 min)

The pipeline flow is:

PDF → Text Extraction → Chunking → Embeddings → FAISS Index → Query → Retrieval → LLM → Answer

This ensures that:
- Only relevant information is used
- The model does not hallucinate
- Answers are grounded in actual document content

---

## Code Walkthrough (2–3 min)

Now I will briefly walk through the code:

- `loader.py`: extracts text from PDF  
- `chunker.py`: splits text into chunks  
- `embedder.py`: converts text into embeddings  
- `indexer.py`: stores embeddings in FAISS  
- `retriever.py`: retrieves top-k relevant chunks  
- `generator.py`: generates answers using LLM  
- `pipeline.py`: connects everything end-to-end  

---

## Demo (1–2 min)

Now I will demonstrate the system:

Example question:
"What is the attendance policy?"

The system retrieves relevant chunks and provides an answer based on the handbook content.

---

## Key Highlights (30 sec)

- Works with unstructured data (PDF)
- Uses semantic search instead of keyword search
- Improves accuracy using retrieval + LLM
- Scalable for multiple documents

---

## Conclusion (20–30 sec)

This project demonstrates my ability to work with:
- document processing
- embeddings and vector databases
- semantic retrieval
- LLM-based applications

Thank you.
