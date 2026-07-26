# 🧠 Scalable Multimodal Hybrid RAG Agent

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green.svg)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agents-orange.svg)](https://www.langchain.com/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/LLM-Llama_3.3_70B_(Groq)-purple.svg)](https://groq.com/)

An enterprise-grade, production-ready Multimodal Hybrid RAG System designed for high-precision document retrieval and conversational QA. 

This architecture combines Sparse Keyword Search (BM25) and Dense Vector Retrieval (ChromaDB) with a Cross-Encoder Re-ranker to eliminate context noise. The agent's workflow and conversation states are managed using LangGraph paired with persistent state check-pointing (MemorySaver).

---

## 🌟 Key Features

* Hybrid Search Retrieval: Combines the semantic strength of Dense Embeddings (all-MiniLM-L6-v2) with the exact-keyword accuracy of Sparse Search (BM25) via an EnsembleRetriever (50/50 weighting).
* Two-Stage Re-Ranking: Utilizes a Cross-Encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2) to re-score and filter the top candidates, guaranteeing high context relevance for the LLM.
* Stateful Agent Workflow: Built with LangGraph to manage multi-step routing, document retrieval nodes, and context synthesis cleanly.
* Conversation Memory Persistence: Employs MemorySaver check-pointing keyed by unique thread_ids, allowing seamless multi-turn conversations without context bleed.
* Modular Codebase: Fully decoupled architecture separating document ingestion, hybrid retrieval, agent graphs, and presentation layers (Streamlit).

---

## 🏗️ System Architecture

`text
┌────────────────┐     ┌─────────────────────────────────────────┐
│ User Prompt    │ ──> │ LangGraph Stateful Workflow             │
└────────────────┘     └─────────────────────────────────────────┘
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │   Retrieve Node       │
                       └───────────────────────┘
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         │                                                   │
         ▼                                                   ▼
┌─────────────────┐                                 ┌─────────────────┐
│ Dense Retrieval │ (Chroma VectorStore)            │ Sparse Search   │ (BM25 Keyword)
└─────────────────┘                                 └─────────────────┘
         │                                                   │
         └─────────────────────────┬─────────────────────────┘
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │ Cross-Encoder Rerank  │ (MS-MARCO Model)
                       └───────────────────────┘
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │   Generate Node       │ (Llama-3.3-70b via Groq)
                       └───────────────────────┘
                                   │
                                   ▼

                       ┌───────────────────────┐
                       │ Stateful Memory State │ (MemorySaver Threading)
                      └───────────────────────┘
**Created By: Yassin Sanad**
---

