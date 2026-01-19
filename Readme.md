# 🚀 RAG Pipeline with MongoDB Atlas Vector Search

A production-ready Retrieval-Augmented Generation (RAG) system that combines MongoDB Atlas Vector Search with OpenAI for intelligent document querying. This system processes PDF documents, stores them with semantic embeddings, and enables natural language question-answering.

## 📋 Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Workflow Visualization](#workflow-visualization)
- [Project Structure](#project-structure)

## 🎯 Overview

This project implements a complete RAG (Retrieval-Augmented Generation) pipeline that:
1. **Ingests** PDF documents from URLs
2. **Processes** text into semantically meaningful chunks
3. **Generates** vector embeddings using OpenAI
4. **Stores** documents in MongoDB Atlas with vector search capabilities
5. **Retrieves** relevant context using semantic similarity search
6. **Generates** accurate answers using GPT-4

## 🏗️ System Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Pipeline Architecture                     │
└─────────────────────────────────────────────────────────────────┘

    PDF Document (URL)
          │
          ▼
    ┌──────────────┐
    │ PyPDFLoader  │  ← Load PDF from MongoDB investor docs
    └──────┬───────┘
           │
           ▼
    ┌──────────────────────┐
    │ Text Splitter        │  ← Split into 400-char chunks (30 overlap)
    │ - Chunk Size: 400    │
    │ - Overlap: 30        │
    └──────┬───────────────┘
           │
           ▼
    ┌──────────────────────┐
    │ OpenAI Embeddings    │  ← Generate 1536-dim vectors
    │ (text-embedding-3)   │
    └──────┬───────────────┘
           │
           ▼
    ┌──────────────────────┐
    │ MongoDB Atlas        │  ← Store docs + embeddings
    │ - Collection: docs   │
    │ - Vector Index       │
    └──────┬───────────────┘
           │
           ├────────────────┐
           │                │
           ▼                ▼
    ┌─────────────┐  ┌──────────────┐
    │ Vector      │  │ GPT-4        │
    │ Search      │  │ Generation   │
    │ (Retrieval) │→ │ (Answer)     │
    └─────────────┘  └──────────────┘
```

## ✨ Features

- **📄 Automated PDF Processing**: Load and process PDFs directly from URLs
- **🔍 Semantic Search**: Vector-based similarity search using cosine similarity
- **🤖 AI-Powered Q&A**: GPT-4 powered answer generation with context
- **💾 Persistent Storage**: MongoDB Atlas for scalable document storage
- **⚡ Fast Retrieval**: Optimized vector search with HNSW indexing
- **🔐 Secure**: Environment-based configuration for API keys
- **📊 Real-time Updates**: Dynamic document ingestion and querying


## 📊 Workflow Visualization

### Data Ingestion Flow
```
┌─────────────────────────────────────────────────────────┐
│                  Data Ingestion Pipeline                 │
└─────────────────────────────────────────────────────────┘

Step 1: Load Environment Variables
   ↓
   • OPENAI_API_KEY (from .env)
   • MONGO_URI (from .env)

Step 2: Initialize Clients
   ↓
   • OpenAI Client (for embeddings & GPT)
   • MongoDB Client (for storage)

Step 3: Load PDF Document
   ↓
   • PyPDFLoader fetches PDF from URL
   • Extracts text from all pages
   Output: List of Document objects

Step 4: Text Chunking
   ↓
   • RecursiveCharacterTextSplitter
   • Chunk Size: 400 characters
   • Overlap: 30 characters
   Output: ~300 document chunks

Step 5: Generate Embeddings
   ↓
   • For each chunk, call OpenAI API
   • Model: text-embedding-3-small
   • Output: 1536-dimensional vector

Step 6: Store in MongoDB
   ↓
   • Insert document with:
     - text (original content)
     - embedding (vector)
     - metadata (page, source, etc.)
   Output: Documents stored in collection

Step 7: Create Vector Index
   ↓
   • Index Type: vectorSearch
   • Similarity: cosine
   • Dimensions: 1536
   Output: Searchable vector index
```

### Query & Retrieval Flow
```
┌─────────────────────────────────────────────────────────┐
│              Question Answering Pipeline                 │
└─────────────────────────────────────────────────────────┘

User Query: "What are MongoDB's AI investments?"
   ↓
Step 1: Generate Query Embedding
   ↓
   • Convert question to 1536-dim vector
   • Uses same embedding model

Step 2: Vector Similarity Search
   ↓
   • MongoDB $vectorSearch aggregation
   • Cosine similarity comparison
   • Returns top 5 most similar chunks
   Output: [
     {text: "...", score: 0.89},
     {text: "...", score: 0.85},
     ...
   ]

Step 3: Context Preparation
   ↓
   • Combine retrieved chunks into context
   • Format: "context_1 context_2 context_3..."

Step 4: Prompt Construction
   ↓
   • Template: "Use the following context...
                {context}
                Question: {query}"

Step 5: GPT-4 Generation
   ↓
   • Send prompt to GPT-4o
   • Model generates answer based on context
   • No hallucination (grounded in retrieved docs)

Step 6: Return Answer
   ↓
   Output: Coherent, contextual answer ✓
```

