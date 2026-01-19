# 🚀 RAG Pipeline with MongoDB Atlas Vector Search

A production-ready Retrieval-Augmented Generation (RAG) system that combines MongoDB Atlas Vector Search with OpenAI for intelligent document querying. This system processes PDF documents, stores them with semantic embeddings, and enables natural language question-answering.

## 📋 Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Workflow Visualization](#workflow-visualization)
- [Technical Details](#technical-details)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Author](#author)

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

## 📋 Prerequisites

- Python 3.8+
- MongoDB Atlas account (free tier works)
- OpenAI API key
- Active internet connection

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/KrishnaSaiKoduru/rag-mongodb-vectorsearch.git
cd rag-mongodb-vectorsearch
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Required Packages
```txt
openai==1.6.1
pymongo==4.6.1
langchain==0.1.0
langchain-community==0.0.13
langchain-text-splitters==0.0.1
python-dotenv==1.0.0
pypdf==3.17.4
```

## ⚙️ Configuration

### 1. Create `.env` File
Create a `.env` file in the project root:
```env
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here

# MongoDB Atlas Configuration
MONGO_URI=your-mongodb-connection-string-here
```

### 2. Get OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create new API key
3. Copy to `.env` file

### 3. Set Up MongoDB Atlas
1. Create account at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create a free cluster
3. Get connection string
4. Add to `.env` file

**Connection String Format:**
```
mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?appName=RAG
```

## 📖 Usage Guide

### Step-by-Step Workflow

#### 1️⃣ **Environment Setup**
```python
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
mongo_uri = os.getenv("MONGO_URI")
```

#### 2️⃣ **Initialize OpenAI Client**
```python
from openai import OpenAI

openai_client = OpenAI(api_key=openai_api_key)
model = "text-embedding-3-small"

def get_embedding(text):
    response = openai_client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding
```

#### 3️⃣ **Load and Process PDF**
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("https://investors.mongodb.com/node/12236/pdf")
data = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=30
)
documents = text_splitter.split_documents(data)
```

#### 4️⃣ **Connect to MongoDB**
```python
from pymongo import MongoClient
from pymongo.server_api import ServerApi

mongo_client = MongoClient(mongo_uri, server_api=ServerApi('1'))
db = mongo_client["rag_db"]
collection = db["docs"]
```

#### 5️⃣ **Store Documents with Embeddings**
```python
# Add embeddings to documents
for doc in documents:
    doc_dict = {
        "text": doc.page_content,
        "embedding": get_embedding(doc.page_content),
        "metadata": doc.metadata
    }
    collection.insert_one(doc_dict)
```

#### 6️⃣ **Create Vector Search Index**
```python
from pymongo.operations import SearchIndexModel

search_index_model = SearchIndexModel(
    definition={
        "fields": [{
            "type": "vector",
            "path": "embedding",
            "similarity": "cosine",
            "numDimensions": 1536
        }]
    },
    name="vector_index",
    type="vectorSearch"
)

collection.create_search_index(model=search_index_model)
```

#### 7️⃣ **Query Documents**
```python
def get_query_results(query, num_results=5):
    query_embedding = get_embedding(query)
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "exact": True,
                "limit": num_results
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    results = collection.aggregate(pipeline)
    return list(results)
```

#### 8️⃣ **Generate AI Answers (RAG)**
```python
# Get relevant context
query = "What are MongoDB's investments on AI?"
context_docs = get_query_results(query)
context_string = " ".join([doc["text"] for doc in context_docs])

# Create prompt
prompt = f"""Use the following pieces of context to answer the question at the end.
    {context_string}
    Question: {query}
"""

# Generate answer
completion = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}]
)

print(completion.choices[0].message.content)
```

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

## 🔧 Technical Details

### Embedding Model
- **Model**: `text-embedding-3-small`
- **Dimensions**: 1536
- **Max Tokens**: 8191
- **Use Case**: Semantic search, similarity comparison

### Text Chunking Strategy
- **Chunk Size**: 400 characters
  - Small enough for focused context
  - Large enough for semantic coherence
- **Overlap**: 30 characters
  - Preserves context across boundaries
  - Prevents information loss

### Vector Search Configuration
- **Index Type**: HNSW (Hierarchical Navigable Small World)
- **Similarity Metric**: Cosine similarity
- **Search Type**: Exact match (high accuracy)
- **Results**: Top 5 most relevant chunks

### LLM Configuration
- **Model**: GPT-4o
- **Temperature**: Default (0.7)
- **Max Tokens**: Automatic
- **Role**: Answer generation from context

## 📁 Project Structure
```
rag-mongodb-vectorsearch/
│
├── rag_mongo.ipynb          # Main Jupyter notebook
├── .env                      # Environment variables (not tracked)
├── .gitignore               # Git ignore rules
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
└── .venv/                  # Virtual environment (not tracked)
```

## 🐛 Troubleshooting

### Common Issues

**1. ModuleNotFoundError for langchain.chains**
```bash
pip install langchain langchain-community
```

**2. MongoDB Connection Failed**
```
- Check MongoDB Atlas cluster is running
- Verify connection string in .env
- Whitelist your IP address in Atlas
- Check username/password are correct
```

**3. OpenAI API Error**
```
- Verify API key is valid
- Check account has credits
- Ensure .env file is loaded
```

**4. Vector Index Not Found**
```
- Wait 1-2 minutes after creating index
- Verify index name matches query
- Check index status in Atlas UI
```

**5. Client Variable Conflict**
```
- Use openai_client for OpenAI
- Use mongo_client for MongoDB
- Never reuse variable names
```

## 🎓 Learning Resources

- [MongoDB Vector Search Documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [RAG Architecture Explained](https://docs.anthropic.com/claude/docs/retrieval-augmented-generation)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Krishna Sai Koduru**
- GitHub: [@KrishnaSaiKoduru](https://github.com/KrishnaSaiKoduru)
- LinkedIn: [Krishna Sai Koduru](https://www.linkedin.com/in/krishnasaikoduru)

## 🙏 Acknowledgments

- MongoDB for Atlas Vector Search capabilities
- OpenAI for powerful embedding and language models
- LangChain for document processing utilities
- The open-source community

## 📧 Contact

For questions, issues, or feedback:
- Open an [Issue](https://github.com/KrishnaSaiKoduru/rag-mongodb-vectorsearch/issues)
- Email: your.email@example.com

---

⭐ **If you find this project useful, please give it a star!**

**Built with ❤️ by Krishna | Powered by MongoDB Atlas & OpenAI**
