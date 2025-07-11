# 🚀 RAG System with Gemini & LangChain
  
  <img width="1920" height="1080" alt="Screenshot (41)" src="https://github.com/user-attachments/assets/d02818a1-d369-403a-bc0a-3af5672c1a48" />
  <img width="1920" height="1080" alt="Screenshot (42)" src="https://github.com/user-attachments/assets/9cd24460-fa85-4c70-acb7-e48102889bd4" />
  <img width="1920" height="1080" alt="Screenshot (43)" src="https://github.com/user-attachments/assets/8afe2924-ab44-47f2-86d1-300b5d42e0d0" />
  <img width="1920" height="1080" alt="Screenshot (44)" src="https://github.com/user-attachments/assets/e544216b-4d85-4cdf-8418-089e2444729b" />
  <img width="1920" height="1080" alt="Screenshot (45)" src="https://github.com/user-attachments/assets/e8e49c35-e786-4f2b-b079-97a5e48ac0b3" />
  
A production-ready **Retrieval-Augmented Generation (RAG)** system that lets you query your documents using Google's cutting-edge Gemini AI model.

## 🌟 Features

- **Document Processing**: Upload and chunk PDF documents
- **Smart Search**: Semantic retrieval using Gemini embeddings
- **AI Answers**: Natural language responses with Gemini Pro
- **REST API**: FastAPI backend with Swagger docs
- **Persistent Storage**: ChromaDB vector store
- **Easy Deployment**: Docker container support

## 🛠️ Tech Stack

- **Backend**: FastAPI
- **AI Models**: Google Gemini (embeddings & chat)
- **Vector DB**: ChromaDB
- **Orchestration**: LangChain
- **Containerization**: Docker

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Google API key (for Gemini)
- Docker (optional)

### Local Installation
```bash
git clone https://github.com/AbuZar-Ansarii/Rag---FastAPI.git
cd rag-gemini

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your Google API key to .env
