from langchain_community.document_loaders import PyPDFLoader  
from langchain_community.vectorstores import Chroma  
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.config import settings
import os
from typing import Any, Dict

class RAGSystem:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.GEMINI_API_KEY
        )
        self.llm = ChatGoogleGenerativeAI(model=settings.LLM_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.vector_db = None
        self.qa_chain = None

    def initialize_qa_chain(self) -> None:
        """Initializes the RetrievalQA chain with the current vector DB."""
        if not self.vector_db:
            raise ValueError("Vector database not initialized. Please load documents first.")

        custom_template = (
            "You are a helpful assistant. Use the context below to answer the question.\n"
            "Context: {context}\n"
            "Question: {question}\n"
            "Answer:"
        )

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=custom_template
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vector_db.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def upload_documents(self, file_path: str) -> int:
        """
        Loads a PDF, splits it into chunks, creates a vector DB, and initializes the QA chain.
        Returns the number of chunks created.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found at {file_path}")

        try:
            loader = PyPDFLoader(file_path)
            document = loader.load()
        except Exception as e:
            raise RuntimeError(f"Failed to load PDF: {e}")

        chunks = self.text_splitter.split_documents(document)

        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=settings.PERSIST_DIRECTORY
        )
        self.vector_db.persist()
        self.initialize_qa_chain()
        return len(chunks)

    def query(self, query: str):
        """
        Queries the QA chain with a question and returns the answer and source documents.
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Please load documents first.")

        result = self.qa_chain.invoke({"query": query})
        return {
            "query": query,
            "answer": result['result']
        }

# Singleton instance
rag_system = RAGSystem()