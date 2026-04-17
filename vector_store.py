import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import streamlit as st

class ResearchVectorStore:
    def __init__(self):
        # Use a lightweight, high-performance model for 2026 standards
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )

    def create_store(self, text: str, source_name: str = "Research Paper"):
        """Converts raw text into a searchable FAISS index."""
        if not text.strip():
            return None
            
        # 1. Create Document objects
        docs = [Document(page_content=text, metadata={"source": source_name})]
        
        # 2. Split into chunks
        chunks = self.text_splitter.split_documents(docs)
        
        # 3. Build Vector Store
        vector_db = FAISS.from_documents(chunks, self.embeddings)
        return vector_db

    def similarity_search(self, vector_db, query: str, k: int = 3):
        """Retrieves the top K relevant chunks for a given query."""
        if vector_db is None:
            return "No vector store available to search."
            
        results = vector_db.similarity_search(query, k=k)
        return "\n\n".join([res.page_content for res in results])

# Singleton instance to be used across the LangGraph nodes
research_memory = ResearchVectorStore()
