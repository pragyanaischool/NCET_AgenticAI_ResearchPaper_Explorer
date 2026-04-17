import os
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

class ResearchTools:
    def __init__(self):
        self.search = SerpAPIWrapper()
        # Initialize embeddings for RAG (Research Augmented Generation)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    @tool
    def search_papers(self, query: str):
        """Searches the web and academic databases for research papers using SerpAPI."""
        return self.search.run(f"site:arxiv.org OR site:researchgate.net {query}")

    @tool
    def process_pdf(self, pdf_url: str):
        """Downloads and extracts text from a PDF URL to be used for deep analysis."""
        try:
            loader = PyMuPDFLoader(pdf_url)
            data = loader.load()
            # Combine all pages
            full_text = " ".join([page.page_content for page in data])
            return full_text[:10000]  # Return first 10k chars to stay within context limits
        except Exception as e:
            return f"Error processing PDF: {str(e)}"

    def create_vector_store(self, text: str):
        """Converts raw text into a searchable FAISS vector database."""
        chunks = self.text_splitter.split_text(text)
        vector_db = FAISS.from_texts(chunks, self.embeddings)
        return vector_db

# Example of standalone tool initialization
paper_search_tool = ResearchTools().search_papers
pdf_tool = ResearchTools().process_pdf
