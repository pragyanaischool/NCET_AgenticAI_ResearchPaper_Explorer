import os
from langchain_groq import ChatGroq
from langchain_community.utilities import SerpAPIWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# CORRECT IMPORT FOR 2026:
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

class ResearchNodes:
    def __init__(self):
        self.llm = ChatGroq(model_name="llama3-70b-8192")
        self.search = SerpAPIWrapper()

    def researcher(self, state):
        """Node for gathering data."""
        topic = state["topic"]
        results = self.search.run(f"Latest papers on {topic}")
        return {"messages": [f"Researcher found: {results[:200]}..."], "data": results}

    def analyst(self, state):
        """Node for summarizing data."""
        data = state["data"]
        summary = self.llm.invoke(f"Summarize this research: {data}")
        return {"messages": [f"Analysis complete."], "summary": summary.content}
