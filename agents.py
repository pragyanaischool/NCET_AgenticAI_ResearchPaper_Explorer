import os
from langchain_groq import ChatGroq
from langchain_community.utilities import SerpAPIWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

class ResearchAgents:
    def __init__(self):
        # Initialize the LLM (Groq)
        self.llm = ChatGroq(
            temperature=0, 
            model_name="openai/gpt-oss-120b",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        # Initialize Search Tool
        self.search_tool = SerpAPIWrapper()

    def researcher_node(self, state):
        """
        The Research Agent: Searches for papers and extracts content.
        """
        topic = state.get("topic")
        print(f"--- RESEARCHING: {topic} ---")
        
        # 1. Web Search for papers
        search_query = f"latest research papers and PDF links for {topic}"
        raw_results = self.search_tool.run(search_query)
        
        # 2. Update state with raw findings
        return {
            "research_notes": [raw_results],
            "status": "analyzing"
        }

    def analyst_node(self, state):
        """
        The Analyst Agent: Synthesizes search results into a structured format.
        """
        notes = "\n".join(state.get("research_notes", []))
        topic = state.get("topic")
        print(f"--- ANALYZING FINDINGS ---")

        prompt = (
            f"You are a Senior Research Analyst. Based on these raw notes: {notes}, "
            f"provide a structured breakdown of the key breakthroughs in {topic}. "
            f"Focus on methodology and practical applications."
        )
        
        analysis = self.llm.invoke(prompt)
        
        return {
            "final_summary": analysis.content,
            "status": "completed"
        }

    def technical_writer_node(self, state):
        """
        Optional: Converts analysis into a formal LaTeX or Markdown report.
        """
        analysis = state.get("final_summary")
        prompt = f"Convert this analysis into a formal scientific report format: {analysis}"
        
        report = self.llm.invoke(prompt)
        return {"final_summary": report.content}
