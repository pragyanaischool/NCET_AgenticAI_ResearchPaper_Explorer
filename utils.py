import os
import re
from typing import List
import streamlit as st
from langchain_core.messages import BaseMessage

class ResearchUtils:
    @staticmethod
    def validate_api_keys():
        """Ensures all required secrets are present before running the graph."""
        required_keys = ["GROQ_API_KEY", "SERPAPI_API_KEY"]
        missing_keys = [key for key in required_keys if not os.getenv(key) and key not in st.secrets]
        
        if missing_keys:
            st.error(f"Missing API Keys: {', '.join(missing_keys)}. Please add them to your .env or Streamlit Secrets.")
            st.stop()

    @staticmethod
    def format_debate_history(messages: List[BaseMessage]) -> str:
        """Converts LangGraph message state into a readable string for the UI."""
        formatted_history = ""
        for msg in messages:
            # Extracts the prefix (PROPONENT/OPPONENT) we added in the agents file
            formatted_history += f"{msg.content}\n\n"
        return formatted_history

    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Utility to find PDF or ResearchGate URLs from raw search strings."""
        url_pattern = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w.-]*\.pdf')
        return url_pattern.findall(text)

    @staticmethod
    def clean_text(text: str) -> str:
        """Removes excessive whitespace and citations for cleaner LLM processing."""
        text = re.sub(r'\[\d+\]', '', text)  # Remove [1], [2] style citations
        text = re.sub(r'\s+', ' ', text)     # Collapse multiple spaces/newlines
        return text.strip()

def get_session_id():
    """Generates a unique ID for LangGraph thread persistence if needed."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = os.urandom(8).hex()
    return st.session_state.session_id
