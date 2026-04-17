import os
from typing import Annotated, TypedDict, List
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# --- 1. Define the Debate State ---
class DebateState(TypedDict):
    topic: str
    messages: List[BaseMessage]
    current_speaker: str
    iteration: int

class ResearchDebateGraph:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.7, # Higher temperature for creative debating
            model_name="llama-3.3-70b-versatile",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    # --- Node 1: The Proponent ---
    def proponent_node(self, state: DebateState):
        """Argues IN FAVOR of the research breakthrough or methodology."""
        topic = state['topic']
        chat_history = state.get('messages', [])
        
        prompt = f"""
        You are a Visionary Researcher. Your goal is to argue STONGLY IN FAVOR of: {topic}.
        Highlight the benefits, the innovation, and the positive future impact.
        Keep your response concise but intellectually sharp.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)] + chat_history)
        
        return {
            "messages": [AIMessage(content=f"PROPONENT: {response.content}")],
            "iteration": state['iteration'] + 1,
            "current_speaker": "opponent"
        }

    # --- Node 2: The Opponent ---
    def opponent_node(self, state: DebateState):
        """Argues AGAINST or provides CRITICAL SKEPTICISM regarding the topic."""
        topic = state['topic']
        chat_history = state.get('messages', [])
        
        prompt = f"""
        You are a Critical Peer Reviewer. Your goal is to find flaws or limitations in: {topic}.
        Challenge the assumptions and point out ethical concerns or technical hurdles.
        Be rigorous but professional.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)] + chat_history)
        
        return {
            "messages": [AIMessage(content=f"OPPONENT: {response.content}")],
            "current_speaker": "judge"
        }

    # --- Node 3: The Judge ---
    def judge_node(self, state: DebateState):
        """Synthesizes both sides and reaches a balanced conclusion."""
        chat_history = state.get('messages', [])
        
        prompt = """
        You are a Senior Research Editor. Analyze the debate between the Proponent and Opponent.
        Synthesize their points and provide a final verdict on the validity of the research topic.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)] + chat_history)
        
        return {
            "messages": [AIMessage(content=f"JUDGE VERDICT: {response.content}")],
            "current_speaker": "end"
        }
