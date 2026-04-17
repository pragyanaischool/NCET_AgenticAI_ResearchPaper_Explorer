from typing import Annotated, TypedDict, List, Union
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    # This stores the research topic
    topic: str
    # 'operator.add' allows nodes to append messages to the list rather than overwriting it
    messages: Annotated[List[BaseMessage], operator.add]
    # Stores raw research data/links
    research_data: List[str]
    # Current status for Streamlit progress bars
    status: str
