from langchain_groq import ChatGroq
from tools import paper_search_tool  # Our refactored tool
from langchain_core.messages import AIMessage

def research_node(state: AgentState):
    topic = state['topic']
    # 1. Execute Search
    results = paper_search_tool.invoke(topic)
    
    # 2. Return state updates
    return {
        "research_data": [results],
        "messages": [AIMessage(content=f"Research complete for {topic}.")],
        "status": "Analyzing"
    }
