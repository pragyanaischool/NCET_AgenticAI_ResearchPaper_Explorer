import streamlit as st
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from agents import ResearchNodes

# 1. Define State
class GraphState(TypedDict):
    topic: str
    messages: List[str]
    data: str
    summary: str

# 2. Build Graph
nodes = ResearchNodes()
workflow = StateGraph(GraphState)

workflow.add_node("research", nodes.researcher)
workflow.add_node("analyze", nodes.analyst)

workflow.set_entry_point("research")
workflow.add_edge("research", "analyze")
workflow.add_edge("analyze", END)

app = workflow.compile()

# 3. Streamlit UI
st.title("NCET Research Explorer")
query = st.text_input("Topic:")

if st.button("Run"):
    results = app.invoke({"topic": query, "messages": []})
    st.markdown(results["summary"])
