import streamlit as st
from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import research_node, analysis_node # Assume analysis_node is defined similarly

# Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("researcher", research_node)
workflow.add_node("analyzer", analysis_node)

# Define the sequence
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "analyzer")
workflow.add_edge("analyzer", END)

# Compile
app_engine = workflow.compile()

# Streamlit UI
st.title("NCET AgenticAI Explorer")
user_input = st.text_input("Enter topic:")

if st.button("Explore"):
    initial_state = {"topic": user_input, "messages": [], "research_data": [], "status": "Started"}
    # Run the graph
    config = {"configurable": {"thread_id": "1"}}
    final_state = app_engine.invoke(initial_state, config)
    st.write(final_state['messages'][-1].content)
