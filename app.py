import streamlit as st
from crewai import Task, Crew
from langchain_groq import ChatGroq

from agents import get_agents
from debate_agents import get_debate_agents
from tools import extract_multiple_pdfs, fetch_arxiv_paper
from vector_store import create_vector_store, retrieve_context

# =========================
# LLM
# =========================
groq_api_key = st.secrets["GROQ_API_KEY"]
#llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.3-70b-versatile")
llm = "groq/llama-3.3-70b-versatile"
st.title("📚 AI Research Assistant (RAG + Debate AI)")

uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True)
arxiv_links = st.text_area("arXiv Links")

domain = st.text_input("Domain")
focus = st.text_input("Focus Area")
query = st.text_input("Ask Research Question")

if st.button("Run"):

    texts = []

    # PDFs
    if uploaded_files:
        texts.append(extract_multiple_pdfs(uploaded_files))

    # arXiv
    if arxiv_links:
        for link in arxiv_links.split("\n"):
            if link.strip():
                texts.append(fetch_arxiv_paper(link.strip()))

    combined_text = "\n\n".join(texts)

    # =========================
    # 🔍 RAG
    # =========================
    vector_db = create_vector_store(combined_text)

    context = retrieve_context(
        vector_db,
        query if query else f"{domain} {focus}"
    )

    # =========================
    # 🤖 AGENTS
    # =========================
    (
        understanding_agent,
        gap_agent,
        comparison_agent,
        writer_agent,
        idea_agent
    ) = get_agents(llm)

    proposer, critic, refiner = get_debate_agents(llm)

    # =========================
    # TASKS
    # =========================
    t1 = Task(
        description=f"Analyze papers:\n{context}",
        agent=understanding_agent
    )

    t2 = Task(
        description=f"Find research gaps:\n{context}",
        agent=gap_agent
    )

    t3 = Task(
        description=f"Compare approaches:\n{context}",
        agent=comparison_agent
    )

    t4 = Task(
        description=f"""
DOMAIN: {domain}
FOCUS: {focus}

Generate novel research idea
""",
        agent=idea_agent
    )

    # Debate
    d1 = Task(description="Propose idea", agent=proposer)
    d2 = Task(description="Criticize idea", agent=critic)
    d3 = Task(description="Refine idea", agent=refiner)

    t5 = Task(
        description=f"""
Write:
- Abstract
- Introduction
- Problem Statement
- Method
""",
        agent=writer_agent
    )

    crew = Crew(
        agents=[
            understanding_agent,
            gap_agent,
            comparison_agent,
            idea_agent,
            proposer,
            critic,
            refiner,
            writer_agent
        ],
        tasks=[t1, t2, t3, t4, d1, d2, d3, t5],
        verbose=True
    )

    result = crew.kickoff()

    st.subheader("📊 Output")
    st.write(result)
