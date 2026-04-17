import streamlit as st
from crewai import Task, Crew
from langchain_groq import ChatGroq

from agents import get_agents
from debate_agents import get_debate_agents
from tools import extract_pdf_text, fetch_arxiv_paper
from vector_store import create_vector_store, retrieve_context

# LLM
groq_api_key = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.3-70b-versatile")

st.title("📚 AI Research Assistant (RAG + Debate Mode)")

uploaded_files = st.file_uploader("Upload Research Papers", accept_multiple_files=True)
arxiv_links = st.text_area("Enter arXiv Links (one per line)")

domain = st.text_input("Research Domain")
focus = st.text_input("Focus Area")

query = st.text_input("Ask Specific Research Question (for RAG retrieval)")

if st.button("Run Analysis"):

    texts = []

    # PDF ingestion
    if uploaded_files:
        for file in uploaded_files:
            texts.append(extract_pdf_text(file))

    # arXiv ingestion
    if arxiv_links:
        for link in arxiv_links.split("\n"):
            if link.strip():
                texts.append(fetch_arxiv_paper(link.strip()))

    combined_text = "\n\n".join(texts)

    # =========================
    # 🔍 VECTOR STORE CREATION
    # =========================
    vector_db = create_vector_store(combined_text)

    # Retrieve context dynamically
    retrieved_context = retrieve_context(
        vector_db,
        query if query else f"{domain} {focus}"
    )

    # =========================
    # 🤖 BASE AGENTS
    # =========================
    (
        ingestion_agent,
        understanding_agent,
        gap_agent,
        comparison_agent,
        writer_agent,
        idea_agent
    ) = get_agents(llm)

    # =========================
    # 🧠 CORE TASKS (WITH RAG)
    # =========================
    task1 = Task(
        description=f"""
Analyze research papers using ONLY this context:

{retrieved_context}

Extract:
- Problem
- Methods
- Results
- Limitations
""",
        agent=understanding_agent
    )

    task2 = Task(
        description=f"""
Using this context:

{retrieved_context}

Find deep research gaps.
""",
        agent=gap_agent
    )

    task3 = Task(
        description=f"""
Compare and synthesize insights from:

{retrieved_context}
""",
        agent=comparison_agent
    )

    # =========================
    # 💡 IDEA GENERATION
    # =========================
    idea_task = Task(
        description=f"""
Based on gaps + domain:

DOMAIN: {domain}
FOCUS: {focus}

Generate ONE strong novel research idea.
""",
        agent=idea_agent
    )

    # =========================
    # ⚔️ DEBATE MODE
    # =========================
    proposer, critic, refiner = get_debate_agents(llm)

    debate_task1 = Task(
        description="Propose a breakthrough research idea",
        agent=proposer
    )

    debate_task2 = Task(
        description="Critically evaluate the proposed idea. Be strict.",
        agent=critic
    )

    debate_task3 = Task(
        description="Refine the idea to make it publishable",
        agent=refiner
    )

    # =========================
    # ✍️ WRITING TASK
    # =========================
    task4 = Task(
        description=f"""
Write research draft:

DOMAIN: {domain}
FOCUS: {focus}

Include:
- Abstract
- Introduction
- Problem Statement
- Proposed Method
""",
        agent=writer_agent
    )

    # =========================
    # 🧠 CREW EXECUTION
    # =========================
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
        tasks=[
            task1,
            task2,
            task3,
            idea_task,
            debate_task1,
            debate_task2,
            debate_task3,
            task4
        ],
        verbose=True
    )

    result = crew.kickoff()

    st.subheader("📊 Research Output")
    st.write(result)
