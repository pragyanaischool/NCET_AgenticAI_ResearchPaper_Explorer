from crewai import Agent

def get_agents(llm):

    understanding_agent = Agent(
        role="Research Paper Analyst",
        goal="Extract structured insights from research papers",
        backstory="Expert academic researcher",
        llm=llm,
        verbose=True
    )

    gap_agent = Agent(
        role="Research Gap Expert",
        goal="Identify research gaps and limitations",
        backstory="Critical thinker and reviewer",
        llm=llm,
        verbose=True
    )

    comparison_agent = Agent(
        role="Cross Paper Analyst",
        goal="Compare multiple research papers",
        backstory="Senior AI researcher",
        llm=llm,
        verbose=True
    )

    writer_agent = Agent(
        role="Research Writer",
        goal="Write academic paper sections",
        backstory="IEEE paper expert",
        llm=llm,
        verbose=True
    )

    idea_agent = Agent(
        role="Innovator",
        goal="Generate novel research ideas",
        backstory="Creative AI scientist",
        llm=llm,
        verbose=True
    )

    return (
        understanding_agent,
        gap_agent,
        comparison_agent,
        writer_agent,
        idea_agent
    )
