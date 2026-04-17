from crewai import Agent

def get_debate_agents(llm):

    proposer = Agent(
        role="Idea Proposer",
        goal="Propose a novel research idea",
        backstory="Breakthrough innovator",
        llm=llm,
        verbose=True
    )

    critic = Agent(
        role="Reviewer",
        goal="Critically analyze and reject weak ideas",
        backstory="Strict academic reviewer",
        llm=llm,
        verbose=True
    )

    refiner = Agent(
        role="Refiner",
        goal="Improve the idea to make it publishable",
        backstory="Expert in refining research",
        llm=llm,
        verbose=True
    )

    return proposer, critic, refiner
