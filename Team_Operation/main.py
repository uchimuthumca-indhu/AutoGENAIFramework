from config import load_environment
from agent import Agent
from task import Task
from team import Team

def main():
    # 1. Load Environment Variables (.env)
    load_environment()

    print("\n--- Initializing AutoGenFramework ---")

    # 2. Define Agents
    researcher = Agent(
        name="DataAnalyzer",
        role="Senior Researcher",
        goal="Gather accurate market data and summarize key insights.",
        backstory="Expert in web research and trend analysis with 5 years of experience."
    )

    writer = Agent(
        name="ContentWriter",
        role="Copywriter",
        goal="Create engaging, professional summaries based on research reports.",
        backstory="Specialized in transforming complex technical data into readable content."
    )

    # 3. Define Tasks
    task1 = Task(
        description="Analyze the latest trends in Generative AI frameworks for 2026.",
        expected_output="A bulleted list of top 3 trends.",
        assigned_agent=researcher
    )

    task2 = Task(
        description="Draft a short blog post summarizing the analyzed trends.",
        expected_output="A 200-word engaging blog article.",
        assigned_agent=writer
    )

    # 4. Create a Team
    ai_team = Team(
        name="AI Innovations Team",
        agents=[researcher, writer],
        tasks=[task1, task2]
    )

    # 5. Team Operations
    
    # Operation A: Run the team workflow
    ai_team.run()

    # Operation B: Watch live logs of the execution
    ai_team.watch_live()

    # Operation C: Reset the team state for a new run
    ai_team.reset()
    
    # Verify reset worked by checking task statuses
    print(f"Task 1 status after reset: {task1.status}")

if __name__ == "__main__":
    main()