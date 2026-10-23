import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 1. Load environment variables from the .env file
load_dotenv()

async def main():
    # Retrieve the OpenAI API key securely
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment variables or .env file.")

    # 2. Configure the model client using the modern extension client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=api_key,
    )

    # 3. Define specialized team agents
    writer_agent = AssistantAgent(
        name="Writer",
        model_client=model_client,
        system_message=(
            "You are a technical content writer. "
            "Write clear, structured, and concise technical drafts based on user prompts."
        )
    )

    observer_agent = AssistantAgent(
        name="Observer",
        model_client=model_client,
        system_message=(
            "You are an expert technical reviewer and observer. "
            "Critically analyze the writer's draft for accuracy, clarity, and completeness. "
            "Provide constructive feedback if changes are needed, or output the exact word 'APPROVED' "
            "once the content meets top-tier standards."
        )
    )

    # 4. Define a termination condition that stops the team workflow when 'APPROVED' is mentioned
    termination = TextMentionTermination("APPROVED")

    # 5. Create a Team workflow combining the participants
    team = RoundRobinGroupChat(
        participants=[writer_agent, observer_agent],
        termination_condition=termination
    )

    # 6. Execute the team workflow with a live console output stream
    task = "Write a short architectural overview explaining why multi-agent AI systems are useful."
    print(f"=== Starting AutoGen Team Workflow ===\nTask: {task}\n")
    
    await Console(team.run_stream(task=task))

if __name__ == "__main__":
    asyncio.run(main())