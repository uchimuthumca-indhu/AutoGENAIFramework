import asyncio
import os
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 1. Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

async def main():
    # 2. Initialize Model Client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key,
    )

    # 3. Define Team Agents with specific roles
    writer = AssistantAgent(
        name="writer",
        model_client=model_client,
        system_message=(
            "You are a technical writer. Draft concise explanations based on the topic. "
            "Pass your work to the reviewer for feedback."
        ),
    )

    reviewer = AssistantAgent(
        name="reviewer",
        model_client=model_client,
        system_message=(
            "You are a reviewer. Review the content provided by the writer for accuracy and clarity. "
            "If everything looks good, approve it and respond with 'TERMINATE'. "
            "If changes are needed, provide brief feedback."
        ),
    )

    # 4. Set up Termination Condition (Stops when 'TERMINATE' is output by any agent)
    termination_condition = TextMentionTermination("TERMINATE")

    # 5. Create the Team (RoundRobinGroupChat switches turns between writer and reviewer)
    team = RoundRobinGroupChat(
        participants=[writer, reviewer],
        termination_condition=termination_condition,
    )

    # 6. Run the Team with Console UI streaming
    task = "Write a 2-sentence summary explaining microservices architecture."
    print(f"Starting Team Task: {task}\n" + "-" * 50)
    
    await Console(team.run_stream(task=task))

if __name__ == "__main__":
    asyncio.run(main())