import asyncio
import os
from dotenv import load_dotenv

# AutoGen v0.4 Imports
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 1. Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

async def main():
    # 2. Configure the Model Client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key,
    )

    # 3. Define the Agents
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="You are a helpful and concise AI python assistant.",
    )

    user_proxy = UserProxyAgent(
        name="user_proxy"
    )

    # 4. Create a Team/Group Chat
    team = RoundRobinGroupChat([assistant, user_proxy])

    # 5. Run the Team with a prompt
    stream = team.run_stream(task="Write a short 3-line python script to check system memory usage.")
    
    async for message in stream:
        print(message)

if __name__ == "__main__":
    asyncio.run(main())