import asyncio
import os
from dotenv import load_dotenv
from ddgs import DDGS
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Load environment variables from a .env file
load_dotenv()

def search_weather_news(city: str) -> str:
    """Search for today's weather news and current conditions for the specified city."""
    return str(DDGS().news(query=f"{city} weather today", max_results=5))

# Ask the user for the target city dynamically
city = input("Which city? ")

# Initialize the OpenAI model client (automatically reads OPENAI_API_KEY from environment/.env)
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

# Configure the AssistantAgent with the weather tool and reflection enabled
agent = AssistantAgent(
    name="weather_news",
    model_client=model_client,
    tools=[search_weather_news],
    system_message=(
        "You are a concise weather news assistant. "
        "When given a city, search for its weather today and "
        "reply in exactly 3 short lines summarizing the conditions."
    ),
    reflect_on_tool_use=True
)

async def main():
    # Run the agent stream and display output via the Console UI
    await Console(agent.run_stream(task=f"What is today's weather news for {city}?"))
    
    # Properly close the model client at the end of execution
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())