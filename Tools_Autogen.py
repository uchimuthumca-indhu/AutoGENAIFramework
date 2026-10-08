import asyncio
import os

# Disable OpenTelemetry context tracking in AutoGen
os.environ["AUTOGEN_TELEMETRY_ENABLED"] = "false"

import asyncio
from dotenv import load_dotenv

load_dotenv()

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_agentchat.base import TaskResult
from autogen_ext.models.openai import OpenAIChatCompletionClient


# Verification helper
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing. Check your .env file.")


# =====================================================================
# 1. TOOLS: Custom Python functions
# =====================================================================
async def get_weather(city: str) -> str:
    """Fetch current weather for a given city."""
    weather_data = {
        "New York": "72°F, Clear sky",
        "Tokyo": "65°F, Light rain",
        "London": "55°F, Cloudy",
    }
    return weather_data.get(city, f"70°F, Partly cloudy in {city}")


async def calculate_travel_time(distance_miles: float, speed_mph: float = 60.0) -> str:
    """Calculate travel time in hours given distance and speed."""
    if speed_mph <= 0:
        return "Speed must be greater than zero."
    hours = distance_miles / speed_mph
    return f"Estimated travel time: {hours:.2f} hours."


# =====================================================================
# 2. OBSERVED AGENT: Observer Pattern Implementation
# =====================================================================
class ObservedAgent:
    """Wrapper to monitor and log agent stream events."""

    def __init__(self, agent: AssistantAgent):
        self.agent = agent

    async def execute_and_observe(self, task_message: TextMessage) -> TaskResult:
        print(f"\n[Observer Log] Starting task for agent: '{self.agent.name}'")
        print(f"[Observer Log] Task Source: {task_message.source} | Content: '{task_message.content}'\n")

        task_result = None

        # Fully iterate through the stream without early returns inside the loop
        async for event in self.agent.run_stream(task=task_message.content):
            if isinstance(event, BaseChatMessage):
                print(f"  ├─ [{event.source} -> {event.type}]: {getattr(event, 'content', event)}")
            elif isinstance(event, TaskResult):
                task_result = event

        if task_result:
            print(f"\n[Observer Log] Task finished. Total messages exchanged: {len(task_result.messages)}")
            return task_result

        # Fallback if TaskResult wasn't emitted during streaming
        return await self.agent.run(task=task_message.content)


# =====================================================================
# 3. MAIN EXECUTION FLOW
# =====================================================================
async def main():
    # Initializes using OPENAI_API_KEY loaded via dotenv
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    agent = AssistantAgent(
        name="travel_assistant",
        model_client=model_client,
        tools=[get_weather, calculate_travel_time],
        system_message="You are a helpful travel planning assistant.",
    )

    observed_agent = ObservedAgent(agent)

    task_input = TextMessage(
        content="What is the weather in Tokyo, and how long does it take to drive 180 miles at 60 mph?",
        source="user",
    )

    print("=== Executing AutoGen Task ===")
    result = await observed_agent.execute_and_observe(task_input)

    print("\n=== Final Response ===")
    print(result.messages[-1].content)


if __name__ == "__main__":
    asyncio.run(main())