import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# ==========================================
# 1. CUSTOM TOOLS
# ==========================================
def get_company_benefits() -> str:
    """Returns standard company benefits to include in job descriptions."""
    return "Benefits: Remote-first work environment, health insurance, learning allowance, and 20 paid leave days."


def count_lines(text: str) -> int:
    """Counts the number of non-empty lines in a given text string."""
    return len([line for line in text.split("\n") if line.strip()])


# ==========================================
# 2. MAIN ASYNC AGENT WORKFLOW
# ==========================================
async def main():
    # Verify API key is loaded properly
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing. Please check your .env file.")

    # Initialize the Model Client
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    # Initialize the AssistantAgent with Custom Tools and System Message
    hr_assistant = AssistantAgent(
        name="hr_assistant",
        model_client=model_client,
        system_message="You are an HR assistant.",
        tools=[get_company_benefits, count_lines],
    )

    # Define the Task Message
    task = (
        "Write a job description for a Junior Python Developer consisting of EXACTLY 5 numbered lines/bullets. "
        "Line 1 should be the title. Lines 2 to 5 should be key responsibilities."
    )

    print("--- Starting Agent Task Execution ---\n")

    # Run the Agent with Stream Observation using Console UI
    await Console(hr_assistant.run_stream(task=task))

    # Close the model client to prevent script hanging
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())