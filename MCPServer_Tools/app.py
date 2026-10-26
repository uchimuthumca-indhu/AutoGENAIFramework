import asyncio
import os
import sys
from dotenv import load_dotenv

# 1. Load environment variables from .env file
load_dotenv()

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools

async def main():
    # 2. Use sys.executable to ensure the subprocess uses the active virtual environment
    server_params = StdioServerParams(
        command=sys.executable,
        args=["mcp_server.py"]
    )

    print("Connecting to MCP server and loading tools...")
    # 3. Automatically discover and fetch tools from the MCP server
    mcp_tools = await mcp_server_tools(server_params)
    print(f"Successfully loaded tools: {[t.name for t in mcp_tools]}")

    # 4. Initialize the LLM client (picks up OPENAI_API_KEY from environment)
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    # 5. Create an AutoGen Assistant Agent configured with the MCP tools
    assistant = AssistantAgent(
        name="ElectronicPriceAssistant",
        model_client=model_client,
        tools=mcp_tools,
        system_message="You are a helpful shopping assistant that uses available tools to check electronic device prices, stock levels, and specifications."
    )

    # 6. Define a user query for the agent team
    task = "Can you check the price and stock details for an iPhone 16 Pro and a MacBook Pro 16?"
    print(f"\n--- Running Task: {task} ---\n")

    # Execute the agent workflow using run_stream and stream output to the console
    await Console(assistant.run_stream(task=task))

if __name__ == "__main__":
    asyncio.run(main())