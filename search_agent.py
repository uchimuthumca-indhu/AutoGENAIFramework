import asyncio
import os
from typing import Annotated

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from duckduckgo_search import DDGS

# Load environment variables from .env file
load_dotenv()


# ==========================================
# 1. CUSTOM TOOL: DUCKDUCKGO WEB SEARCH
# ==========================================
def duckduckgo_search(
    query: Annotated[str, "The search query term or phrase to search on DuckDuckGo."]
) -> str:
    """Performs a live web search using DuckDuckGo and returns formatted top results."""
    try:
        results = list(DDGS().text(keywords=query, max_results=5))
        if not results:
            return "No search results found."

        formatted_results = []
        for idx, res in enumerate(results, 1):
            title = res.get("title", "No Title")
            snippet = res.get("body", "No Description")
            url = res.get("href", "")
            formatted_results.append(f"{idx}. {title}\n   Snippet: {snippet}\n   URL: {url}")

        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Error executing DuckDuckGo search: {str(e)}"


# ==========================================
# 2. MAIN ASYNC AGENT WORKFLOW
# ==========================================
async def main():
    # Ensure API key is loaded
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set. Check your .env file.")

    # Initialize the Model Client with reflect_on_tool_use=True
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        reflect_on_tool_use=True  # Enables explicit reasoning step after tool calls
    )

    # Initialize AssistantAgent with custom tools
    research_agent = AssistantAgent(
        name="web_researcher",
        model_client=model_client,
        system_message=(
            "You are a web research assistant. When asked for current information, search "
            "the web using `duckduckgo_search`. Reflect carefully on the search results "
            "to ensure accuracy before providing your final response."
        ),
        tools=[duckduckgo_search],
    )

    # Task definition
    task = "Search for the latest news on OpenAI's recent model releases and summarize the main updates."

    print("--- Starting AutoGen Web Search with Tool Reflection ---\n")

    # Stream agent execution using Console UI
    await Console(research_agent.run_stream(task=task))

    # Clean up model client session
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())