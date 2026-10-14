import asyncio
import os
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 1. Load environment variables
load_dotenv()

async def main():
    # 2. Initialize the Model Client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # 3. Initialize ListMemory
    memory = ListMemory()

    # 4. Add items using direct MemoryContent instantiation
    await memory.add(
        MemoryContent(
            content="User prefers code examples written in Python and concise explanations.",
            mime_type=MemoryMimeType.TEXT,
        )
    )
    await memory.add(
        MemoryContent(
            content="The target project architecture is microservices-based.",
            mime_type=MemoryMimeType.TEXT,
        )
    )

    # 5. Attach memory to the AssistantAgent
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        memory=[memory],
        system_message="You are a helpful technical assistant. Use your memory to answer questions.",
    )

    # 6. Run the agent task
    response = await assistant.run(
        task="What language and architecture preferences should we follow for our next service?"
    )

    for message in response.messages:
        print(f"{message.source}: {message.content}\n")

if __name__ == "__main__":
    asyncio.run(main())