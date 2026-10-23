import asyncio
import os
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 1. Load environment variables from the .env file
load_dotenv()


async def main():
  # Verify that the API key is present
  if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY not found. Please check your .env file or environment"
        " variables."
    )

  # Initialize the OpenAI model client
  model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

  # 2. Create the AI Assistant Agent
  assistant_agent = AssistantAgent(
      name="assistant_agent",
      model_client=model_client,
      system_message=(
          "You are a helpful AI copywriter. Your goal is to help the user"
          " draft and refine text content. When the user approves the result or"
          ' says "APPROVE", output "TERMINATE" to conclude the session.'
      ),
  )

  # 3. Create the Human-in-the-Loop User Proxy Agent
  # This pauses execution and prompts the console for human input using input()
  user_proxy_agent = UserProxyAgent(
      name="user_proxy",
      input_func=input,  # Collects live feedback/replies from the human user
  )

  # 4. Set up the Text Mention Termination Condition
  # The conversation will automatically stop when "TERMINATE" is mentioned
  termination_condition = TextMentionTermination("TERMINATE")

  # 5. Create a Team (Group Chat) combining the participants and the termination condition
  team = RoundRobinGroupChat(
      participants=[assistant_agent, user_proxy_agent],
      termination_condition=termination_condition,
  )

  # 6. Define the Task for the agents to solve
  task = (
      "I need a short, professional 2-sentence bio for my LinkedIn profile"
      " highlighting my experience in Python development."
  )

  print(f"--- Starting Task ---\n{task}\n---------------------")

  # Run the team workflow and stream messages live to the console
  await Console(team.run_stream(task=task))

  # Clean up the model client resources
  await model_client.close()


if __name__ == "__main__":
  asyncio.run(main())