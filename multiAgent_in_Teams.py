import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import core AgentChat components from AutoGen v0.4+
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main():
    # 1. Initialize the Model Client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # 2. Define the Agents
    # Writer Agent creates initial draft content
    writer_agent = AssistantAgent(
        name="Writer",
        model_client=model_client,
        system_message=(
            "You are a concise tech copywriter. Draft short, engaging technical "
            "explanations. When the Reviewer approves your work, end your final "
            "message with 'TERMINATE'."
        ),
    )

    # Reviewer Agent evaluates and refines the draft
    reviewer_agent = AssistantAgent(
        name="Reviewer",
        model_client=model_client,
        system_message=(
            "You are a technical editor. Review drafts for clarity and brevity. "
            "If the content looks good and accurate, tell the Writer to approve "
            "and output 'TERMINATE'."
        ),
    )

    # 3. Configure Termination Conditions
    # Stops if an agent mentions 'TERMINATE' or if the conversation hits 6 turns
    text_termination = TextMentionTermination("TERMINATE")
    max_msg_termination = MaxMessageTermination(max_messages=6)
    
    # Combine conditions using the OR (|) operator
    combined_termination = text_termination | max_msg_termination

    # 4. Construct the Team
    # RoundRobinGroupChat orchestrates participants in sequential turns
    team = RoundRobinGroupChat(
        participants=[writer_agent, reviewer_agent],
        termination_condition=combined_termination,
    )

    # 5. Execute the Task with Streamed Output
    task_prompt = "Write a 2-sentence explanation of what a Webhook is for beginners."
    
    print(f"\n--- Starting Task: {task_prompt} ---\n")
    await Console(team.run_stream(task=task_prompt))


if __name__ == "__main__":
    # Run the asynchronous main execution loop
    asyncio.run(main())