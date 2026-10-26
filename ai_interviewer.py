import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Load environment variables from the .env file
load_dotenv()

async def main():
    # 1. Configure LLM Model Client (Automatically grabs OPENAI_API_KEY from .env)
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # 2. Define AI Interviewer Agent
    interviewer = AssistantAgent(
        name="AI_Interviewer",
        model_client=model_client,
        system_message=(
            "You are a professional technical AI interviewer. "
            "Your task is to ask the candidate EXACTLY 3 interview questions, one at a time. "
            "Wait for the candidate's response after each question. "
            "After the candidate answers the 3rd and final question, give a brief closing remark and say explicitly 'Interview Complete. TERMINATE'."
        ),
    )

    # 3. Define User Proxy Agent (Human-in-the-Loop via Console)
    user_proxy = UserProxyAgent(
        name="Candidate",
        description="A human candidate answering interview questions.",
    )

    # 4. Define Termination Condition
    termination = TextMentionTermination("TERMINATE")

    # 5. Create Team / Orchestrator
    team = RoundRobinGroupChat(
        participants=[interviewer, user_proxy],
        termination_condition=termination
    )

    # 6. Run Interactive Conversation with Console UI
    print("=== AI Interviewer Application Started ===")
    print("The interviewer will ask you 3 questions. Type your answers when prompted in the console.\n")

    await Console(team.run_stream(task="Hello! I am ready for the interview. Please ask your first question."))

if __name__ == "__main__":
    asyncio.run(main())