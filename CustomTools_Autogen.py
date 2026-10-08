import asyncio
import os
from typing import Annotated
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.base import Response
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Load environment variables
load_dotenv()

# ==========================================
# 1. CUSTOM TOOLS
# ==========================================

def check_blood_pressure(
    patient_id: Annotated[str, "The unique identifier of the patient"]
) -> str:
    """Checks the blood pressure of a patient."""
    return f"Patient {patient_id} Blood Pressure: 120/80 mmHg (Normal)"

def check_heart_rate(
    patient_id: Annotated[str, "The unique identifier of the patient"]
) -> str:
    """Checks the current heart rate of a patient."""
    return f"Patient {patient_id} Heart Rate: 72 bpm (Normal)"

def check_oxygen_level(
    patient_id: Annotated[str, "The unique identifier of the patient"]
) -> str:
    """Checks the blood oxygen level (SpO2) of a patient."""
    return f"Patient {patient_id} Oxygen Level: 98% (Normal)"

# ==========================================
# 2. OBSERVED AGENT (Fixed Event Listener)
# ==========================================

class ObservedAgent:
    """Observer pattern wrapper to monitor and log agent events safely."""
    
    def __init__(self, agent: AssistantAgent):
        self.agent = agent
        
    async def run_monitored_task(self, task_message: TextMessage):
        print("==========================================")
        print(f"[OBSERVER LOG] Monitoring Task for Agent: '{self.agent.name}'")
        print(f"[OBSERVER LOG] Incoming Message: '{task_message.content}'")
        print("==========================================")
        
        async for event in self.agent.on_messages_stream([task_message], cancellation_token=None):
            # 1. Handle Response object (The final completion result)
            if isinstance(event, Response):
                print("\n[OBSERVED EVENT] Final Response:")
                print(event.chat_message.content)
            
            # 2. Handle Message events (User, Assistant, Tool calls, Tool executions)
            elif hasattr(event, "source"):
                print(f"\n[OBSERVED EVENT] {event.source}:")
                if hasattr(event, "content"):
                    print(event.content)
                elif hasattr(event, "thought"):
                    print(f"Thought: {event.thought}")
            
            # 3. Fallback for any other event structural types
            else:
                print(f"\n[OBSERVED EVENT]: {event}")

        print("\n==========================================")
        print("[OBSERVER LOG] Task completed successfully.")
        print("==========================================")


# ==========================================
# 3. MAIN WORKFLOW
# ==========================================

async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    health_assistant = AssistantAgent(
        name="HealthCheckAssistant",
        model_client=model_client,
        tools=[
            check_blood_pressure,
            check_heart_rate,
            check_oxygen_level
        ],
        system_message=(
            "You are a medical triage assistant. When asked about a patient's vital signs, "
            "use the available tools to check blood pressure, heart rate, and oxygen levels, "
            "and summarize their overall status."
        )
    )

    observed_assistant = ObservedAgent(health_assistant)

    task_input = TextMessage(
        content="Please run a full vital check (blood pressure, heart rate, oxygen level) for patient 'P-104' and provide a summary.",
        source="user"
    )

    await observed_assistant.run_monitored_task(task_input)

if __name__ == "__main__":
    asyncio.run(main())