import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load Environment Variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)


# ==========================================
# 2. State Management Functions (Save, Read, Load)
# ==========================================

def save_state_to_file(filepath: str, state_data: dict):
    """Writes agent conversation history and metadata to a JSON file."""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=4, ensure_ascii=False)
        print(f"\n[State Management] State successfully saved to '{filepath}'.")
    except Exception as e:
        print(f"\n[State Management] Error saving state: {e}")


def read_state_from_file(filepath: str) -> dict:
    """Reads and returns state data from a JSON file."""
    if not os.path.exists(filepath):
        print(f"\n[State Management] File '{filepath}' does not exist.")
        return {}
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            state_data = json.load(f)
        print(f"\n[State Management] State successfully read from '{filepath}'.")
        return state_data
    except Exception as e:
        print(f"\n[State Management] Error reading state: {e}")
        return {}


# ==========================================
# 3. Define the Agent Class
# ==========================================

class SimpleAgent:
    def __init__(self, name: str, system_message: str, model: str = "gpt-4o"):
        self.name = name
        self.system_message = system_message
        self.model = model
        # The conversation history acts as the agent's state memory
        self.conversation_history = [
            {"role": "system", "content": self.system_message}
        ]

    def run_task(self, user_prompt: str) -> str:
        """Sends a task to the model, updates history, and returns the response."""
        self.conversation_history.append({"role": "user", "content": user_prompt})
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.3
            )
            reply = response.choices[0].message.content
            
            # Append assistant response to history
            self.conversation_history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            error_msg = f"API Error: {e}"
            print(error_msg)
            return error_msg

    def get_state(self) -> dict:
        """Exports the agent's current configuration and message history."""
        return {
            "name": self.name,
            "system_message": self.system_message,
            "conversation_history": self.conversation_history
        }

    def load_state(self, state_dict: dict):
        """Loads state into the agent."""
        self.name = state_dict.get("name", self.name)
        self.system_message = state_dict.get("system_message", self.system_message)
        self.conversation_history = state_dict.get("conversation_history", self.conversation_history)


# ==========================================
# 4. Main Workflow Execution
# ==========================================

if __name__ == "__main__":
    state_file_path = "agent_state.json"

    # --- Step 1: Initialize Agent 1 (Researcher) & Run Task ---
    print("--- STEP 1: Running Task with Researcher Agent ---")
    researcher = SimpleAgent(
        name="ResearcherAgent",
        system_message="You are an expert technology researcher. Provide clear, concise outline summaries."
    )
    
    task = "Research the top 3 benefits of adopting Microservices Architecture."
    print(f"Task: {task}\n")
    
    research_output = researcher.run_task(task)
    print(f"[{researcher.name} Output]:\n{research_output}\n")

    # --- Step 2: Save State to File ---
    print("--- STEP 2: Saving Agent State ---")
    current_state = researcher.get_state()
    save_state_to_file(state_file_path, current_state)

    # --- Step 3: Read State from File ---
    print("--- STEP 3: Reading State from File ---")
    loaded_state_data = read_state_from_file(state_file_path)
    print(f"Loaded {len(loaded_state_data.get('conversation_history', []))} messages from saved file.")

    # --- Step 4: Load State into Another Agent (Editor Agent) ---
    print("\n--- STEP 4: Passing State to Editor Agent ---")
    
    editor = SimpleAgent(
        name="EditorAgent",
        system_message="You are a professional editor. You take research notes and polish them into executive memos."
    )
    
    # Load the previous state/history into the editor agent
    editor.load_state(loaded_state_data)
    # Optionally update its system prompt to reflect its new role while keeping prior context
    editor.system_message = "You are a professional editor. Polish prior research into an executive summary memo."
    # Update system message in history if needed
    editor.conversation_history[0] = {"role": "system", "content": editor.system_message}

    # Ask the Editor agent to continue using the loaded history
    editor_prompt = "Using the research gathered above, rewrite this into an executive summary memo for leadership."
    print(f"Prompting Editor with context history...\n")
    
    final_output = editor.run_task(editor_prompt)
    print(f"[{editor.name} Output]:\n{final_output}")
    
    print("\nWorkflow completed successfully!")