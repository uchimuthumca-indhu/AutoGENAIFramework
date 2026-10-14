class Agent:
    def __init__(self, name: str, role: str, goal: str, backstory: str = ""):
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory

    def execute_task(self, task_description: str) -> str:
        """Simulates agent execution of a given task."""
        # In a real framework, this would call an LLM client (e.g., OpenAI, Anthropic)
        response = f"[{self.name} ({self.role})] successfully processed task: '{task_description}' achieving goal: {self.goal}"
        return response

    def __repr__(self):
        return f"Agent(name='{self.name}', role='{self.role}')"