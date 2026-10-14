import time
from typing import List
from agent import Agent
from task import Task

class Team:
    def __init__(self, name: str, agents: List[Agent], tasks: List[Task]):
        self.name = name
        self.agents = agents
        self.tasks = tasks
        self.execution_logs = []

    def run(self) -> List[str]:
        """Executes all tasks sequentially across team agents."""
        print(f"\n==============================")
        print(f" Team '{self.name}' Execution Started ")
        print(f"==============================\n")
        
        results = []
        for task in self.tasks:
            log_msg = f"Executing task via team workflow..."
            self.execution_logs.append(log_msg)
            
            result = task.execute()
            results.append(result)
            self.execution_logs.append(result)
            
        print(f"Team '{self.name}' Execution Completed Successfully.\n")
        return results

    def watch_live(self):
        """Simulates a live monitoring feed of the team's ongoing activities and logs."""
        print(f"\n--- [WATCH LIVE] Monitoring Team: {self.name} ---")
        if not self.execution_logs:
            print("No execution activity logged yet. Run the team first.")
            return

        for idx, log in enumerate(self.execution_logs, 1):
            print(f"[Live Feed #{idx}] {log}")
            time.sleep(0.3) # Simulating live stream delay
        print("-----------------------------------------------\n")

    def reset(self):
        """Resets all tasks and clears team execution logs."""
        print(f"[*] Resetting Team '{self.name}' state...")
        for task in self.tasks:
            task.reset()
        self.execution_logs.clear()
        print(f"[*] Team '{self.name}' has been successfully reset.\n")