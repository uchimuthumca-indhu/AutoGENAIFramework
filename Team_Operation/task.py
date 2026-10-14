class Task:
    def __init__(self, description: str, expected_output: str, assigned_agent=None):
        self.description = description
        self.expected_output = expected_output
        self.assigned_agent = assigned_agent
        self.status = "Pending"  # Pending, In Progress, Completed
        self.output = None

    def execute(self):
        """Executes the task using the assigned agent."""
        if not self.assigned_agent:
            raise ValueError(f"Task '{self.description}' has no agent assigned.")
        
        self.status = "In Progress"
        print(f"-> Starting Task: {self.description}")
        
        # Simulate execution
        self.output = self.assigned_agent.execute_task(self.description)
        self.status = "Completed"
        print(f"-> Completed Task: {self.description}\n")
        return self.output

    def reset(self):
        """Resets the task state."""
        self.status = "Pending"
        self.output = None

    def __repr__(self):
        return f"Task(status='{self.status}', description='{self.description[:30]}...')"