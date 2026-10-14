import os
from dotenv import load_dotenv

def load_environment():
    """Loads environment variables from a .env file."""
    load_dotenv()
    
    # Example check for an API key (e.g., OpenAI or similar LLM provider)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[WARNING] OPENAI_API_KEY not found in environment variables. Ensure your .env file is configured.")
    else:
        print("[INFO] Environment variables loaded successfully.")