import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get the key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Check if it exists
if not GROQ_API_KEY:
    print(
        "⚠️ WARNING: GROQ_API_KEY is missing from .env file. The AI Compliance Check will not work."
    )
if not REPLICATE_API_TOKEN:
    print(
        "⚠️ WARNING: REPLICATE_API_TOKEN is missing from .env file. The AI Compliance Check will not work."
    )
