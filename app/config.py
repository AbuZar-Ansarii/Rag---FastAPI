import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    EMBEDDING_MODEL = "models/embedding-001"
    LLM_MODEL = "gemini-1.5-flash"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    # Use absolute paths for directories
    PERSIST_DIRECTORY = str((Path(__file__).parent.parent / "db").resolve())
    DOCUMENTS_DIR = str((Path(__file__).parent.parent / "documents").resolve())

    def validate(self):
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in environment variables.")

settings = Settings()
