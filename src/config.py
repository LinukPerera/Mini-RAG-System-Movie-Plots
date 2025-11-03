import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
import getpass

@dataclass
class Config:
    # Data settings
    DATA_PATH: str = "data/sample_movies.json"
    MAX_DOCUMENTS: int = 400
    VECTOR_DB_PATH: str = "data/vector_store"
    
    # Chunking settings
    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 50
    SMART_CHUNKING: bool = True
    
    # Retrieval settings
    TOP_K: int = 3
    SIMILARITY_THRESHOLD: float = 0.3
    
    # Embedding settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # LLM settings
    GEMINI_MODEL: str = "gemini-2.0-flash-exp"
    
    def __post_init__(self):
        """Initialize after dataclass creation"""
        # Load environment variables with error handling
        try:
            load_dotenv()
        except Exception as e:
            print(f"Warning: Could not load .env file: {e}")
    
    @property
    def API_KEY(self) -> str:
        """Get API key from environment or prompt user"""
        api_key = os.getenv('GEMINI_API_KEY')
        
        # Check if API key is set and valid
        if api_key and api_key.startswith('AIzaSy'):
            return api_key
        
        # If we get here, we need to get the API key from user
        print("\nGemini API Key Required")
        print("=" * 40)
        print("To use the full RAG system with AI responses, you need a Gemini API key.")
        print("Get one from: https://aistudio.google.com/app/apikey")
        print("\nEnter your Gemini API key (starts with 'AIzaSy...'):")
        
        api_key = getpass.getpass("API Key: ").strip()
        
        # Validate the key format
        if api_key.startswith('AIzaSy'):
            # Save to .env file for future use
            try:
                with open('.env', 'w') as f:
                    f.write(f"GEMINI_API_KEY={api_key}\n")
                print("API key saved to .env file")
            except Exception as e:
                print(f"Could not save to .env file: {e}")
            return api_key
        else:
            print("Invalid API key format. Continuing with fallback responses.")
            return None