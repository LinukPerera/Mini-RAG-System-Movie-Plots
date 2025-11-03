import sys
import os
import json
import getpass
from typing import List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rag_system import MovieRAGSystem
from src.config import Config

def ensure_results_dir():
    """Ensure results directory exists"""
    os.makedirs("results", exist_ok=True)

def get_user_queries() -> List[str]:
    """Get queries from user input"""
    print("\nMovie RAG Query System")
    print("=" * 50)
    print("Enter your questions about movies (one per line).")
    print("Press Enter twice when finished.\n")
    
    queries = []
    print("Enter your queries:")
    
    while True:
        try:
            line = input().strip()
            if not line:
                if queries:
                    break
                else:
                    print("Please enter at least one query.")
                    continue
            queries.append(line)
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)
    
    return queries

def get_default_queries() -> List[str]:
    """Return default queries if user doesn't provide any"""
    return [
        "Which movies feature artificial intelligence or robots?",
        "Tell me about movies that involve dreams or virtual reality",
        "What are some good sci-fi movies from the 1990s?",
        "Find movies with prison escape plots",
        "Movies about technology or computers",
        "Which films have superheroes or superpowers?",
        "Romantic comedies from the 2000s"
    ]

def main():
    """Main entry point for the Movie RAG system"""
    print("🎥 Movie Plot RAG System")
    print("=" * 50)
    
    # Ensure results directory exists
    ensure_results_dir()
    
    # Initialize configuration (this will handle API key prompting)
    config = Config()
    
    # Create RAG system
    rag_system = MovieRAGSystem(config)
    
    # Get queries from user or use defaults
    queries = get_user_queries()
    if not queries:
        print("\nUsing default queries...")
        queries = get_default_queries()
    
    print(f"\nProcessing {len(queries)} queries...")
    print("=" * 50)
    
    # Process each query
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 50)
        
        # Get response from RAG system
        response = rag_system.query(query)
        
        # Display results
        print(f"Answer: {response['answer']}")
        print(f"Reasoning: {response['reasoning']}")
        print(f"Retrieved {len(response['contexts'])} context chunks")
        
        # Save results
        output_file = f"results/query_{i:02d}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(response, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved to: {output_file}")
        print("=" * 50)
    
    print(f"\nAll done! Processed {len(queries)} queries.")
    print("Results saved in the 'results/' directory")

if __name__ == "__main__":
    main()