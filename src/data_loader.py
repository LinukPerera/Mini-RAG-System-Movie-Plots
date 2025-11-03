import json
import os
from typing import List, Dict, Any
from .config import Config

class DataLoader:
    def __init__(self, config: Config):
        self.config = config
    
    def load_and_preprocess(self) -> List[Dict[str, Any]]:
        """Load and preprocess movie data"""
        try:
            with open(self.config.DATA_PATH, 'r', encoding='utf-8') as f:
                movies = json.load(f)
        except FileNotFoundError:
            print(f"Error: Data file {self.config.DATA_PATH} not found.")
            print("Please make sure you have run the filtering script and the file exists.")
            return []
        
        # Limit to max documents
        movies = movies[:self.config.MAX_DOCUMENTS]
        
        # Preprocess: combine title and extract for better context
        processed_movies = []
        for movie in movies:
            if 'extract' in movie and movie['extract']:
                # Create enhanced content with more context
                year = movie.get('year', 'Unknown')
                genres = movie.get('genres', [])
                title = movie.get('title', 'Unknown')
                
                content = f"Movie: {title}. Year: {year}. "
                if genres:
                    content += f"Genres: {', '.join(genres)}. "
                content += f"Plot: {movie['extract']}"
                
                processed_movies.append({
                    'title': title,
                    'year': year,
                    'content': content,
                    'genres': genres,
                    'metadata': {
                        'year': year,
                        'genres': genres,
                        'title': title
                    }
                })
        
        print(f"Processed {len(processed_movies)} movies with substantial content")
        return processed_movies