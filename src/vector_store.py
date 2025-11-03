import numpy as np
from typing import List, Dict, Any, Tuple
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from .config import Config

class VectorStore:
    def __init__(self, config: Config):
        self.config = config
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
    
    def build_index(self, chunks: List[Dict[str, Any]]):
        """Build FAISS index from chunks and save it"""
        self.chunks = chunks
        self.chunk_metadata = [chunk['metadata'] for chunk in chunks]
        
        # Generate embeddings
        texts = [chunk['content'] for chunk in chunks]
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        
        # Save the index and metadata
        self._save_index()
        
        print(f"Built FAISS index with {len(chunks)} chunks")
    
    def load_index(self) -> bool:
        """Load existing FAISS index if available"""
        index_path = os.path.join(self.config.VECTOR_DB_PATH, "index.faiss")
        metadata_path = os.path.join(self.config.VECTOR_DB_PATH, "metadata.pkl")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.chunks = data['chunks']
                    self.chunk_metadata = data['metadata']
                
                print(f"Loaded existing vector store with {len(self.chunks)} chunks")
                return True
            except Exception as e:
                print(f"Error loading vector store: {e}")
                return False
        return False
    
    def _save_index(self):
        """Save FAISS index and metadata"""
        os.makedirs(self.config.VECTOR_DB_PATH, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(self.config.VECTOR_DB_PATH, "index.faiss")
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata_path = os.path.join(self.config.VECTOR_DB_PATH, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.chunk_metadata
            }, f)
        
        print(f"Vector store saved to {self.config.VECTOR_DB_PATH}")
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar chunks"""
        if top_k is None:
            top_k = self.config.TOP_K
        
        if self.index is None:
            print("Index not built yet.")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search for more results initially to ensure we get some
        search_k = min(top_k * 3, len(self.chunks))
        scores, indices = self.index.search(query_embedding.astype(np.float32), search_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks) and score >= self.config.SIMILARITY_THRESHOLD:
                results.append((self.chunks[idx], float(score)))
            
            if len(results) >= top_k:
                break
        
        # If no results meet threshold, return the top ones anyway
        if not results and len(indices[0]) > 0:
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunks) and i < top_k:
                    results.append((self.chunks[idx], float(score)))
        
        return results