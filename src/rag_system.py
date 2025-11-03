import google.generativeai as genai
import json
import os
from typing import Dict, Any, List
from .data_loader import DataLoader
from .chunker import SmartChunker
from .vector_store import VectorStore
from .config import Config

class MovieRAGSystem:
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.chunker = SmartChunker(config)
        self.vector_store = VectorStore(config)
        self.initialized = False
        
        # Initialize Gemini (will prompt for API key if needed)
        api_key = config.API_KEY
        if api_key:
            genai.configure(api_key=api_key)
            self.llm = genai.GenerativeModel(config.GEMINI_MODEL)
            print("Gemini AI initialized")
        else:
            self.llm = None
            print("Using fallback responses (no Gemini API key)")
    
    def initialize(self) -> bool:
        """Initialize the RAG system, loading existing vector store if available"""
        print("Initializing RAG System...")
        
        # Try to load existing vector store first
        if self.vector_store.load_index():
            self.initialized = True
            return True
        
        # Otherwise, build from scratch
        print("Loading movie data...")
        documents = self.data_loader.load_and_preprocess()
        
        if not documents:
            print("No documents loaded. Please check your data file.")
            return False
        
        print("Chunking documents...")
        chunks = self.chunker.chunk_documents(documents)
        
        print("Building vector index...")
        self.vector_store.build_index(chunks)
        
        self.initialized = True
        print(f"RAG system initialized with {len(chunks)} chunks from {len(documents)} movies")
        return True
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a query and return structured response"""
        if not self.initialized:
            success = self.initialize()
            if not success:
                return {
                    "answer": "System initialization failed. Please check your data file.",
                    "contexts": [],
                    "reasoning": "Initialization error"
                }
        
        # Retrieve relevant chunks
        retrieved_chunks = self.vector_store.search(question, top_k=5)
        
        print(f"🔎 Retrieved {len(retrieved_chunks)} relevant chunks")
        
        if not retrieved_chunks:
            return {
                "answer": "I couldn't find relevant information about this topic in the movie plots database.",
                "contexts": [],
                "reasoning": "No relevant movie plot information was found for the query."
            }
        
        # Prepare context
        contexts = [chunk['content'] for chunk, score in retrieved_chunks]
        context_text = "\n\n".join(contexts)
        
        # Generate answer
        prompt = self._build_prompt(question, context_text)
        
        if self.llm:
            try:
                response = self.llm.generate_content(prompt)
                answer = response.text.strip()
            except Exception as e:
                print(f"Error calling Gemini API: {e}")
                answer = self._generate_fallback_answer(question, retrieved_chunks)
        else:
            answer = self._generate_fallback_answer(question, retrieved_chunks)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(question, retrieved_chunks)
        
        return {
            "answer": answer,
            "contexts": contexts,
            "reasoning": reasoning
        }
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build the prompt for the LLM"""
        return f"""Based on the following movie plot information, answer the question. Be specific and reference the relevant movies.

Movie Plot Context:
{context}

Question: {question}

Please provide a concise answer that directly addresses the question. If the context contains relevant information, reference specific movie titles and plot points.

Answer:"""
    
    def _generate_fallback_answer(self, question: str, retrieved_chunks: List) -> str:
        """Generate answer without LLM"""
        movie_titles = list(set([chunk['metadata']['title'] for chunk, score in retrieved_chunks]))
        
        # Simple keyword-based answer generation
        question_lower = question.lower()
        
        if any(term in question_lower for term in ['artificial intelligence', 'ai', 'robot']):
            return f"Based on the movie plots, these films feature artificial intelligence or robots: {', '.join(movie_titles[:3])}."
        elif any(term in question_lower for term in ['dream', 'dreaming', 'virtual reality']):
            return f"These movies involve dreams or virtual reality: {', '.join(movie_titles[:3])}."
        elif any(term in question_lower for term in ['prison', 'escape', 'jail']):
            return f"These movies feature prison settings or escapes: {', '.join(movie_titles[:3])}."
        elif any(term in question_lower for term in ['sci-fi', 'science fiction', 'future']):
            return f"Science fiction movies include: {', '.join(movie_titles[:3])}."
        elif any(term in question_lower for term in ['superhero', 'superpower', 'marvel', 'dc']):
            return f"Superhero movies include: {', '.join(movie_titles[:3])}."
        elif any(term in question_lower for term in ['romantic', 'romance', 'comedy']):
            return f"Romantic comedies include: {', '.join(movie_titles[:3])}."
        else:
            return f"Based on movie plots including {', '.join(movie_titles[:3])}. The plots discuss various themes related to your question."
    
    def _generate_reasoning(self, question: str, retrieved_chunks: List) -> str:
        """Generate reasoning for the retrieval and answer process"""
        top_movies = []
        for chunk, score in retrieved_chunks[:3]:
            title = chunk['metadata']['title']
            if title not in top_movies:
                top_movies.append(title)
        
        if not top_movies:
            return "No relevant movie plot information was found for the query."
        
        reasoning_parts = [
            f"The question was about '{question}'.",
            f"I searched through movie plots and found relevant information from {len(retrieved_chunks)} chunks.",
            f"The most relevant movies were: {', '.join(top_movies)}.",
            "I used these plot details to form a comprehensive answer."
        ]
        
        return " ".join(reasoning_parts)