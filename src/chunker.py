import re
from typing import List, Dict, Any
from .config import Config

class SmartChunker:
    def __init__(self, config: Config):
        self.config = config
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk documents using smart strategy"""
        chunks = []
        
        for doc in documents:
            doc_chunks = self._chunk_document(doc)
            chunks.extend(doc_chunks)
        
        return chunks
    
    def _chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a single document using smart strategies"""
        content = document['content']
        words = content.split()
        
        if len(words) <= self.config.CHUNK_SIZE:
            # No need to chunk
            return [{
                'content': content,
                'metadata': document['metadata'],
                'chunk_id': f"{document['metadata']['title']}_0"
            }]
        
        if self.config.SMART_CHUNKING:
            return self._semantic_chunking(content, document)
        else:
            return self._fixed_size_chunking(content, document)
    
    def _fixed_size_chunking(self, content: str, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fixed-size chunking with overlap"""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), self.config.CHUNK_SIZE - self.config.CHUNK_OVERLAP):
            chunk_words = words[i:i + self.config.CHUNK_SIZE]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'content': chunk_text,
                'metadata': document['metadata'],
                'chunk_id': f"{document['metadata']['title']}_{i//self.config.CHUNK_SIZE}"
            })
            
            if i + self.config.CHUNK_SIZE >= len(words):
                break
        
        return chunks
    
    def _semantic_chunking(self, content: str, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Smart chunking that respects sentence boundaries and narrative structure"""
        # Split into sentences first
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_length = len(sentence_words)
            
            # If adding this sentence exceeds chunk size and we have content, save current chunk
            if (current_length + sentence_length > self.config.CHUNK_SIZE and 
                current_length > 0):
                
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'content': chunk_text,
                    'metadata': document['metadata'],
                    'chunk_id': f"{document['metadata']['title']}_{len(chunks)}"
                })
                
                # Keep some overlap by carrying over the last few sentences
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk[-1:]
                current_chunk = overlap_sentences
                current_length = sum(len(s.split()) for s in overlap_sentences)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add the final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'content': chunk_text,
                'metadata': document['metadata'],
                'chunk_id': f"{document['metadata']['title']}_{len(chunks)}"
            })
        
        return chunks