"""
Embedding generation using Cohere API
"""

import os
from typing import List, Optional
import cohere
from dotenv import load_dotenv

load_dotenv()


class EmbeddingClient:
    """Client for generating embeddings using Cohere API"""
    
    def __init__(self, model: str = 'multilingual-22-12'):
        """
        Initialize Cohere client for embeddings
        
        Args:
            model: Cohere embedding model to use
                  Default: 'multilingual-22-12' (768 dimensions)
                  Other options: 'embed-english-v3.0' (1024 dimensions), 
                                'embed-multilingual-v3.0' (1024 dimensions)
        """
        api_key = os.getenv("COHERE_API_KEY")
        self.co = cohere.Client(api_key)
        self.model = model
    
    def embed_texts(self, texts: List[str], input_type: str = "search_document") -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            input_type: Type of input - "search_document" for documents, "search_query" for queries
                       (Note: only used for v3 models, ignored for multilingual-22-12)
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Cohere has a limit on batch size, process in chunks if needed
        batch_size = 96  # Cohere's recommended batch size
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # The older multilingual-22-12 model doesn't support input_type
            if 'v3' in self.model or 'v2' in self.model:
                response = self.co.embed(
                    texts=batch,
                    model=self.model,
                    input_type=input_type
                )
            else:
                response = self.co.embed(
                    texts=batch,
                    model=self.model
                )
            all_embeddings.extend(response.embeddings)
        
        return all_embeddings
    
    def embed_text(self, text: str, input_type: str = "search_document") -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string to embed
            input_type: Type of input - "search_document" for documents, "search_query" for queries
            
        Returns:
            Embedding vector
        """
        embeddings = self.embed_texts([text], input_type=input_type)
        return embeddings[0] if embeddings else []
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query
        
        Args:
            query: Search query text
            
        Returns:
            Embedding vector optimized for search
        """
        return self.embed_text(query, input_type="search_query")
    
    def embed_document(self, document: str) -> List[float]:
        """
        Generate embedding for a document
        
        Args:
            document: Document text
            
        Returns:
            Embedding vector optimized for indexing
        """
        return self.embed_text(document, input_type="search_document")


def test_embeddings():
    """Test the embedding client"""
    client = EmbeddingClient()
    
    # Test single text embedding
    text = "Hello from Cohere!"
    embedding = client.embed_text(text)
    print(f"Single text embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Test query vs document embedding
    query = "What is machine learning?"
    query_embedding = client.embed_query(query)
    doc_embedding = client.embed_document(query)
    
    print(f"\nQuery embedding dimension: {len(query_embedding)}")
    print(f"Document embedding dimension: {len(doc_embedding)}")
    
    # Test batch embedding
    texts = [
        'Hello from Cohere!',
        'Machine learning is awesome',
        'Natural language processing'
    ]
    embeddings = client.embed_texts(texts)
    print(f"\nBatch embedding count: {len(embeddings)}")
    print(f"Each embedding dimension: {len(embeddings[0])}")


if __name__ == "__main__":
    test_embeddings()