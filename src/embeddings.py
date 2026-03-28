"""
Embedding and Vector Store Module

Handles:
- Loading embedding models (Thai-capable)
- Building vector index
- Similarity search
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingModel:
    """
    Wrapper for embedding models with Thai language support.
    
    Recommended models:
    - multilingual-e5-large: Best multilingual performance
    - multilingual-e5-base: Faster, good performance
    - intfloat/multilingual-e5-small: Fastest, decent performance
    """
    
    def __init__(self, model_name: str = 'intfloat/multilingual-e5-large'):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model name
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        print(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        print("Model loaded successfully!")
    
    def encode(self, texts: List[str], 
               batch_size: int = 32,
               show_progress: bool = False) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
        
        Returns:
            numpy array of embeddings
        """
        # Add prefix for E5 models (required for proper encoding)
        if 'e5' in self.model_name.lower():
            texts = [f"passage: {text}" for text in texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query for retrieval.
        
        For E5 models, queries need a different prefix.
        """
        if 'e5' in self.model_name.lower():
            query = f"query: {query}"
        
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embedding[0]
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self.model is None:
            self._load_model()
        return self.model.get_sentence_embedding_dimension()


class VectorStore:
    """
    Vector store using ChromaDB for efficient similarity search.
    """
    
    def __init__(self, 
                 persist_directory: str = 'output/index',
                 collection_name: str = 'fahmai_kb'):
        """
        Initialize vector store.
        
        Args:
            persist_directory: Directory to persist index
            collection_name: Name of ChromaDB collection
        """
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "chromadb not installed. "
                "Run: pip install chromadb"
            )
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_model = None
        
        self._initialize_client()
        self._initialize_collection()
    
    def _initialize_client(self):
        """Initialize ChromaDB client."""
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
    
    def _initialize_collection(self):
        """Initialize or get the collection."""
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=None
            )
        except Exception:
            self.collection = None
    
    def add_documents(self, 
                      chunks: List[Dict[str, Any]],
                      embedding_model: EmbeddingModel,
                      batch_size: int = 100) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of chunk dicts with 'content' and 'metadata'
            embedding_model: EmbeddingModel instance
            batch_size: Batch size for adding documents
        
        Returns:
            Number of documents added
        """
        self.embedding_model = embedding_model
        
        # Get or create collection
        if self.collection is None:
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=None  # We handle embeddings manually
                )
                print(f"Using existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=None,
                    metadata={"hnsw:space": "cosine"}
                )
                print(f"Created new collection: {self.collection_name}")
        
        # Prepare documents in batches
        total_chunks = len(chunks)
        added = 0
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            
            # Extract content and metadata
            documents = [chunk['content'] for chunk in batch]
            metadatas = [chunk['metadata'] for chunk in batch]
            ids = [f"{m['filename']}_{m.get('section', 'main')}_{m['chunk_id']}" 
                   for chunk, m in zip(batch, metadatas)]
            
            # Generate embeddings
            embeddings = embedding_model.encode(
                documents,
                batch_size=batch_size,
                show_progress=(i == 0)
            ).tolist()
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            added += len(batch)
            print(f"Added {added}/{total_chunks} chunks...")
        
        return added
    
    def search(self, 
               query: str,
               top_k: int = 5,
               filter_metadata: Optional[Dict] = None) -> List[Tuple[Dict, float]]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
        
        Returns:
            List of (chunk_metadata, distance) tuples
        """
        if self.collection is None or self.embedding_model is None:
            raise ValueError("Vector store not initialized with embedding model")
        
        # Encode query
        query_embedding = self.embedding_model.encode_query(query).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata,
            include=['metadatas', 'documents', 'distances']
        )
        
        # Format results
        formatted_results = []
        if results['metadatas'] and results['metadatas'][0]:
            for i in range(len(results['metadatas'][0])):
                chunk_info = {
                    'metadata': results['metadatas'][0][i],
                    'content': results['documents'][0][i],
                    'distance': results['distances'][0][i]
                }
                formatted_results.append(chunk_info)
        
        return formatted_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if self.collection is None:
            return {'count': 0}
        
        return {
            'count': self.collection.count(),
            'collection_name': self.collection_name
        }
    
    def reset(self):
        """Reset the vector store (delete all documents)."""
        if self.client is not None:
            try:
                self.client.delete_collection(self.collection_name)
                self.collection = None
                print("Vector store reset successfully!")
            except Exception as e:
                print(f"Error resetting vector store: {e}")


def build_index(kb_path: str,
                index_path: str = 'output/index',
                model_name: str = 'intfloat/multilingual-e5-large',
                chunk_size: int = 500,
                chunk_overlap: int = 50) -> Tuple[VectorStore, EmbeddingModel]:
    """
    Build vector index from knowledge base.
    
    Args:
        kb_path: Path to knowledge base directory
        index_path: Path to save index
        model_name: Embedding model name
        chunk_size: Chunk size for preprocessing
        chunk_overlap: Chunk overlap
    
    Returns:
        Tuple of (VectorStore, EmbeddingModel)
    """
    from src.preprocessing import prepare_chunks
    
    # Load embedding model
    embedding_model = EmbeddingModel(model_name=model_name)
    
    # Prepare chunks
    print("Loading and chunking documents...")
    chunks = prepare_chunks(kb_path, chunk_size, chunk_overlap)
    print(f"Total chunks: {len(chunks)}")
    
    # Build vector store
    vector_store = VectorStore(persist_directory=index_path)
    vector_store.add_documents(chunks, embedding_model)
    
    print(f"\nIndex built successfully!")
    print(f"Stats: {vector_store.get_stats()}")
    
    return vector_store, embedding_model


if __name__ == '__main__':
    # Test building index
    vector_store, model = build_index(
        kb_path='data/knowledge_base',
        index_path='output/index'
    )
    
    # Test search
    results = vector_store.search("iPhone กันน้ำได้ไหม", top_k=3)
    print(f"\nSearch results:")
    for r in results:
        print(f"  - {r['metadata'].get('filename')}: {r['content'][:100]}...")
