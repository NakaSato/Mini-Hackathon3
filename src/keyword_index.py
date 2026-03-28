"""
Keyword Indexing Module using BM25

Handles:
- Tokenizing Thai text using pythainlp
- Managing BM25 index for keyword search
- Saving and loading index from disk
"""

import os
import pickle
from typing import List, Dict, Any, Optional
from pythainlp.tokenize import word_tokenize
from rank_bm25 import BM25Okapi


class BM25Index:
    """
    BM25 index for fast keyword-based retrieval.
    """
    
    def __init__(self, persist_directory: str = 'output/index'):
        """
        Initialize BM25 index.
        
        Args:
            persist_directory: Directory to save/load index
        """
        self.persist_directory = persist_directory
        self.index_path = os.path.join(persist_directory, 'bm25_index.pkl')
        self.bm25 = None
        self.chunks = []
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Thai/English text for indexing.
        """
        # Lowercase and tokenize
        tokens = word_tokenize(text.lower(), engine='newmm')
        # Remove whitespace and single characters (except numbers)
        tokens = [t.strip() for t in tokens if t.strip() and (len(t.strip()) > 1 or t.isdigit())]
        return tokens
    
    def build(self, chunks: List[Dict[str, Any]]):
        """
        Build BM25 index from document chunks.
        
        Args:
            chunks: List of chunk dicts from preprocessing
        """
        print(f"Building BM25 index from {len(chunks)} chunks...")
        self.chunks = chunks
        
        # Prepare corpus for BM25
        corpus_tokens = []
        for chunk in chunks:
            text = f"{chunk['content']} {chunk['metadata'].get('filename', '')} {chunk['metadata'].get('section', '')}"
            tokenized = self.tokenize(text)
            corpus_tokens.append(tokenized)
        
        # Initialize BM25
        self.bm25 = BM25Okapi(corpus_tokens)
        print("BM25 index built successfully!")
        
        # Save to disk
        self.save()
    
    def save(self):
        """Save the index and chunks to disk."""
        print(f"Saving BM25 index to {self.index_path}...")
        data = {
            'bm25': self.bm25,
            'chunks': self.chunks
        }
        with open(self.index_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self) -> bool:
        """
        Load the index from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(self.index_path):
            return False
        
        try:
            print(f"Loading BM25 index from {self.index_path}...")
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.chunks = data['chunks']
            return True
        except Exception as e:
            print(f"Error loading BM25 index: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using BM25.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of ranked results with scores
        """
        if self.bm25 is None or not self.chunks:
            return []
        
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top indices
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score <= 0:
                continue
                
            # Copy chunk and add score
            chunk = self.chunks[idx].copy()
            chunk['keyword_score'] = float(score)
            results.append(chunk)
            
        return results

if __name__ == '__main__':
    # Simple test
    index = BM25Index()
    test_chunks = [
        {'content': 'โน้ตบุ๊คสายฟ้า X9 Pro มี RAM 16GB', 'metadata': {'filename': 'SF-SP-002_saifah_phone_x9_pro'}},
        {'content': 'หูฟัง คลื่นเสียง Z5 Pro ตัดเสียงเงียบ', 'metadata': {'filename': 'KS-HP-005_headon_300'}},
    ]
    index.build(test_chunks)
    
    query = "RAM 16GB"
    results = index.search(query)
    print(f"Query: {query}")
    for r in results:
        print(f"Score: {r['keyword_score']:.3f} - Content: {r['content']}")
