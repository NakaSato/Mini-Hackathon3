"""
Retrieval System Module

Handles:
- Hybrid search (semantic + keyword)
- Re-ranking
- Context assembly
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter


class RetrievalSystem:
    """
    Hybrid retrieval system combining semantic search with BM25 keyword matching.
    Uses Reciprocal Rank Fusion (RRF) to combine results.
    """
    
    def __init__(self, 
                 vector_store,
                 embedding_model,
                 bm25_index=None,
                 top_k: int = 5,
                 rrf_k: int = 60,
                 alpha: float = 0.5):
        """
        Initialize retrieval system.
        
        Args:
            vector_store: VectorStore instance
            embedding_model: EmbeddingModel instance
            bm25_index: BM25Index instance
            top_k: Number of final results to return
            rrf_k: Constant for RRF (default 60)
            alpha: Weight for vector search (1-alpha for BM25)
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.bm25_index = bm25_index
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.alpha = alpha
    
    def rrf_fuse(self, 
                 vector_results: List[Dict[str, Any]], 
                 keyword_results: List[Dict[str, Any]], 
                 top_k: int) -> List[Dict[str, Any]]:
        """
        Fuse results from vector and keyword search using Reciprocal Rank Fusion.
        """
        scores = {}  # id -> score
        doc_map = {} # id -> doc
        
        # Helper to get a unique ID for a chunk
        def get_id(doc):
            m = doc['metadata']
            return f"{m['filename']}_{m.get('section', 'main')}_{m.get('chunk_id', 0)}"

        # Process vector results
        for rank, doc in enumerate(vector_results):
            doc_id = get_id(doc)
            scores[doc_id] = scores.get(doc_id, 0) + self.alpha * (1.0 / (self.rrf_k + rank + 1))
            doc_map[doc_id] = doc
            doc_map[doc_id]['vector_rank'] = rank + 1

        # Process keyword results
        for rank, doc in enumerate(keyword_results):
            doc_id = get_id(doc)
            scores[doc_id] = scores.get(doc_id, 0) + (1.0 - self.alpha) * (1.0 / (self.rrf_k + rank + 1))
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            doc_map[doc_id]['keyword_rank'] = rank + 1
            
        # Sort and return
        fused_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        results = []
        for doc_id in fused_ids[:top_k]:
            doc = doc_map[doc_id]
            doc['combined_score'] = scores[doc_id]
            results.append(doc)
            
        return results

    def _get_anchor_docs(self, query: str) -> List[Dict[str, Any]]:
        """
        Identify product entities in query and find their specific files.
        """
        anchors = []
        # Common product names and SKUs
        product_patterns = [
            r'Watch S3 Ultra', r'Watch S3 Pro', r'Watch S3', r'HeadPro X1', r'HeadPro X1 SE',
            r'SaiFah Phone DuoPad', r'SaiFah Phone X9 Pro', r'SaiFah Tab S9 Pro',
            r'SaiFah Tab Mini 7', r'SaiFah Pen Gen [12]', r'Senior Plus', r'WK-SW-\d+', r'SF-SP-\d+',
            r'JC-CS-\d+'
        ]
        
        for p in product_patterns:
            if re.search(p, query, re.IGNORECASE):
                # Search by filename/metadata in BM25 if available
                if self.bm25_index:
                    # Clean the pattern for simple matching
                    clean_p = p.replace(r' [12]', '').replace(r'\d+', '').strip()
                    # INCREASED top_k to 20 to pull more chunks of the same file
                    results = self.bm25_index.search(clean_p, top_k=20)
                    anchors.extend(results)
        return anchors

    def search(self, query: str, 
                top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Hybrid search using Vector + BM25 with RRF fusion + Anchor Boost.
        """
        k = top_k or self.top_k
        candidates = max(50, k * 5)
        
        # 1. Anchor Search (Entity detection)
        anchor_results = self._get_anchor_docs(query)
        
        # 2. Vector Search
        vector_results = self.vector_store.search(query, top_k=candidates)
        
        # 3. Keyword Search (BM25)
        keyword_results = []
        if self.bm25_index:
            keyword_results = self.bm25_index.search(query, top_k=candidates)
        
        # 4. Fusion
        fused = self.rrf_fuse(vector_results, keyword_results, candidates)
        
        # 5. Inject Anchors at the top
        final_results = []
        seen_ids = set()
        
        def get_id(doc):
            m = doc['metadata']
            return f"{m['filename']}_{m.get('section', 'main')}_{m.get('chunk_id', 0)}"

        for doc in anchor_results:
            doc_id = get_id(doc)
            if doc_id not in seen_ids:
                doc['combined_score'] = 2.0 # High fixed score for anchor matches
                final_results.append(doc)
                seen_ids.add(doc_id)
        
        for doc in fused:
            doc_id = get_id(doc)
            if doc_id not in seen_ids:
                final_results.append(doc)
                seen_ids.add(doc_id)
                
        return final_results[:k]

    def multi_search(self, queries: List[str], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform search for multiple query variations and fuse the results.
        Uses score-averaging for overlapping chunks to reward high-confidence documents.
        """
        k = top_k or self.top_k
        
        doc_scores = {} # doc_id -> sum of scores
        doc_counts = {} # doc_id -> number of occurrences
        doc_map = {}    # doc_id -> doc
        
        for q in queries:
            results = self.search(q, top_k=max(20, k))
            for r in results:
                m = r['metadata']
                doc_id = f"{m['filename']}_{m.get('section', 'main')}_{m.get('chunk_id', 0)}"
                
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + r['combined_score']
                doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
                
                if doc_id not in doc_map or r['combined_score'] > doc_map[doc_id]['combined_score']:
                    doc_map[doc_id] = r
                    
        # Calculate final scores
        final_results = []
        for doc_id, base_doc in doc_map.items():
            avg_score = doc_scores[doc_id] / len(queries)
            # Apply a small boost for documents found in multiple queries
            frequency_boost = 1.0 + (doc_counts[doc_id] / len(queries)) * 0.2
            
            base_doc['combined_score'] = avg_score * frequency_boost
            final_results.append(base_doc)
                
        return sorted(final_results, key=lambda x: x['combined_score'], reverse=True)[:k]
    
    def hyde_search(self, original_query: str, hyde_query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        High-accuracy search using both original question and hypothetical answer.
        """
        return self.multi_search([original_query, hyde_query], top_k=top_k)

    
    def search_with_filters(self, query: str,
                           filter_metadata: Dict[str, str],
                           top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search with metadata filters.
        Currently falls back to vector search for simplicity with filters.
        """
        k = top_k or self.top_k
        results = self.vector_store.search(
            query,
            top_k=k,
            filter_metadata=filter_metadata
        )
        for doc in results:
            doc['combined_score'] = 1.0 - doc.get('distance', 1.0)
        return results


def create_retriever(vector_store, 
                     embedding_model,
                     bm25_index=None,
                     top_k: int = 5,
                     alpha: float = 0.5) -> RetrievalSystem:
    """
    Factory function to create retrieval system.
    """
    return RetrievalSystem(
        vector_store=vector_store,
        embedding_model=embedding_model,
        bm25_index=bm25_index,
        top_k=top_k,
        alpha=alpha
    )


class ContextAssembler:
    """
    Assembles retrieved chunks into coherent context for LLM.
    """
    
    def __init__(self, 
                 max_context_length: int = 4000,
                 include_metadata: bool = True):
        """
        Initialize context assembler.
        
        Args:
            max_context_length: Maximum context length in characters
            include_metadata: Whether to include metadata in context
        """
        self.max_context_length = max_context_length
        self.include_metadata = include_metadata
    
    def assemble(self, results: List[Dict[str, Any]]) -> str:
        """
        Assemble retrieved results into context string.
        
        Args:
            results: List of retrieval results
        
        Returns:
            Formatted context string
        """
        if not results:
            return "ไม่พบข้อมูลที่เกี่ยวข้อง"
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results):
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            
            # Build section header
            section_parts = []
            
            if self.include_metadata:
                if metadata.get('filename'):
                    section_parts.append(f"[เอกสาร: {metadata['filename']}]")
                if metadata.get('section'):
                    section_parts.append(f"[ส่วน: {metadata['section']}]")
            
            # Add content
            section_text = ' '.join(section_parts) + '\n' + content if section_parts else content
            
            # Check length
            if current_length + len(section_text) > self.max_context_length:
                break
            
            context_parts.append(section_text)
            current_length += len(section_text)
        
        return '\n\n---\n\n'.join(context_parts)
    
    def assemble_with_sources(self, results: List[Dict[str, Any]]) -> Tuple[str, List[Dict]]:
        """
        Assemble context with source tracking.
        
        Returns:
            Tuple of (context string, list of source metadata)
        """
        if not results:
            return "ไม่พบข้อมูลที่เกี่ยวข้อง", []
        
        context_parts = []
        sources = []
        current_length = 0
        
        for i, result in enumerate(results):
            content = result.get('content', '')
            metadata = result.get('metadata', {}).copy()
            metadata['score'] = result.get('combined_score', 0)
            
            source_marker = f"[{i+1}]"
            section_text = f"{source_marker} {content}"
            
            if current_length + len(section_text) > self.max_context_length:
                break
            
            context_parts.append(section_text)
            sources.append(metadata)
            current_length += len(section_text)
        
        context = '\n\n'.join(context_parts)
        return context, sources


if __name__ == '__main__':
    # Test retrieval system
    from src.embeddings import VectorStore, EmbeddingModel
    
    # Load existing index
    embedding_model = EmbeddingModel('intfloat/multilingual-e5-large')
    vector_store = VectorStore(persist_directory='output/index')
    
    retriever = create_retriever(
        vector_store, 
        embedding_model,
        top_k=5,
        keyword_boost=0.3
    )
    
    # Test query
    query = "Watch S3 Ultra กันน้ำได้กี่ ATM"
    results = retriever.search(query)
    
    print(f"Query: {query}")
    print(f"\nResults:")
    for i, r in enumerate(results):
        print(f"\n{i+1}. Score: {r['combined_score']:.3f}")
        print(f"   File: {r['metadata'].get('filename')}")
        print(f"   Content: {r['content'][:200]}...")
