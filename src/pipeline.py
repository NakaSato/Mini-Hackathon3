"""
End-to-End Pipeline Module

Combines all components:
- Document loading and indexing
- Retrieval
- Answer selection
- Submission generation
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import re
from tqdm import tqdm
from dotenv import load_dotenv

from src.answer_selector import create_answer_selector, SpecialCaseDetector
from src.llm_selector import LLMSelector
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class FahMaiRAGPipeline:
    """
    Complete RAG pipeline for FahMai challenge.
    
    Uses:
    - Best Thai embedding: BAAI/bge-m3
    - ThaiLLM: KBTG-Labs/ThaiLLM-8B-Instruct (Qwen3-8B based, 63B Thai tokens)
    """
    
    def __init__(self,
                 kb_path: str = 'data/knowledge_base',
                 index_path: str = 'output/index',
                 embedding_model: str = 'BAAI/bge-m3',
                 thai_llm_model: str = 'KBTG-Labs/ThaiLLM-8B-Instruct',
                 top_k: int = 5,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 use_lightweight: bool = True,
                 cloud_mode: bool = False):
        """
        Initialize the complete pipeline.

        Args:
            kb_path: Path to knowledge base
            index_path: Path to save/load vector index
            embedding_model: Best Thai embedding model
            thai_llm_model: Thai LLM model for answer selection
            top_k: Number of documents to retrieve
            chunk_size: Chunk size for preprocessing
            chunk_overlap: Chunk overlap
            use_lightweight: Use lightweight cross-encoder (faster) vs full Thai LLM
            cloud_mode: Run in cloud/Colab mode with full ThaiLLM (requires GPU)
        """
        self.kb_path = kb_path
        self.index_path = index_path
        self.embedding_model_name = embedding_model
        self.thai_llm_model = thai_llm_model
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Cloud mode overrides lightweight setting
        self.use_lightweight = use_lightweight or not cloud_mode
        self.cloud_mode = cloud_mode
        
        # Components (initialized lazily)
        self.embedding_model = None
        self.vector_store = None
        self.retriever = None
        self.answer_selector = None
        self.special_case_detector = None
        
        # Cache for results
        self.results_cache = {}
    
    def initialize(self, build_index: bool = False):
        """
        Initialize all pipeline components.
        
        Args:
            build_index: If True, rebuild the index
        """
        from src.embeddings import EmbeddingModel, VectorStore, build_index as build_vector_index
        from src.retrieval import create_retriever, ContextAssembler
        from src.answer_selector_thai import create_answer_selector
        from src.answer_selector import SpecialCaseDetector
        from src.keyword_index import BM25Index
        from src.preprocessing import prepare_chunks
        
        print("Initializing pipeline components...")
        
        # Initialize/Load BM25 Index
        self.bm25_index = BM25Index(persist_directory=self.index_path)
        
        # Load/build embedding model and vector store
        if build_index or not os.path.exists(self.index_path):
            print("\nBuilding Hybrid Index (Vector + BM25)...")
            # Build Vector Store
            self.vector_store, self.embedding_model = build_vector_index(
                kb_path=self.kb_path,
                index_path=self.index_path,
                model_name=self.embedding_model_name,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            # Build BM25 Index (re-using prepare_chunks to get documents)
            chunks = prepare_chunks(self.kb_path, self.chunk_size, self.chunk_overlap)
            self.bm25_index.build(chunks)
        else:
            print("\nLoading existing Hybrid Index...")
            self.embedding_model = EmbeddingModel(self.embedding_model_name)
            self.vector_store = VectorStore(persist_directory=self.index_path)
            self.vector_store.embedding_model = self.embedding_model # Link model
            if not self.bm25_index.load():
                print("Warning: Failed to load BM25 index. Hybrid search will fall back to vector only.")
        
        # Create retriever
        self.retriever = create_retriever(
            self.vector_store,
            self.embedding_model,
            bm25_index=self.bm25_index,
            top_k=self.top_k,
            alpha=0.5 # Equal weight for Vector and BM25
        )
        
        # Create context assembler
        self.context_assembler = ContextAssembler(
            max_context_length=4000,
            include_metadata=True
        )
        
        # Create Thai LLM answer selector
        # Create Thai LLM answer selector
        print(f"\nLoading Thai LLM Reasoning Engine: {self.thai_llm_model}...")
        
        # We always use ThaiLLMAnswerSelector for LLM tasks now, 
        # as it has the superior Reasoning/CoT logic.
        self.answer_selector = create_answer_selector(
            use_thai_llm=True,
            model_name=self.thai_llm_model,
            use_lightweight=self.use_lightweight
        )
        
        # Create special case detector
        self.special_case_detector = SpecialCaseDetector(
            self.embedding_model,
            no_data_threshold=0.012, # Adjusted for RRF score scale
            unrelated_threshold=0.30
        )
        
        print("Pipeline initialized successfully!")
    
    def answer_question(self, 
                       question: str, 
                       choices: List[str],
                       use_special_case: bool = True,
                       use_multi_query: bool = True) -> Tuple[int, Dict[str, Any]]:
        """
        Answer a single question using reasoning-enhanced RAG.
        """
        debug_info = {
            'question': question,
            'choices': choices
        }
        
        # Step 1: Retrieval with HyDE (if in reasoning/lightweight mode)
        if hasattr(self.answer_selector, 'generate_hyde_query'):
            hyde_query = self.answer_selector.generate_hyde_query(question)
            debug_info['hyde_query'] = hyde_query
            retrieval_results = self.retriever.hyde_search(question, hyde_query, top_k=self.top_k)
        elif use_multi_query:
            # Simple query expansion backup
            queries = [question]
            keywords = " ".join(re.findall(r'[\u0e00-\u0e7f]+', question)[:3])
            if keywords and keywords != question:
                queries.append(keywords)
            retrieval_results = self.retriever.multi_search(queries, top_k=self.top_k)
        else:
            retrieval_results = self.retriever.search(question, top_k=self.top_k)
            
        debug_info['retrieval_results'] = [
            {
                'filename': r['metadata'].get('filename'),
                'section': r['metadata'].get('section'),
                'score': r.get('combined_score'),
                'content_preview': r['content'][:200]
            } for r in retrieval_results
        ]
        
        # Step 2: Assemble context (using increased depth for reasoning)
        context = self.context_assembler.assemble(retrieval_results)
        debug_info['context'] = context
        
        # Step 3: Select answer with Reasoning
        # We bypass heuristic SpecialCaseDetector for 1.00 Accuracy goal 
        # unless explicitly requested and no reasoning selector is available.
        if use_special_case and self.special_case_detector and not hasattr(self.answer_selector, 'select'):
             # This is a legacy path
             special_case = self.special_case_detector.detect(question, retrieval_results)
             if special_case is not None:
                 debug_info['special_case'] = special_case
                 return special_case, debug_info
        
        answer = self.answer_selector.select(
            question=question,
            choices=choices,
            context=context
        )
        debug_info['answer'] = answer
        
        return answer, debug_info

    
    def answer_all_questions(self,
                             questions_path: str = 'data/questions.csv',
                             output_path: str = 'output/submission.csv',
                             use_special_case: bool = True,
                             use_multi_query: bool = True,
                             save_debug: bool = True) -> pd.DataFrame:
        """
        Answer all questions and generate submission.
        
        Args:
            questions_path: Path to questions CSV
            output_path: Path for submission CSV
            use_special_case: Use special case detection
            use_multi_query: Use multi-query retrieval
            save_debug: Save debug information
            
        Returns:
            DataFrame with answers
        """
        # Load questions
        print(f"Loading questions from {questions_path}...")
        questions_df = pd.read_csv(questions_path)
        
        results = []
        debug_results = []
        
        print(f"Answering {len(questions_df)} questions...\n")
        
        # Create results list
        results = []
        
        # Prepare for incremental saving
        submission_cols = ['id', 'answer']
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
        # Clear existing file
        with open(output_path, 'w') as f:
            f.write(','.join(submission_cols) + '\n')
            
        pbar = tqdm(total=len(questions_df), desc="Answering Questions")
        
        for i, row in questions_df.iterrows():
            question_id = row['id']
            question = row['question']
            choices = [str(row[f'choice_{j}']) for j in range(1, 11)]
            
            # Answer question
            answer, debug_info = self.answer_question(
                question, choices, use_special_case, use_multi_query
            )
            
            # Add results
            res = {
                'id': question_id,
                'answer': answer
            }
            results.append(res)
            
            # Incremental save submission
            with open(output_path, 'a') as f:
                f.write(f"{question_id},{answer}\n")
                
            # Optional: Incremental save debug info
            if save_debug:
                debug_path = output_path.replace('.csv', '_debug.json')
                with open(debug_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)
                    
            pbar.update(1)
            
        pbar.close()
        
        results_df = pd.DataFrame(results)
        
        # Print statistics
        print("\n=== Answer Distribution ===")
        print(results_df['answer'].value_counts().sort_index())
        
        return results_df
    
    def evaluate_against_sample(self,
                                submission_path: str,
                                sample_path: str = 'data/sample_submission.csv') -> Dict[str, float]:
        """
        Evaluate submission against sample (for testing).
        
        Note: This is just for testing since sample has dummy answers.
        """
        submission = pd.read_csv(submission_path)
        sample = pd.read_csv(sample_path)
        
        # Merge on id
        merged = submission.merge(sample, on='id', suffixes=('_pred', '_true'))
        
        # Calculate accuracy
        correct = (merged['answer_pred'] == merged['answer_true']).sum()
        total = len(merged)
        accuracy = correct / total
        
        return {
            'accuracy': accuracy,
            'correct': int(correct),
            'total': total
        }


def run_pipeline(build_index: bool = False,
                 questions_path: str = 'data/questions.csv',
                 output_path: str = 'output/submission.csv',
                 use_lightweight: bool = True,
                 cloud_mode: bool = False) -> pd.DataFrame:
    """
    Run the complete pipeline.

    Args:
        build_index: Rebuild index if True
        questions_path: Path to questions
        output_path: Path for submission
        use_lightweight: Use lightweight cross-encoder (faster)
        cloud_mode: Run in cloud/Colab mode with full ThaiLLM (requires GPU)

    Returns:
        Results DataFrame
    """
    pipeline = FahMaiRAGPipeline(
        kb_path='data/knowledge_base',
        index_path='output/index',
        embedding_model='intfloat/multilingual-e5-large',
        thai_llm_model='KBTG-Labs/ThaiLLM-8B-Instruct',
        top_k=5,
        use_lightweight=use_lightweight,
        cloud_mode=cloud_mode
    )
    
    pipeline.initialize(build_index=build_index)
    
    results = pipeline.answer_all_questions(
        questions_path=questions_path,
        output_path=output_path,
        use_special_case=True,
        save_debug=True
    )
    
    return results


if __name__ == '__main__':
    # Run pipeline
    results = run_pipeline(build_index=True)
    print(f"\nGenerated {len(results)} answers")
    print(results.head(10))
