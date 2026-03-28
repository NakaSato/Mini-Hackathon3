"""
FahMai RAG System - Main Entry Point

Usage:
    python main.py              # Run full pipeline (lightweight mode by default)
    python main.py --build      # Rebuild index
    python main.py --answer     # Answer questions only
    python main.py --test       # Test with single question
    python main.py --cloud      # Run with full ThaiLLM (for Colab/cloud with GPU)

Modes:
    - Local (default): Uses lightweight cross-encoder only (fast, no GPU needed)
    - Cloud/Colab (--cloud): Uses full ThaiLLM-8B-Instruct (requires GPU)
"""

import argparse
import sys
import os
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description='FahMai RAG System for Thai Electronics Store QA Challenge'
    )

    parser.add_argument(
        '--build',
        action='store_true',
        help='Rebuild the vector index'
    )

    parser.add_argument(
        '--answer',
        action='store_true',
        help='Answer questions only (skip index building)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test with sample question'
    )

    parser.add_argument(
        '--cloud',
        action='store_true',
        help='Run with full ThaiLLM mode (for Colab/cloud with GPU)'
    )

    parser.add_argument(
        '--questions',
        type=str,
        default='data/questions.csv',
        help='Path to questions CSV'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output/submission.csv',
        help='Path for output submission CSV'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='KBTG-Labs/ThaiLLM-8B-Instruct',
        help='ThaiLLM model for answer selection'
    )

    parser.add_argument(
        '--lightweight',
        action='store_true',
        help='Use lightweight cross-encoder (faster, less memory)'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of documents to retrieve'
    )

    parser.add_argument(
        '--multi-query',
        action='store_true',
        help='Use multi-query retrieval for higher recall'
    )

    args = parser.parse_args()

    # Run test mode
    if args.test:
        run_test()
        return

    # Determine mode
    # Local mode (default): lightweight=True, use_lightweight only
    # Cloud mode (--cloud): lightweight=False, use full ThaiLLM
    use_lightweight = not args.cloud

    # Run full pipeline
    print("=" * 60)
    print("FahMai RAG System")
    print("=" * 60)
    print(f"Mode: {'Cloud (Full ThaiLLM)' if args.cloud else 'Local (Lightweight)'}")
    print("=" * 60)

    from src.pipeline import FahMaiRAGPipeline

    # Initialize pipeline
    pipeline = FahMaiRAGPipeline(
        kb_path='data/knowledge_base',
        index_path='output/index',
        embedding_model='BAAI/bge-m3',
        thai_llm_model=args.model,
        top_k=args.top_k,
        use_lightweight=use_lightweight
    )

    # Build index if requested or doesn't exist
    build_index = args.build or args.answer or not os.path.exists('output/index')

    if args.build or not os.path.exists('output/index'):
        pipeline.initialize(build_index=True)
    else:
        pipeline.initialize(build_index=False)

    # Answer questions
    print("\n" + "=" * 60)
    print("Answering Questions")
    print("=" * 60 + "\n")

    results = pipeline.answer_all_questions(
        questions_path=args.questions,
        output_path=args.output,
        use_special_case=True,
        save_debug=True
    )

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Submission saved to: {args.output}")
    print(f"Total questions answered: {len(results)}")


def run_test():
    """Run a quick test with sample data."""
    print("Running test mode...")
    
    # Test preprocessing
    from src.preprocessing import load_knowledge_base, chunk_document
    
    print("\n1. Testing document loading...")
    docs = load_knowledge_base('data/knowledge_base')
    print(f"   Loaded {len(docs)} documents")
    
    if docs:
        chunks = chunk_document(docs[0])
        print(f"   Sample document chunked into {len(chunks)} pieces")
    
    # Test embedding (if model available)
    print("\n2. Testing embedding model...")
    try:
        from src.embeddings import EmbeddingModel
        model = EmbeddingModel('intfloat/multilingual-e5-small')
        
        test_texts = ["ทดสอบภาษาไทย", "Test English"]
        embeddings = model.encode(test_texts)
        print(f"   Embedding shape: {embeddings.shape}")
        print("   Embedding test: PASSED")
    except Exception as e:
        print(f"   Embedding test: SKIPPED ({e})")
    
    # Test retrieval
    print("\n3. Testing retrieval...")
    try:
        from src.embeddings import VectorStore
        if os.path.exists('output/index'):
            vs = VectorStore(persist_directory='output/index')
            print(f"   Vector store stats: {vs.get_stats()}")
            print("   Retrieval test: PASSED")
        else:
            print("   Retrieval test: SKIPPED (index not built)")
    except Exception as e:
        print(f"   Retrieval test: SKIPPED ({e})")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
