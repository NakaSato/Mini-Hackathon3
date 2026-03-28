# FahMai RAG System - Project Context

## Project Overview

This is a **Retrieval-Augmented Generation (RAG) system** built for the **FahMai (ฟ้าใหม่) Thai Electronics Store QA Challenge**. The system answers 100 multiple-choice questions about a fictional Thai electronics store using a knowledge base of product documentation, store policies, and store information.

### Key Features

- **Hybrid Search**: Combines semantic vector search (BGE-M3 embeddings) with BM25 keyword search using Reciprocal Rank Fusion (RRF)
- **Thai Language Support**: Full Thai text processing with `pythainlp` tokenization and Thai-capable embedding models
- **Dual Execution Modes**:
  - **Local Mode (default)**: Lightweight cross-encoder for fast inference (no GPU required)
  - **Cloud Mode**: Full ThaiLLM-8B-Instruct for maximum accuracy (requires GPU or Google Colab)
- **Special Case Detection**: Automatic handling of out-of-scope questions (Choice 10) and missing data (Choice 9)

### Architecture

```
Knowledge Base (Markdown) → Preprocessing → Embedding (BGE-M3) → ChromaDB Vector Store
                                                              ↓
Questions (CSV) → Hybrid Search (Vector + BM25 + RRF) → Context Assembly → Answer Selection
```

## Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.14+ |
| **Package Manager** | `uv` (primary), `pip` (alternative) |
| **Embeddings** | `BAAI/bge-m3` (default), `intfloat/multilingual-e5-large` |
| **Vector Store** | ChromaDB (persistent) |
| **LLM** | `KBTG-Labs/ThaiLLM-8B-Instruct` (cloud mode only) |
| **Lightweight** | `BAAI/bge-reranker-v2-m3` cross-encoder |
| **Thai NLP** | `pythainlp` for word tokenization |
| **Keyword Search** | `rank-bm25` with Thai tokenization |

## Project Structure

```
Mini-Hackathon3/
├── data/
│   ├── knowledge_base/       # 118 Markdown files (products, policies, store info)
│   ├── questions.csv         # 100 multiple-choice questions
│   └── sample_submission.csv # Submission template
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Markdown parsing, section-based chunking
│   ├── embeddings.py         # Embedding model wrapper, ChromaDB vector store
│   ├── retrieval.py          # Hybrid search with RRF fusion
│   ├── keyword_index.py      # BM25 index with Thai tokenization
│   ├── answer_selector.py    # Special case detection (Choice 9/10)
│   ├── answer_selector_thai.py  # ThaiLLM answer selection
│   ├── llm_selector.py       # LLM selection utilities
│   └── pipeline.py           # End-to-end pipeline orchestration
├── output/
│   ├── index/                # Persisted vector + BM25 indices
│   └── submission.csv        # Generated submission file
├── main.py                   # CLI entry point
├── FahMai_RAG_Colab.ipynb    # Google Colab notebook for cloud execution
├── requirements.txt          # Pip dependencies
├── pyproject.toml           # Uv project configuration
└── README.md                # User documentation
```

## Building and Running

### Installation

```bash
# Using uv (recommended)
uv sync

# Using pip
pip install -r requirements.txt
```

### Running the Pipeline

**Local Development (Default - No GPU Required):**
```bash
# Build index and answer all questions (lightweight mode)
python main.py

# Test setup
python main.py --test

# Rebuild index only
python main.py --build
```

**Cloud/Colab Mode (Full ThaiLLM - GPU Required):**
```bash
# Full ThaiLLM mode (requires GPU with 16GB+ VRAM)
python main.py --cloud

# Or use Google Colab: Open FahMai_RAG_Colab.ipynb
```

### CLI Options

```bash
python main.py [OPTIONS]

Options:
  --build       Rebuild the vector index
  --answer      Answer questions only (skip index building)
  --test        Run test with sample data
  --cloud       Enable full ThaiLLM mode (for Colab/cloud with GPU)
  --lightweight Force lightweight mode (cross-encoder only)
  --model MODEL ThaiLLM model name (default: KBTG-Labs/ThaiLLM-8B-Instruct)
  --top-k K     Number of documents to retrieve (default: 5)
  --questions PATH  Path to questions CSV
  --output PATH     Path for output submission CSV
```

### Configuration

Create a `.env` file (optional):
```bash
LOCAL_RUN_THAI_LLM=false  # Set to true only for local GPU testing
EMBEDDING_MODEL=BAAI/bge-m3
TOP_K=5
ALPHA=0.5  # RRF weight: 0.5 = equal vector/BM25 weighting
DEVICE=auto
```

## Key Components

### 1. Document Preprocessing (`src/preprocessing.py`)
- Parses 118 Markdown files from `data/knowledge_base/`
- Chunks by document sections (preserves structure)
- Extracts metadata: SKU, brand, category, price
- Normalizes Thai text

### 2. Embedding & Indexing (`src/embeddings.py`, `src/keyword_index.py`)
- **Vector Index**: BGE-M3 embeddings (1024-dim) stored in ChromaDB
- **Keyword Index**: BM25 with `pythainlp` Thai tokenization
- Index persisted to `output/index/`

### 3. Hybrid Retrieval (`src/retrieval.py`)
- **Vector Search**: Cosine similarity on embeddings
- **BM25 Search**: Keyword matching with Thai tokens
- **RRF Fusion**: Reciprocal Rank Fusion (k=60, alpha=0.5)
- **Anchor Boost**: Product entity detection for exact SKU matches
- **HyDE Support**: Hypothetical document expansion

### 4. Answer Selection (`src/answer_selector_thai.py`, `src/answer_selector.py`)
- **Lightweight Mode**: BGE-Reranker cross-encoder with logical guards
- **Full Mode**: ThaiLLM-8B-Instruct with Chain-of-Thought reasoning
- **Special Cases**: Choice 9 (no data), Choice 10 (out-of-scope)

### 5. Pipeline (`src/pipeline.py`)
- Orchestrates all components
- Incremental submission saving
- Debug output generation

## Development Practices

### Code Style
- Type hints throughout the codebase
- Docstrings for all classes and public methods
- Modular design with clear separation of concerns

### Testing
- Use `--test` flag for quick setup verification
- Test mode checks: document loading, embedding, retrieval

### Performance Optimization
- **Index Reuse**: Build once, reuse for all questions
- **Batch Processing**: Embeddings generated in batches (default: 100)
- **Incremental Saving**: Submission saved after each question
- **Lightweight Default**: Fast iteration without GPU

## Common Workflows

### First-Time Setup
```bash
# 1. Install dependencies
uv sync

# 2. Test setup
python main.py --test

# 3. Build index and run (lightweight mode)
python main.py
```

### Iterative Development
```bash
# After modifying retrieval logic:
python main.py --answer  # Skip index rebuilding

# After modifying preprocessing:
python main.py --build   # Rebuild index
```

### Final Submission (Best Accuracy)
```bash
# On Colab with GPU:
# Open FahMai_RAG_Colab.ipynb → Run all cells

# Or on local machine with powerful GPU:
python main.py --cloud --top-k 10
```

## Output Format

### Submission CSV (`output/submission.csv`)
```csv
id,answer
1,5
2,3
...
100,2
```

### Debug JSON (`output/submission_debug.json`)
- Retrieval results per question
- Context used for answer selection
- Special case flags
- Answer reasoning (if available)

## Performance Benchmarks

| Mode | Time (100 questions) | GPU Required | Accuracy |
|------|---------------------|--------------|----------|
| Lightweight | 5-10 minutes | No | Good |
| Full ThaiLLM | 30-60 minutes | Yes (16GB+) | Best |

## Troubleshooting

### Out of Memory
- Use default lightweight mode (no ThaiLLM)
- Don't use `--cloud` without sufficient GPU memory
- Clear GPU cache in Colab: `torch.cuda.empty_cache()`

### Slow Performance
- Use lightweight mode (default)
- Reduce `--top-k` value
- Pre-build index with `--build`

### Thai Text Issues
- Ensure UTF-8 encoding
- Verify `pythainlp` installation
- Use Thai-capable embedding models (BGE-M3)

## Related Documentation

- `README.md` - User-facing documentation
- `DATA_FLOW.md` - Detailed data flow diagrams
- `OVERVIEW.md` - Challenge overview and rules
- `FahMai_RAG_Colab.ipynb` - Google Colab notebook
