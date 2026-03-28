# FahMai RAG System - ThaiLLM Version

RAG system for the FahMai (ฟ้าใหม่) Thai electronics store challenge using **official ThaiLLM models**.

## Features

- **Thai Embedding**: `BAAI/bge-m3` - Top-ranked multilingual model for Thai (BGE-M3)
- **ThaiLLM**: `KBTG-Labs/ThaiLLM-8B-Instruct` - Official ThaiLLM (Qwen3-8B based, trained on 63B tokens)
- **Hybrid Retrieval**: Vector (Semantic) + BM25 (Keyword) using **Reciprocal Rank Fusion (RRF)**
- **Thai-Aware**: Full support for Thai word tokenization using `pythainlp`
- **Special Case Detection**: Automatic handling of Choice 9 (no data) and Choice 10 (out-of-scope)

## Models

| Component | Model | Details |
|-----------|-------|---------|
| **Embedding** | `BAAI/bge-m3` | Top-tier for Thai retrieval (BGE-M3) |
| **ThaiLLM** | `KBTG-Labs/ThaiLLM-8B-Instruct` | Official 8B model (63B tokens) |
| **Hybrid** | `Vector + BM25` | Reciprocal Rank Fusion (RRF) |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Full pipeline with lightweight mode (recommended for speed)
python main.py --lightweight

# Full pipeline with full ThaiLLM (better accuracy, slower)
python main.py

# Rebuild index
python main.py --build

# Test setup
python main.py --test
```

## Project Structure

```
.
├── data/
│   ├── knowledge_base/       # Product docs, policies, store info
│   ├── questions.csv         # 100 questions
│   └── sample_submission.csv # Submission template
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Document parsing & chunking
│   ├── embeddings.py         # Multilingual-e5 embeddings, ChromaDB
│   ├── retrieval.py          # Hybrid retrieval system
│   ├── answer_selector_thai.py  # ThaiLLM answer selection
│   ├── answer_selector.py    # Special case detection
│   └── pipeline.py           # End-to-end pipeline
├── scripts/
│   └── explore_data.py       # Data exploration script
├── output/
│   ├── index/                # Vector index (auto-generated)
│   └── submission.csv        # Final submission (auto-generated)
├── requirements.txt
├── main.py
└── README.md
```

## Configuration

The system uses these default models:

| Component | Model | Purpose |
|-----------|-------|---------|
| **Embedding** | `BAAI/bge-m3` | Thai semantic search (BGE-M3) |
| **ThaiLLM** | `KBTG-Labs/ThaiLLM-8B-Instruct` | Answer selection (8B model) |
| **Keywords** | `rank-bm25` | Thai-aware BM25 search |

## Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --build       Rebuild the vector index
  --answer      Answer questions only (skip index building)
  --test        Run test with sample data
  --lightweight Use cross-encoder (faster, less memory)
  --model MODEL ThaiLLM model name (default: KBTG-Labs/ThaiLLM-8B-Instruct)
  --top-k K     Number of documents to retrieve
  --questions PATH  Path to questions CSV
  --output PATH     Path for output submission
```

## Pipeline Components

### 1. Document Preprocessing
- Parses markdown files
- Chunks by sections (respects document structure)
- Thai text normalization
- Metadata extraction (SKU, brand, category)

### 2. Embedding & Vector Store
- Uses `BAAI/bge-m3` for Thai embeddings
- ChromaDB for persistent vector storage
- Cosine similarity search

### 3. Hybrid Retrieval
- **Vector Search**: Semantic matching using `BGE-M3`
- **Keyword Search**: BM25 with `pythainlp` tokenization
- **RRF Fusion**: Reciprocal Rank Fusion (k=60) for robust combining
- **SKU Matching**: High precision for technical product codes

### 4. Answer Selection
- **ThaiLLM** (`KBTG-Labs/ThaiLLM-8B-Instruct`) for answer selection
- Cross-encoder mode for speed (10-100x faster)
- Special case detection (Choice 9 & 10)

### 5. Special Case Handling
- **Choice 9**: "ไม่มีข้อมูลนี้ในฐานข้อมูล" - Detected by low retrieval scores
- **Choice 10**: "คำถามนี้ไม่เกี่ยวข้องกับร้านฟ้าใหม่" - Detected by topic classification

## Output

Generates `output/submission.csv`:

```csv
id,answer
1,5
2,3
...
100,2
```

Plus debug info in `output/submission_debug.json`.

## Performance Tips

1. **Use `--lightweight`** for faster iteration during development
2. **Increase `--top-k`** for better recall (default: 5)
3. **Rebuild index** with `--build` if you modify the knowledge base
4. **GPU recommended** for full ThaiLLM mode (8B model)

## Troubleshooting

### Out of Memory
- Use `--lightweight` mode
- Reduce batch size in embeddings.py
- Use CPU mode (slower but works)

### Slow Inference
- Lightweight mode is 10-100x faster
- Pre-build index with `--build`
- Reduce `--top-k` value

### Thai Text Issues
- Ensure UTF-8 encoding
- Check pythainlp installation
- Verify embedding model supports Thai

## About ThaiLLM

**KBTG-Labs/ThaiLLM-8B-Instruct** is the official Thai language model:
- **Architecture**: Merge of ThaiLLM-8B and Qwen3-8B using mergekit
- **Size**: 8B parameters
- **Training**: 63 billion tokens (31.5B Thai from Fineweb2-TH)
- **Features**: Enhanced instruction-following, thinking/non-thinking modes
- **Requirements**: `transformers>=4.51.0`, `torch.bfloat16` recommended
- **License**: Apache 2.0

See: https://huggingface.co/KBTG-Labs/ThaiLLM-8B-Instruct
See also: Technical Report arxiv: 2601.04597 (THaLLE-ThaiLLM)
