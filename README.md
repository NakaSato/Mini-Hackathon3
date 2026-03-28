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

**Local Development (Lightweight Mode - Default)**

Runs on CPU, no GPU required. Uses cross-encoder for fast answer selection.

```bash
# Build index and answer questions (lightweight mode)
python main.py

# Or explicitly specify lightweight mode
python main.py --lightweight

# Test setup
python main.py --test
```

**Cloud/Colab Mode (Full ThaiLLM)**

Runs full ThaiLLM-8B-Instruct model. Requires GPU (16GB+ VRAM recommended).

```bash
# Full ThaiLLM mode (for cloud/GPU)
python main.py --cloud

# Or use Google Colab notebook
# See FahMai_RAG_Colab.ipynb
```

### 3. Google Colab (Recommended for Full ThaiLLM)

For best performance with full ThaiLLM, use Google Colab:

1. Open [`FahMai_RAG_Colab.ipynb`](FahMai_RAG_Colab.ipynb)
2. Connect to GPU runtime (Runtime > Change runtime type > GPU)
3. Run all cells in order
4. Download submission CSV

**Note:** The Colab notebook is configured for **cloud mode** with full ThaiLLM by default.

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
  --cloud       Run with full ThaiLLM mode (for Colab/cloud with GPU)
  --lightweight Use lightweight cross-encoder (faster, less memory)
  --model MODEL ThaiLLM model name (default: KBTG-Labs/ThaiLLM-8B-Instruct)
  --top-k K     Number of documents to retrieve
  --questions PATH  Path to questions CSV
  --output PATH     Path for output submission

Modes:
  Local (default):  Uses lightweight cross-encoder only (fast, no GPU needed)
  Cloud (--cloud):  Uses full ThaiLLM-8B-Instruct (requires GPU)
```

## Configuration

Create a `.env` file for custom settings (optional):

```bash
# .env
LOCAL_RUN_THAI_LLM=false  # Set to true only for local GPU testing
EMBEDDING_MODEL=BAAI/bge-m3
TOP_K=5
```

See `.env.example` for all available options.

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

1. **Use Local Mode (default)** for faster iteration during development - no GPU needed
2. **Use Cloud Mode (`--cloud`)** or Colab for full ThaiLLM accuracy
3. **Increase `--top-k`** for better recall (default: 5)
4. **Rebuild index** with `--build` if you modify the knowledge base
5. **GPU recommended** for full ThaiLLM mode (8B model requires 16GB+ VRAM)

## Running Modes

| Mode | Command | GPU Required | Speed | Accuracy | Use Case |
|------|---------|--------------|-------|----------|----------|
| **Local** | `python main.py` | No | Fast (5-10 min) | Good | Development, testing |
| **Cloud** | `python main.py --cloud` | Yes | Slow (30-60 min) | Best | Final submission |
| **Colab** | Run notebook | Yes (free) | Slow (30-60 min) | Best | No local GPU needed |

## Troubleshooting

### Out of Memory (Local)
- Use default lightweight mode (no ThaiLLM loading)
- Don't use `--cloud` flag on local machine without sufficient GPU memory
- Reduce batch size in embeddings.py

### Out of Memory (Colab)
- Clear GPU cache (run the provided cell in notebook)
- Use lightweight mode for testing
- Restart Colab runtime

### Slow Inference
- Use lightweight mode (default for local)
- Pre-build index with `--build`
- Reduce `--top-k` value

### Thai Text Issues
- Ensure UTF-8 encoding
- Check pythainlp installation
- Verify embedding model supports Thai

### Model Loading Issues
- **Local**: ThaiLLM won't load by default (use `--cloud` only on GPU machines)
- **Colab**: Ensure GPU runtime is selected (Runtime > Change runtime type > GPU)

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
