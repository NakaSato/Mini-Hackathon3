# FahMai RAG System - Data Flow

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FAHMAI RAG PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │  Knowledge   │
    │    Base      │
    │  (Markdown   │
    │    Files)    │
    └──────┬───────┘
           │
           ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  PHASE 1: DOCUMENT PREPROCESSING                              │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
    │  │   Parse     │→ │   Chunk     │→ │  Normalize  │          │
    │  │  Markdown   │  │  by Section │  │  Thai Text  │          │
    │  └─────────────┘  └─────────────┘  └─────────────┘          │
    └──────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  PHASE 2: EMBEDDING & INDEXING                                │
    │  ┌─────────────────────────────────────────────────────┐     │
    │  │  Embedding Model: multilingual-e5-large             │     │
    │  │  - Converts text → 1024-dim vectors                 │     │
    │  │  - Thai-capable semantic embeddings                 │     │
    │  └─────────────────────────────────────────────────────┘     │
    │                          │                                    │
    │                          ▼                                    │
    │  ┌─────────────────────────────────────────────────────┐     │
    │  │  Vector Store: ChromaDB                              │     │
    │  │  - Stores vectors + metadata                         │     │
    │  │  - Enables fast similarity search                    │     │
    │  └─────────────────────────────────────────────────────┘     │
    └──────────────────────────────────────────────────────────────┘
           │
           │  (Index built once, reused for all questions)
           ▼
    ┌──────────────┐
    │   Questions  │
    │    (CSV)     │
    │   100 rows   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  PHASE 3: RETRIEVAL (per question)                            │
    │                                                               │
    │  Question: "Watch S3 Ultra กันน้ำได้กี่ ATM"                 │
    │                                                               │
    │  ┌──────────────────────────────────────────────────────┐    │
    │  │  1. Hybrid Search                                     │    │
    │  │     - Semantic search (embedding similarity)          │    │
    │  │     - Keyword boost (SKU, product names)              │    │
    │  │     - Re-ranking by combined score                    │    │
    │  └──────────────────────────────────────────────────────┘    │
    │                          │                                    │
    │                          ▼                                    │
    │  ┌──────────────────────────────────────────────────────┐    │
    │  │  2. Top-K Results (k=5)                               │    │
    │  │     [Doc1: Watch S3 Ultra specs] (score: 0.92)        │    │
    │  │     [Doc2: Watch S3 Ultra features] (score: 0.87)     │    │
    │  │     [Doc3: Warranty policy] (score: 0.65)             │    │
    │  │     ...                                               │    │
    │  └──────────────────────────────────────────────────────┘    │
    └──────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  PHASE 4: SPECIAL CASE DETECTION                              │
    │                                                               │
    │  ┌─────────────────┐         ┌─────────────────┐             │
    │  │  Is Unrelated?  │────────→│  Return 10      │             │
    │  │  (not FahMai)   │  Yes    │  (out-of-scope) │             │
    │  └─────────────────┘         └─────────────────┘             │
    │           │ No                                                │
    │           ▼                                                   │
    │  ┌─────────────────┐         ┌─────────────────┐             │
    │  │  No Data?       │────────→│  Return 9       │             │
    │  │  (low scores)   │  Yes    │  (no info)      │             │
    │  └─────────────────┘         └─────────────────┘             │
    │           │ No                                                │
    │           ▼                                                   │
    │  → Continue to Answer Selection                               │
    └──────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  PHASE 5: ANSWER SELECTION                                    │
    │                                                               │
    │  ┌─────────────────────────────────────────────────────┐     │
    │  │  Context Assembly                                     │     │
    │  │  - Combine top-K retrieved documents                 │     │
    │  │  - Format with metadata                              │     │
    │  │  - Max 4000 chars                                    │     │
    │  └─────────────────────────────────────────────────────┘     │
    │                          │                                    │
    │                          ▼                                    │
    │  ┌─────────────────────────────────────────────────────┐     │
    │  │  ThaiLLM-8B-Instruct                                  │     │
    │  │  Input:                                               │     │
    │  │  - System: "You are FahMai assistant..."             │     │
    │  │  - Context: [retrieved documents]                    │     │
    │  │  - Question: "Watch S3 Ultra กันน้ำได้กี่ ATM"       │     │
    │  │  - Choices: [10 options]                             │     │
    │  │                                                       │     │
    │  │  Output: "5"  (the answer number)                    │     │
    │  └─────────────────────────────────────────────────────┘     │
    └──────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  PHASE 6: SUBMISSION                                          │
    │                                                               │
    │  ┌─────────────────────────────────────────────────────┐     │
    │  │  output/submission.csv                                │     │
    │  │  id,answer                                            │     │
    │  │  1,5                                                  │     │
    │  │  2,3                                                  │     │
    │  │  ...                                                  │     │
    │  │  100,2                                                │     │
    │  └─────────────────────────────────────────────────────┘     │
    └──────────────────────────────────────────────────────────────┘
```

## Detailed Flow

### Phase 1: Document Preprocessing

**File**: `src/preprocessing.py`

```
Markdown Files (118 files)
│
├── products/ (110 files)
│   ├── SF-SP-002_saifah_phone_x9_pro.md
│   ├── DN-LT-001_daonuea_airbook_15.md
│   └── ...
│
├── policies/ (5 files)
│   ├── return_policy.md
│   ├── warranty_policy.md
│   └── ...
│
└── store_info/ (3 files)
    ├── about_fahmai.md
    └── ...

↓ parse_markdown_file()

Parsed Document:
{
  "filepath": "data/knowledge_base/products/SF-SP-002_saifah_phone_x9_pro.md",
  "filename": "SF-SP-002_saifah_phone_x9_pro",
  "content": "...",
  "sections": [
    {"title": "รายละเอียดสินค้า", "content": ["..."]},
    {"title": "สเปคสินค้า", "content": ["..."]},
    {"title": "การรับประกัน", "content": ["..."]}
  ],
  "metadata": {
    "sku": "SF-SP-002",
    "brand": "สายฟ้า",
    "category": "สมาร์ทโฟน",
    "price": "24990"
  }
}

↓ chunk_document()

Chunks:
[
  {
    "content": "สายฟ้า โฟน X9 Pro คือ Flagship สมาร์ทโฟน...",
    "metadata": {
      "sku": "SF-SP-002",
      "filename": "SF-SP-002_saifah_phone_x9_pro",
      "section": "รายละเอียดสินค้า",
      "chunk_id": 0
    }
  },
  ...
]
```

### Phase 2: Embedding & Indexing

**File**: `src/embeddings.py`

```
Chunks ( ~2000 chunks)
│
↓ EmbeddingModel.encode()
│  Model: intfloat/multilingual-e5-large
│  - Prefix: "passage: {text}"
│  - Output: 1024-dimensional vector
│  - Normalized (cosine similarity ready)
│
▼
Vector Store (ChromaDB)
│
├── Collection: "fahmai_kb"
│   ├── Embeddings: [1024-dim vectors]
│   ├── Documents: [chunk text]
│   ├── Metadatas: [{sku, filename, section, ...}]
│   └── IDs: ["{filename}_{section}_{chunk_id}"]
│
└── Index saved to: output/index/
```

### Phase 3: Retrieval

**File**: `src/retrieval.py`

```
Question: "Watch S3 Ultra กันน้ำได้กี่ ATM"
│
├─→ Step 1: Extract Keywords
│   Keywords: ["Watch", "S3", "Ultra", "ATM", "กันน้ำ"]
│
├─→ Step 2: Encode Query
│   Query Embedding: [1024-dim vector]
│   (prefix: "query: {text}")
│
├─→ Step 3: Semantic Search
│   ChromaDB.query()
│   - Cosine similarity with all stored vectors
│   - Return top 13 (k * 2.5 for re-ranking)
│
├─→ Step 4: Keyword Boost
│   For each result:
│   semantic_score = 1.0 - distance
│   keyword_score = matches / total_keywords
│   combined_score = 0.7 * semantic + 0.3 * keyword
│
├─→ Step 5: Re-rank
│   Sort by combined_score (descending)
│
└─→ Step 6: Return Top-K (k=5)
    [
      {content: "...", metadata: {...}, score: 0.92},
      {content: "...", metadata: {...}, score: 0.87},
      ...
    ]
```

### Phase 4: Special Case Detection

**File**: `src/answer_selector.py`

```
Question + Retrieval Results
│
├─→ Check Unrelated (Choice 10)
│   Keywords: ['อาหาร', 'ท่องเที่ยว', 'การเมือง', ...]
│   Electronics keywords: ['มือถือ', 'คอมพิวเตอร์', 'ฟ้าใหม่', ...]
│   │
│   ├─ Has unrelated topic AND no electronics mention
│   │  └─→ Return 10
│   │
│   └─ Otherwise → Continue
│
└─→ Check No Data (Choice 9)
    best_score = max(result['combined_score'])
    │
    ├─ best_score < 0.5
    │  └─→ Return 9
    │
    └─ Otherwise → Continue to Answer Selection
```

### Phase 5: Answer Selection

**File**: `src/answer_selector_thai.py`

```
Question + Choices + Context
│
├─→ Step 1: Format Messages
│   [
│     {"role": "system", "content": "คุณเป็นผู้ช่วยตอบคำถามเกี่ยวกับร้านฟ้าใหม่..."},
│     {"role": "user", "content": "บริบทจากฐานความรู้:\n{context}\n\nคำถาม: {question}\n\nตัวเลือก:\n{choices}..."}
│   ]
│
├─→ Step 2: Apply Chat Template
│   tokenizer.apply_chat_template()
│   - Adds ThaiLLM-specific formatting
│   - enable_thinking=False (default)
│
├─→ Step 3: Generate with ThaiLLM-8B-Instruct
│   model.generate()
│   - max_new_tokens: 10
│   - temperature: 0.1 (low for consistency)
│   - top_p: 0.9
│
├─→ Step 4: Parse Response
│   Response: "5"
│   Extract number with regex: r'\b([1-9]|10)\b'
│
└─→ Return: 5
```

### Phase 6: Submission

**File**: `src/pipeline.py`

```
Results (100 answers)
│
├─→ Create DataFrame
│   pd.DataFrame([
│     {"id": 1, "answer": 5},
│     {"id": 2, "answer": 3},
│     ...
│   ])
│
├─→ Save to CSV
│   output/submission.csv
│
└─→ Save Debug Info
    output/submission_debug.json
    - Retrieval results per question
    - Context used
    - Special case flags
```

## Component Interaction

```
┌────────────────────────────────────────────────────────────────┐
│                        main.py                                  │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              FahMaiRAGPipeline                            │  │
│  │                                                           │  │
│  │  initialize()                                             │  │
│  │    │                                                      │  │
│  │    ├─→ EmbeddingModel (embeddings.py)                    │  │
│  │    ├─→ VectorStore (embeddings.py)                       │  │
│  │    ├─→ RetrievalSystem (retrieval.py)                    │  │
│  │    ├─→ ThaiLLMAnswerSelector (answer_selector_thai.py)   │  │
│  │    └─→ SpecialCaseDetector (answer_selector.py)          │  │
│  │                                                           │  │
│  │  answer_question()                                        │  │
│  │    │                                                      │  │
│  │    ├─→ retriever.search()                                │  │
│  │    ├─→ special_case_detector.detect()                    │  │
│  │    ├─→ context_assembler.assemble()                      │  │
│  │    └─→ answer_selector.select()                          │  │
│  │                                                           │  │
│  │  answer_all_questions()                                   │  │
│  │    │                                                      │  │
│  │    └─→ Loop over 100 questions                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

## Key Data Structures

### Chunk
```python
{
    "content": "สายผ้า โฟน X9 Pro คือ Flagship สมาร์ทโฟน...",
    "metadata": {
        "sku": "SF-SP-002",
        "brand": "สายฟ้า",
        "filename": "SF-SP-002_saifah_phone_x9_pro",
        "section": "รายละเอียดสินค้า",
        "chunk_id": 0,
        "category_folder": "products"
    }
}
```

### Retrieval Result
```python
{
    "content": "Watch S3 Ultra มาพร้อมมาตรฐานกันน้ำระดับ 10 ATM...",
    "metadata": {...},
    "semantic_score": 0.95,
    "keyword_score": 0.80,
    "combined_score": 0.90,
    "distance": 0.05
}
```

### Question Row
```python
{
    "id": 1,
    "question": "Watch S3 Ultra กันน้ำได้กี่ ATM ครับ",
    "choice_1": "3 ATM",
    "choice_2": "IP68",
    ...
    "choice_10": "คำถามนี้ไม่เกี่ยวข้องกับร้านฟ้าใหม่"
}
```

## Performance Characteristics

| Phase | Time (per question) | Bottleneck |
|-------|---------------------|------------|
| Retrieval | ~50-100ms | Vector search |
| Special Case | ~10ms | Keyword matching |
| Answer Selection (Lightweight) | ~200ms | Cross-encoder |
| Answer Selection (Full ThaiLLM) | ~2-5s | LLM generation |

**Total Time (100 questions)**:
- Lightweight mode: ~5-10 minutes
- Full ThaiLLM mode: ~30-60 minutes
