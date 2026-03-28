# FahMai (ฟ้าใหม่) - RAG System Challenge

## Overview

FahMai (ฟ้าใหม่) is a fictional Thai electronics store. You are given a knowledge base of product pages, store policies, and store information (in Thai). Your task is to build a Retrieval-Augmented Generation (RAG) system that answers 100 multiple-choice questions about the store.

### Question Format

Each question has 10 choices:

- **Choices 1-8**: Content-specific answers
- **Choice 9**: "ไม่มีข้อมูลนี้ในฐานข้อมูล" (No data available in the knowledge base)
- **Choice 10**: "คำถามนี้ไม่เกี่ยวข้องกับร้านฟ้าใหม่" (This question is not related to FahMai store)

## Evaluation

- **Metric**: Accuracy (percentage of correct answers)
- **Public Leaderboard**: 60% of questions (visible during competition)
- **Private Leaderboard**: 40% of questions (revealed at end)

### Submission Format

Submit a CSV file with exactly 100 rows:

```csv
id,answer
1,5
2,3
3,7
...
100,2
```

- `id`: integer 1-100
- `answer`: integer 1-10

## Additional Information

### Team Name

การแข่งขันเป็นรูปแบบรายบุคคล โดยผู้เข้าแข่งขันจะต้องตั้งชื่อทีมให้ถูกต้องตามที่กำหนด คือ "รหัสประจำตัวผู้อบรม-ชื่อจริงภาษาไทย" เช่น "600000-นงนุช"

### Leaderboards

- แต่ละทีมจะสามารถส่งผลได้ 8 ครั้งต่อวัน (ตัดรอบวัน เวลา 07:00 น. ตามเวลาประเทศไทย)
- ข้อมูล test 60% จะวัดผลใน Public Leaderboard มีวัตถุประสงค์เพื่อใช้พัฒนา model ตลอดระยะเวลาการแข่งขัน
- ข้อมูล test อีก 40% จะวัดผลใน Private Leaderboard มีวัตถุประสงค์เพื่อใช้ตัดสินผลการแข่งขัน

## Dataset Description

### Knowledge Base (`knowledge_base/`)

Markdown files organized into three folders:

| Folder | Contents | Files |
|--------|----------|-------|
| `products/` | Product spec sheets (specs, pricing, features) | ~30 files |
| `policies/` | Store policies (returns, warranty, shipping, membership) | ~5 files |
| `store_info/` | Branch locations, contact info, promotions | ~5 files |

All documents are in Thai.

### Questions (`questions.csv`)

| Column | Description |
|--------|-------------|
| `id` | Question ID (1-100) |
| `question` | The question in Thai |
| `choice_1` … `choice_10` | Answer choices in Thai |

### Sample Submission (`sample_submission.csv`)

| Column | Description |
|--------|-------------|
| `id` | Question ID (1-100) |
| `answer` | Your predicted answer (integer 1-10) |
