# KorQuAD Dataset - Data Structure

## Overview

KorQuAD (Korean Question Answering Dataset) is a machine reading comprehension dataset where the task is to answer questions based on a given context (Wikipedia articles in HTML format).

---

## JSON File Structure

### Top-Level Structure

```json
{
  "version": "KorQuAD_2.0_train",
  "data": [ ... ]
}
```

**Keys:**
- `version`: Dataset version identifier (e.g., "KorQuAD_2.0_train")
- `data`: Array of article objects

---

## Article Object Structure

Each element in the `data` array represents a Wikipedia article with associated questions and answers:

```json
{
  "title": "예고범",
  "url": "https://ko.wikipedia.org/wiki/예고범",
  "context": "<!DOCTYPE html>...",
  "raw_html": "...",
  "qas": [ ... ]
}
```

**Keys:**

| Key | Type | Description |
|-----|------|-------------|
| `title` | string | Article title (Korean text) |
| `url` | string | Source Wikipedia URL |
| `context` | string | Full HTML content of the article |
| `raw_html` | string | Original HTML markup |
| `qas` | array | Question-Answer pairs for this article |

---

## Question-Answer (QA) Structure

Each element in the `qas` array contains a question and its answer:

```json
{
  "question": "드라마 예고범의 감독은 누구일까?",
  "answer": {
    "text": "나카무라 요시히로, 히라바야시 카츠토시, 사와다 메구미",
    "answer_start": 6302,
    "html_answer_start": 21842,
    "html_answer_text": "나카무라 요시히로, 히라바야시 카츠토시, 사와다 메구미"
  },
  "id": 8089
}
```

**Keys:**

| Key | Type | Description |
|-----|------|-------------|
| `question` | string | Question in Korean |
| `answer` | object | Answer object containing answer details |
| `id` | integer | Unique identifier for this QA pair |

### Answer Object

| Key | Type | Description |
|-----|------|-------------|
| `text` | string | Answer text (plain text) |
| `answer_start` | integer | Character position where answer starts in plain text |
| `html_answer_start` | integer | Character position where answer starts in HTML |
| `html_answer_text` | string | Answer text as it appears in HTML context |

---

## Hierarchy Summary

```
JSON File
│
├── version (string)
│
└── data (array)
    │
    └── Article Object
        │
        ├── title (string)
        ├── url (string)
        ├── context (string - full HTML)
        ├── raw_html (string)
        │
        └── qas (array)
            │
            └── QA Object
                │
                ├── question (string)
                ├── id (integer)
                │
                └── answer (object)
                    │
                    ├── text (string)
                    ├── answer_start (integer)
                    ├── html_answer_start (integer)
                    └── html_answer_text (string)
```

---

## Data Statistics

Based on initial analysis:

- **Files per directory:** ~3 JSON files per training directory
- **Articles per file:** ~1,000 articles
- **QA pairs per file:** ~1,700 pairs
- **Total estimated:** ~36,000 articles, ~61,819 QA pairs

### Text Characteristics

**Questions:**
- Average length: ~32 characters
- Range: 10-94 characters
- Common types: 무엇 (what), 몇 (how many), 언제 (when), 누구 (who)

**Answers:**
- Average length: ~200 characters
- Range: 1-6,436 characters
- Median: 12 characters (mix of short factual and long contextual answers)

---

## Example Data Flow

1. **Load JSON file** → Get `version` and `data` array
2. **Iterate through `data`** → Access each article object
3. **For each article** → Extract `title`, `url`, `context`, and `qas`
4. **For each QA in `qas`** → Get `question`, `answer`, and `id`
5. **Extract answer details** → Get `text` and position information

---

## Key Insights

- **HTML Context**: Articles are stored as full HTML, which includes markup, tags, and metadata
- **Answer Positions**: Two position markers (plain text and HTML) allow for flexible text extraction
- **Korean Language**: All text is in Korean (UTF-8 encoding required)
- **Wikipedia Source**: All articles sourced from Korean Wikipedia
- **Reading Comprehension**: Questions require understanding the context to answer correctly

---

## Usage in Analysis Scripts

```python
import json

# Load file
with open('korquad2.1_train_00.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Access structure
version = data['version']
articles = data['data']

# Iterate through articles
for article in articles:
    title = article['title']
    qas = article['qas']

    # Process each QA pair
    for qa in qas:
        question = qa['question']
        answer_text = qa['answer']['text']
        answer_position = qa['answer']['answer_start']
```

---

## Notes

- All string fields use UTF-8 encoding for Korean characters (한글)
- HTML context includes complete page structure with tags
- Answer positions are zero-indexed
- Some answers may contain HTML tags if extracted from formatted text
- Questions are natural language queries about the article content
