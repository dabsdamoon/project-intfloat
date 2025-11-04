# RAG Pipeline for Embedding Comparison

This pipeline enables comparison between original and finetuned embedding models using a RAG (Retrieval-Augmented Generation) architecture with ChromaDB.

## Overview

The pipeline consists of three main components:

1. **Embedder** (`embedder.py`) - Wrapper for loading and using embedding models
2. **Database Builder** (`build_database.py`) - Creates ChromaDB vector databases
3. **Retriever** (`retriever.py`) - Queries databases and retrieves similar documents

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Build Databases

Build ChromaDB vector databases for both original and finetuned models:

```bash
./run_build_database.sh
```

Or manually build each collection:

```bash
# Build original embeddings collection
python pipeline/build_database.py \
    --dataset-root /mnt/d/datasets/KorQuAD \
    --model-path "intfloat/multilingual-e5-small" \
    --model-type "original" \
    --collection-name "original_embeddings" \
    --db-path ./chroma_db

# Build finetuned embeddings collection
python pipeline/build_database.py \
    --dataset-root /mnt/d/datasets/KorQuAD \
    --model-path ./logs/tensorboard/run_20251103_083449/model \
    --model-type "finetuned" \
    --collection-name "finetuned_embeddings" \
    --db-path ./chroma_db
```

This will:
- Load the KorQuAD evaluation set (once per collection)
- Generate embeddings using the specified model
- Store in ChromaDB with metadata

**Time**: ~5-8 minutes per collection on GPU, ~10-16 minutes total

### 3. Query the Databases

#### Option A: Interactive Demo (Recommended)

Launch the Gradio web interface:

```bash
./run_demo.sh
```

Then open: http://localhost:7860

#### Option B: Command Line

```bash
python pipeline/retriever.py \
    --query "스위스의 수도는?" \
    --top-k 5 \
    --db-path ./chroma_db
```

#### Option C: Python API

```python
from pipeline.retriever import RAGRetriever

# Initialize retriever
retriever = RAGRetriever(
    db_path="./chroma_db",
    finetuned_model_path="./logs/tensorboard/run_20251103_083449/model"
)

# Search with both models
results = retriever.search(
    query="스위스의 수도는?",
    top_k=5,
    model_type="both"  # "original", "finetuned", or "both"
)

# Print comparison
retriever.compare_search("스위스의 수도는?", top_k=5)
```

## Architecture

### Data Format

Documents are stored in ChromaDB with the following structure:

```python
{
    "id": "doc_0",
    "document": "query: answer",  # Combined text
    "embedding": [0.123, 0.456, ...],  # 384-dim vector
    "metadata": {
        "query": "original query text",
        "answer": "original answer text",
        "index": 0
    }
}
```

### Embedding Process

1. **For Documents (stored in database)**:
   ```python
   text = f"{query}: {answer}"
   embedding = model.encode(f"passage: {text}")
   ```

2. **For Queries (at search time)**:
   ```python
   embedding = model.encode(f"query: {user_query}")
   ```

The `query:` and `passage:` prefixes follow the E5 model convention for asymmetric retrieval.

### Similarity Metric

- **Metric**: Cosine similarity
- **ChromaDB distance**: `distance = 1 - cosine_similarity`
- **Score returned**: `score = 1 - distance = cosine_similarity`

## Files

```
pipeline/
├── __init__.py              # Package init
├── embedder.py              # Embedding model wrapper
├── build_database.py        # Database creation script
├── retriever.py             # Query and retrieval logic
└── README.md                # This file

demo/
├── __init__.py              # Package init
└── app.py                   # Gradio web interface

chroma_db/                   # ChromaDB storage (created after build)
├── chroma.sqlite3           # SQLite index
└── <collection_data>        # Vector data
```

## Configuration

### Default Paths

- **Dataset**: `/mnt/d/datasets/KorQuAD`
- **Finetuned Model**: `./logs/tensorboard/run_20251103_083449/model`
- **Database**: `./chroma_db`

### Changing Paths

All scripts accept command-line arguments:

```bash
python pipeline/build_database.py \
    --dataset-root /path/to/dataset \
    --finetuned-model /path/to/model \
    --db-path /path/to/db
```

## Evaluation Metrics

When comparing models, consider:

1. **Relevance**: Are the retrieved answers relevant to the query?
2. **Ranking**: Is the most relevant answer ranked first?
3. **Score Distribution**: How confident is the model (score values)?
4. **Diversity**: Does the model retrieve diverse answers?

## Troubleshooting

### Database Not Found

```
Error: Failed to load collections from ./chroma_db
```

**Solution**: Run `./run_build_database.sh` first to create the databases.

### CUDA Out of Memory

If you encounter OOM errors during database building:

1. Reduce batch size in `embedder.py`:
   ```python
   embeddings = embedder.encode(texts, batch_size=16)  # Reduce from 32
   ```

2. Or use CPU:
   ```python
   # In embedder.py, comment out GPU code
   # self.model = self.model.to('cuda')
   ```

### Model Path Issues

Ensure the finetuned model path exists:

```bash
ls -la ./logs/tensorboard/run_20251103_083449/model
```

Should contain:
- `config.json`
- `pytorch_model.bin` or `model.safetensors`
- `tokenizer_config.json`
- etc.

## Performance

### Database Build Time

- **Dataset size**: ~1000 Q-A pairs (eval set)
- **GPU (CUDA)**: ~5-10 minutes total
- **CPU**: ~20-30 minutes total

### Query Time

- **Cold start** (first query): ~2-3 seconds (model loading)
- **Subsequent queries**: ~100-200ms per query

## Example Queries

Korean questions from KorQuAD:

```
스위스의 수도는?
한국의 전통 음식은?
태양계에서 가장 큰 행성은?
서울의 인구는?
```

## Next Steps

1. **Quantitative Evaluation**: Calculate metrics (NDCG, MAP, MRR) on eval set
2. **Cross-Encoder Reranking**: Add reranker for better top results
3. **Hybrid Search**: Combine dense (embedding) + sparse (BM25) retrieval
4. **Production Deployment**: Add caching, batch processing, API endpoints

## References

- **E5 Embeddings**: [arXiv:2212.03533](https://arxiv.org/abs/2212.03533)
- **ChromaDB**: https://www.trychroma.com/
- **Sentence Transformers**: https://www.sbert.net/

---

Built with ❤️ using Claude Code
