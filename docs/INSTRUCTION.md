# Project EZCareTech

## Dataset
- KorQuAD 2.1 dataset

## Purpose
- Finetune embedding model (`intfloat/multilingual-e5-small`) with LoRA/PEFT
- Compare original embedding vs finetuned embedding via RAG pipeline
- For RAG pipeline, save embedding with open source database (ChromaDB) in local
- Show the result in demo: make users select database with embeddings and allow them send query and get retrieval results and compare them

---

## Quick Start

### Training
```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python train.py

# Monitor with TensorBoard (optional, in a separate terminal)
./run_tensorboard.sh
# Or: tensorboard --logdir=./logs/tensorboard
# Then open: http://localhost:6006
```

### Viewing Experiment Configurations

After training, check what hyperparameters were used for each experiment:

```bash
# List all available models
python utils/view_config.py

# View config for a specific experiment
python utils/view_config.py ./logs/tensorboard/run_20251103_143022/model

# Compare two experiments side-by-side
python utils/view_config.py \
  ./logs/tensorboard/run_20251103_143022/model \
  ./logs/tensorboard/run_20251103_150535/model
```

### Inference

After training, models are saved in the TensorBoard run directory.

**Find your model:**
```bash
# List available trained models
python utils/view_config.py

# Or manually browse:
ls -lt logs/tensorboard/
```

**Use the model:**
```python
from inference import E5KorQuADInference

# Load finetuned model from your experiment run
model_path = "./logs/tensorboard/run_20251103_143022/model"
model = E5KorQuADInference(model_path)

# Search for relevant passages
results = model.search(
    query="스위스의 수도는?",
    passages=["베른은 스위스의 수도이다.", "취리히는 스위스 최대 도시이다."],
    top_k=3
)

# Display results
for r in results:
    print(f"Rank {r['rank']}: {r['passage']} (Score: {r['score']:.4f})")
```

**Or run built-in examples:**
```bash
# Edit inference.py to set your model path, then run:
python inference.py
```

---

## RAG Pipeline & Comparison Demo

Compare original vs finetuned embeddings using a full RAG pipeline with ChromaDB.

### 1. Build Vector Databases

```bash
# Build ChromaDB databases for both models
./run_build_database.sh

# Or manually build each collection separately:
# 1. Build original embeddings
python pipeline/build_database.py \
    --dataset-root /mnt/d/datasets/KorQuAD \
    --model-path "intfloat/multilingual-e5-small" \
    --model-type "original" \
    --collection-name "original_embeddings" \
    --db-path ./chroma_db

# 2. Build finetuned embeddings
python pipeline/build_database.py \
    --dataset-root /mnt/d/datasets/KorQuAD \
    --model-path ./logs/tensorboard/run_20251103_083449/model \
    --model-type "finetuned" \
    --collection-name "finetuned_embeddings" \
    --db-path ./chroma_db
```

This creates two ChromaDB collections:
- `original_embeddings` - Using base model
- `finetuned_embeddings` - Using your trained model

**Time**: ~10-15 minutes on GPU

### 2. Launch Interactive Demo

```bash
# Start Gradio web interface
./run_demo.sh

# Then open: http://localhost:7860
```

The demo allows you to:
- Enter Korean queries
- Compare retrieval results side-by-side
- Adjust top-k results
- Select which model(s) to query

### 3. Command-Line Comparison

```bash
python pipeline/retriever.py \
    --query "스위스의 수도는?" \
    --top-k 5
```

### 4. Python API

```python
from pipeline.retriever import RAGRetriever

# Initialize retriever
retriever = RAGRetriever(
    db_path="./chroma_db",
    finetuned_model_path="./logs/tensorboard/run_20251103_083449/model"
)

# Compare both models
retriever.compare_search("스위스의 수도는?", top_k=5)

# Or get results programmatically
results = retriever.search(
    query="한국의 전통 음식은?",
    top_k=3,
    model_type="both"  # "original", "finetuned", or "both"
)
```

**See `pipeline/README.md` for detailed RAG pipeline documentation.**

**See `docs/UPDATE.md` for detailed documentation.**
