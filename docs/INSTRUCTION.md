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

### Inference
```bash
# Run built-in examples
python inference.py
```

**Or use in your code:**
```python
from inference import E5KorQuADInference

# Load finetuned model
model = E5KorQuADInference("./models/finetuned-e5-small-korquad")

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

**See `docs/UPDATE.md` for detailed documentation.**
