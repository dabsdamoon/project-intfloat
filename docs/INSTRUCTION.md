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

**See `docs/UPDATE.md` for detailed documentation.**
