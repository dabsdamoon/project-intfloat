# Project Updates

## 2025-11-02 - Refactor: Separate Configuration Module

### Changes
- **Created `config.py`**: Separated `TrainingConfig` dataclass into dedicated module
- **Updated `train.py`**: Imports config from `config.py`, cleaner code organization
- **Added `TrainingConfig.print_summary()`**: Pretty-formatted configuration display
- **Added `TrainingConfig.to_dict()`**: Configuration serialization for logging

### Benefits
- Better separation of concerns (config vs training logic)
- Easier to modify hyperparameters without editing main training code
- Reusable configuration across different scripts (future: evaluation, inference)
- Improved code readability

---

## 2025-11-02 - Training Pipeline for Embedding Model Finetuning

### Changes
- **Created `train.py`**: Complete training pipeline for finetuning `intfloat/multilingual-e5-small` with LoRA/PEFT
- **Created `requirements.txt`**: Dependencies for training (torch, transformers, sentence-transformers, peft, chromadb)
- **Cleaned up `dataloader/loader.py`**: Removed debug breakpoint (ipdb.set_trace)

### Features
- **Model**: `intfloat/multilingual-e5-small` (118M params, Korean-compatible)
- **Method**: LoRA/PEFT for efficient training on RTX 2080 8GB
- **Data Processing**: KorQuAD → (question, context) pairs with E5 instruction prefixes
- **Training Config**: Optimized for 8GB GPU (batch_size=16, fp16, gradient checkpointing)
- **Evaluation**: InformationRetrievalEvaluator on validation split
- **Output**: Saves both full model and LoRA weights separately

### LoRA Configuration
- Rank (r): 8
- Alpha: 16
- Dropout: 0.1
- Target modules: ["query", "key", "value"]
- Reduces trainable params to ~2% of total

### Training Hyperparameters
- Batch size: 16
- Epochs: 3
- Learning rate: 2e-5
- Max sequence length: 256
- Warmup steps: 500
- Train/val split: 90/10
- Loss: MultipleNegativesRankingLoss

### Rationale
- Chose multilingual-e5-small over bge-small-en-v1.5 for better Korean support
- BAAI/bge-m3 (560M) too large for 8GB GPU
- LoRA enables finetuning on consumer GPU with minimal quality loss

---

## 2025-11-01 - Move Loader to Separate Package

### Changes
- **Moved `eda/loader.py` → `dataloader/loader.py`**: Pulled loader out of eda directory into its own package
- **Created `dataloader/__init__.py`**: Package initialization with exported functions
- **Updated `eda/korquad_analysis.py`**: Changed import from `from . import loader` to `from dataloader import loader`
- **Updated CLAUDE.md**: Reflected new directory structure

### Rationale
- Emphasizes that dataloader is a standalone, reusable package (not tied to eda)
- Cleaner separation of concerns: dataloader handles I/O, eda handles analysis

---

## 2025-11-01 - Refactor: Separate Loader Module

### Changes
- **Created `eda/loader.py`**: Extracted multiprocessing file loading logic into reusable module
- **Refactored `eda/korquad_analysis.py`**: Now uses loader module; statistics computed from whole accumulated dataset (not per-chunk)
- **Created `eda/__init__.py`**: Added package init file for proper imports
- **Updated CLAUDE.md**: Documented new architecture with loader module

### Architecture Change
- **Before**: `korquad_analysis.py` handled both loading and statistics in chunks, merged chunk results
- **After**: `loader.py` loads all files in parallel → accumulates into one dataset → `korquad_analysis.py` computes statistics from whole dataset

### Rationale
- Loader will be reused for future tasks (not just analysis)
- Multiprocessing purpose: speed up I/O-bound file loading, not analysis
- Computing statistics from whole dataset (not merging per-chunk stats) ensures accuracy for metrics like median

### Testing
- Single file mode: ✓ Works
- Full analysis (2 workers): ✓ Successfully loaded 36 files, computed stats for 35,496 articles, 78,119 QA pairs
