# Advanced Topics: Contrastive Learning & Embedding Models

This document provides resources for deep understanding of contrastive learning, addressing its limitations (especially false negatives), and advanced techniques for training embedding models.

---

## Table of Contents

1. [Contrastive Learning Fundamentals](#contrastive-learning-fundamentals)
2. [False Negative Problem & Solutions](#false-negative-problem--solutions)
3. [Advanced Loss Functions](#advanced-loss-functions)
4. [Negative Sampling Strategies](#negative-sampling-strategies)
5. [Embedding Model Architectures](#embedding-model-architectures)
6. [Practical Implementation Guides](#practical-implementation-guides)
7. [Evaluation & Analysis](#evaluation--analysis)
8. [Related Topics](#related-topics)

---

## Contrastive Learning Fundamentals

### Essential Papers

#### 📄 SimCLR (Google, 2020)
**A Simple Framework for Contrastive Learning of Visual Representations**
- **Paper**: [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
- **Key Insights**:
  - Large batch sizes crucial for good negatives
  - Temperature scaling importance
  - Data augmentation strategies
- **Why Read**: Foundation of modern contrastive learning

#### 📄 MoCo (Facebook, 2020)
**Momentum Contrast for Unsupervised Visual Representation Learning**
- **Paper**: [arXiv:1911.05722](https://arxiv.org/abs/1911.05722)
- **Key Insights**:
  - Queue-based negative sampling
  - Momentum encoder reduces false negative impact
  - Memory bank for consistent representations
- **Why Read**: Addresses scalability and consistency issues

#### 📄 Contrastive Representation Learning (Survey, 2021)
**A Survey on Contrastive Self-supervised Learning**
- **Paper**: [arXiv:2011.00362](https://arxiv.org/abs/2011.00362)
- **Why Read**: Comprehensive overview of the field

### Tutorials & Courses

#### 🎓 Stanford CS330: Deep Multi-Task and Meta Learning
- **URL**: [cs330.stanford.edu](https://cs330.stanford.edu/)
- **Lectures**: Contrastive Learning (Week 5-6)
- **Level**: Graduate
- **Format**: Video lectures + slides

#### 🎓 Lil'Log: Contrastive Representation Learning
- **URL**: [lilianweng.github.io/posts/2021-05-31-contrastive](https://lilianweng.github.io/posts/2021-05-31-contrastive/)
- **Author**: Lilian Weng (OpenAI)
- **Level**: Intermediate
- **Why Read**: Clear explanations with math + intuition

#### 🎥 Yannic Kilcher - SimCLR Explained
- **URL**: [YouTube](https://www.youtube.com/watch?v=APki8LmdJwY)
- **Duration**: ~30 minutes
- **Level**: Beginner-Intermediate
- **Why Watch**: Visual explanations of key concepts

---

## False Negative Problem & Solutions

### Core Papers Addressing False Negatives

#### 📄 Supervised Contrastive Learning (Google, 2020)
**Supervised Contrastive Learning**
- **Paper**: [arXiv:2004.11362](https://arxiv.org/abs/2004.11362)
- **Key Innovation**: Use label information to identify true positives/negatives
- **Solution**:
  ```python
  # Instead of 1 positive per query:
  positives = all_samples_with_same_label

  # Pull ALL positives closer, not just one
  loss = -log(Σ_positive exp(sim) / Σ_all exp(sim))
  ```
- **Drawback**: Requires labels (not always available)
- **When to Use**: When you have class labels or can cluster similar samples

#### 📄 Hard Negative Mining (Facebook, 2021)
**Understanding Hard Negatives in Noise Contrastive Estimation**
- **Paper**: [arXiv:2104.06749](https://arxiv.org/abs/2104.06749)
- **Key Insight**: Not all negatives are equal
- **Solution**:
  - Mine "hard negatives" (similar but incorrect)
  - Ignore "too hard" negatives (likely false negatives)
- **Implementation**:
  ```python
  # Weight negatives by difficulty
  hard_negatives = negatives where 0.3 < similarity < 0.7
  easy_negatives = negatives where similarity < 0.3
  ```

#### 📄 Debiased Contrastive Learning (2020)
**Debiased Contrastive Learning**
- **Paper**: [arXiv:2007.00224](https://arxiv.org/abs/2007.00224)
- **Key Innovation**: Estimate false negative probability and correct loss
- **Solution**:
  ```python
  # Estimate: P(sample is false negative)
  p_false = estimate_false_negative_rate()

  # Adjust loss to reduce false negative impact
  adjusted_loss = debias(loss, p_false)
  ```
- **Why Read**: Theoretical framework for handling noisy negatives

### Practical Strategies

#### Strategy 1: Larger Batch Sizes
**Implementation**:
```python
# Current: batch_size = 16
# Better: batch_size = 64 or 128

# Reduces P(false negative in batch)
# 16 samples → ~3% false negative rate
# 128 samples → ~0.4% false negative rate
```

**Trade-off**: Higher memory usage, may need gradient accumulation

**Resource**: [Scaling Batch Size in Contrastive Learning](https://arxiv.org/abs/2006.09882)

#### Strategy 2: Momentum Contrast (MoCo)
**Key Idea**: Maintain queue of past negatives with consistent encoder

```python
# Instead of only current batch negatives:
negatives = current_batch + queue_of_past_encodings

# Queue size: 4096-65536 samples
# Momentum encoder: θ_momentum = 0.999 * θ_momentum + 0.001 * θ_current
```

**Benefit**: More negatives without larger batches, more consistent representations

**Resource**: [MoCo v2 Paper](https://arxiv.org/abs/2003.04297)

#### Strategy 3: Nearest Neighbor Deduplication
**Key Idea**: Remove near-duplicates before training

```python
# Pre-processing step:
1. Encode all samples with pre-trained model
2. Find k-nearest neighbors for each sample
3. If similarity > threshold (e.g., 0.9), merge/remove
4. Train on deduplicated dataset
```

**When to Use**: When you suspect many semantic duplicates in data

**Resource**: [Semantic Deduplication](https://arxiv.org/abs/2107.06499)

#### Strategy 4: Multi-Positive Contrastive Learning
**Key Idea**: Use multiple positives per query when available

```python
# For question answering:
Q: "Name a fruit"
Positives: ["Apple", "Banana", "Orange"]  # Multiple valid answers

# Loss pulls query close to ALL positives
loss = -log(Σ_positives exp(sim) / Σ_all exp(sim))
```

**When to Use**: When you can identify multiple valid answers

**Resource**: [Multi-Positive Learning](https://arxiv.org/abs/2306.00984)

---

## Advanced Loss Functions

### Beyond Simple Contrastive Loss

#### 1. **InfoNCE Loss** (Original Contrastive Loss)
```python
loss = -log(exp(sim(q, k+)) / Σ exp(sim(q, k)))
```
- **Paper**: [CPC Paper](https://arxiv.org/abs/1807.03748)
- Used in our implementation

#### 2. **SupCon Loss** (Supervised Contrastive)
```python
loss = -Σ_positives log(exp(sim(q, p)) / Σ exp(sim(q, k)))
```
- **Paper**: [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
- Pulls multiple positives together

#### 3. **Triplet Loss with Margin**
```python
loss = max(0, sim(q, neg) - sim(q, pos) + margin)
```
- **Paper**: [FaceNet](https://arxiv.org/abs/1503.03832)
- Classic approach, requires triplet mining

#### 4. **Circle Loss**
```python
# Adaptive margin based on similarity
loss = log(1 + Σ exp(α * (sp - Δp)) * exp(α * (Δn - sn)))
```
- **Paper**: [Circle Loss](https://arxiv.org/abs/2002.10857)
- Unified framework for pair/triplet losses

#### 5. **Cosine Contrastive Loss (CoSent)**
```python
# Direct cosine similarity optimization
loss = (1 - cos(q, pos)) + max(0, cos(q, neg) - threshold)
```
- **Paper**: [CoSENT](https://arxiv.org/abs/2208.08837)
- Good for asymmetric tasks (Q-A matching)

---

## Negative Sampling Strategies

### Hard Negative Mining

#### Paper: **ANCE (Approximate Nearest Neighbor Negative Contrastive Learning)**
- **URL**: [arXiv:2007.00808](https://arxiv.org/abs/2007.00808)
- **Key Idea**: Mine hard negatives using ANN search
- **Implementation**:
  ```python
  # Step 1: Build ANN index from current model
  index = build_ann_index(all_passages, model)

  # Step 2: For each query, find hard negatives
  hard_negs = index.search(query, k=10)
  hard_negs = hard_negs[not in ground_truth]

  # Step 3: Train with hard negatives
  loss = contrastive_loss(query, positive, hard_negs)
  ```
- **When to Use**: When random negatives are too easy

#### Paper: **Dense Passage Retrieval (DPR)**
- **URL**: [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
- **Strategy**: In-batch negatives + BM25 hard negatives
- **Why Read**: Standard baseline for neural retrieval

### Dynamic Hard Negative Mining

#### **Curriculum Learning for Negatives**
- **Paper**: [arXiv:2010.00825](https://arxiv.org/abs/2010.00825)
- **Strategy**: Start with easy negatives, gradually increase difficulty
- **Implementation**:
  ```python
  # Epoch 1-5: Random negatives
  # Epoch 6-10: Semi-hard negatives (0.3 < sim < 0.6)
  # Epoch 11+: Hard negatives (0.6 < sim < 0.8)
  ```

---

## Embedding Model Architectures

### Sentence Transformers (Your Base)

#### 📚 **Sentence-BERT Paper**
- **URL**: [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)
- **Why Read**: Foundation of your model architecture

#### 📚 **E5 Embeddings (Your Model!)**
**Text Embeddings by Weakly-Supervised Contrastive Pre-training**
- **URL**: [arXiv:2212.03533](https://arxiv.org/abs/2212.03533)
- **Key Insights**:
  - Instruction prefixes ("query:", "passage:")
  - Weakly-supervised pre-training
  - Multi-stage training strategy
- **Why Read**: Understand your base model's training

#### 📚 **BGE (Beijing Academy of AI)**
**C-Pack: Packaged Resources To Advance General Chinese Embedding**
- **URL**: [arXiv:2309.07597](https://arxiv.org/abs/2309.07597)
- **Key Insights**:
  - Cross-lingual embeddings
  - Instruction tuning for embeddings
  - Hard negative mining strategies

### Advanced Architectures

#### **ColBERT (Late Interaction)**
- **Paper**: [arXiv:2004.12832](https://arxiv.org/abs/2004.12832)
- **Key Innovation**: Token-level interactions, not just sentence-level
- **When Better**: When fine-grained matching matters

#### **SPLADE (Sparse Lexical and Expansion)**
- **Paper**: [arXiv:2107.05720](https://arxiv.org/abs/2107.05720)
- **Key Innovation**: Learns sparse representations (interpretable)
- **When Better**: When you need explainability

---

## Practical Implementation Guides

### Hugging Face Resources

#### 🔧 **Sentence Transformers Documentation**
- **URL**: [sbert.net](https://www.sbert.net/)
- **Sections to Read**:
  - Training Overview
  - Loss Functions
  - Multi-Task Learning
  - Custom Datasets

#### 🔧 **Massive Text Embedding Benchmark (MTEB)**
- **URL**: [github.com/embeddings-benchmark/mteb](https://github.com/embeddings-benchmark/mteb)
- **Why Important**: Standard evaluation benchmark
- **Use**: Test your finetuned model against baselines

### Training Best Practices

#### 📖 **OpenAI Embeddings Training Guide**
- **URL**: [OpenAI Cookbook](https://cookbook.openai.com/examples/embedding_wikipedia_articles_for_search)
- **Topics**: Data preparation, negative sampling, evaluation

#### 📖 **Pinecone Learning Center - Embeddings**
- **URL**: [pinecone.io/learn/embeddings](https://www.pinecone.io/learn/series/nlp/dense-vector-embeddings-nlp/)
- **Topics**: Practical embedding training and deployment

---

## Evaluation & Analysis

### Evaluation Metrics

#### Paper: **BEIR Benchmark**
**BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models**
- **URL**: [arXiv:2104.08663](https://arxiv.org/abs/2104.08663)
- **Metrics**:
  - Recall@k
  - MRR (Mean Reciprocal Rank)
  - NDCG (Normalized Discounted Cumulative Gain)
- **Code**: [github.com/beir-cellar/beir](https://github.com/beir-cellar/beir)

### Analysis Tools

#### 🛠️ **Embedding Projector (TensorBoard)**
- **URL**: [projector.tensorflow.org](https://projector.tensorflow.org/)
- **Use**: Visualize embedding space with t-SNE/UMAP
- **Tutorial**: [TensorBoard Embeddings](https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin)

#### 🛠️ **Manifold Visualization**
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Reduce to 2D
tsne = TSNE(n_components=2)
embeddings_2d = tsne.fit_transform(embeddings)

# Visualize
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
plt.title("Embedding Space Visualization")
```

---

## Related Topics

### LoRA & Parameter-Efficient Finetuning

#### 📄 **LoRA Paper**
**Low-Rank Adaptation of Large Language Models**
- **URL**: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **Why Read**: Understand your training method

#### 📄 **LoRA Variants**
- **QLoRA**: [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) - 4-bit quantization + LoRA
- **AdaLoRA**: [arXiv:2303.10512](https://arxiv.org/abs/2303.10512) - Adaptive rank allocation

### Retrieval-Augmented Generation (RAG)

#### 📄 **RAG Paper (Facebook AI)**
- **URL**: [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- **Why Read**: Understand downstream use of embeddings

#### 🔧 **LangChain RAG Tutorials**
- **URL**: [langchain.com/docs/use_cases/question_answering](https://python.langchain.com/docs/use_cases/question_answering/)
- **Why Important**: Practical RAG implementation

### Korean NLP Specific

#### 📚 **KorQuAD Papers**
- **KorQuAD 1.0**: [arXiv:1909.07005](https://arxiv.org/abs/1909.07005)
- **KorQuAD 2.0**: [arXiv:2001.03051](https://arxiv.org/abs/2001.03051)
- **Why Read**: Understand your dataset characteristics

#### 🔧 **Korean BERT Models**
- **KoBERT**: [github.com/SKTBrain/KoBERT](https://github.com/SKTBrain/KoBERT)
- **KR-BERT**: [github.com/snunlp/KR-BERT](https://github.com/snunlp/KR-BERT)
- **Why Check**: Alternative base models for Korean

---

## Recommended Learning Path

### Beginner → Intermediate

1. **Week 1-2: Fundamentals**
   - Read: Lil'Log Contrastive Learning tutorial
   - Watch: Yannic Kilcher SimCLR video
   - Implement: Simple contrastive loss from scratch

2. **Week 3-4: False Negatives**
   - Read: Supervised Contrastive Learning paper
   - Read: Debiased Contrastive Learning paper
   - Experiment: Compare different batch sizes on your data

3. **Week 5-6: Advanced Techniques**
   - Read: MoCo paper
   - Read: Hard Negative Mining paper
   - Implement: One advanced technique (e.g., MoCo or hard negatives)

### Intermediate → Advanced

4. **Week 7-8: Architecture Deep Dive**
   - Read: E5 embeddings paper (your base model)
   - Read: BGE paper
   - Analyze: Your model's embedding space with t-SNE

5. **Week 9-10: Evaluation & Production**
   - Read: BEIR benchmark paper
   - Test: Your model on MTEB benchmark
   - Deploy: Simple RAG system with your embeddings

---

## Quick Reference: Addressing False Negatives

### Problem Severity Assessment

```
Low Risk (Your Case):
✅ Specific factual Q-A pairs (KorQuAD)
✅ Diverse answer types
✅ Random sampling unlikely to hit similar pairs
→ Current approach is good

Medium Risk:
⚠️ Open-ended questions
⚠️ Multiple valid answers per question
→ Consider: Larger batch size (32-64)

High Risk:
❌ Many semantic duplicates in data
❌ Ambiguous questions
❌ High answer overlap
→ Need: Hard negative mining + deduplication
```

### Solution Priority (Effort vs Impact)

1. **Lowest Effort, High Impact**:
   - ✅ Increase batch size (16 → 32 or 64)
   - ✅ Train longer (false negatives average out)

2. **Medium Effort, Medium Impact**:
   - 🔄 Implement MoCo (queue-based negatives)
   - 🔄 Add hard negative mining

3. **High Effort, High Impact**:
   - 🔧 Supervised contrastive learning (requires clustering)
   - 🔧 Debiased contrastive learning (complex math)

4. **Research-Level**:
   - 🔬 Custom loss function for your domain
   - 🔬 Multi-stage curriculum learning

---

## Additional Resources

### Communities & Forums

- **Reddit**: [r/MachineLearning](https://reddit.com/r/MachineLearning)
- **Discord**: Hugging Face Discord (embedding channels)
- **Twitter**: Follow @_arohan_, @TheRasbt, @pschwllr (embedding researchers)

### Datasets for Practice

- **MS MARCO**: Large-scale passage ranking
- **Natural Questions**: Google's QA dataset
- **BEIR**: 18 diverse retrieval tasks
- **KorQuAD**: Your current dataset (Korean QA)

### Tools & Libraries

- **Sentence Transformers**: Your current framework
- **FAISS**: Fast similarity search (Facebook AI)
- **Qdrant**: Vector database
- **Weaviate**: Vector search engine

---

## Citation

If you use techniques from this guide, consider citing the original papers. Key citations:

```bibtex
@article{chen2020simple,
  title={A simple framework for contrastive learning of visual representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  journal={ICML},
  year={2020}
}

@article{khosla2020supervised,
  title={Supervised contrastive learning},
  author={Khosla, Prannay and Teterwak, Piotr and Wang, Chen and others},
  journal={NeurIPS},
  year={2020}
}

@article{wang2022text,
  title={Text embeddings by weakly-supervised contrastive pre-training},
  author={Wang, Liang and Yang, Nan and Huang, Xiaolong and others},
  journal={arXiv:2212.03533},
  year={2022}
}
```

---

**Last Updated**: 2025-11-02

**Maintained by**: Claude Code Training Pipeline

**Feedback**: Open an issue or submit a PR with additional resources!
