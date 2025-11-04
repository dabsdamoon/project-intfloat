# OpenAI Evaluation Feature Guide

## Overview

The demo app now includes an **Evaluation** feature that uses OpenAI's `text-embedding-3-small` model as a neutral benchmark to compare retrieval quality between the original and finetuned models.

## How It Works

1. **Search with Both Models**: Select "Both" in the Model Selection dropdown and perform a search
2. **Evaluate Button Appears**: An "⚖️ Evaluate with OpenAI Embeddings" button will appear
3. **Click to Evaluate**: The system will:
   - Embed your query using OpenAI's model
   - Embed all retrieved documents from both models
   - Calculate cosine similarity scores
   - Compare average and maximum similarities
   - Determine which model retrieved more relevant content

## Setup

### 1. Install Dependencies

```bash
pip install openai python-dotenv
```

### 2. Configure API Key

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-api-key-here
```

Get your API key from: https://platform.openai.com/api-keys

### 3. Launch Demo

```bash
./run_demo_wiki.sh
```

## Evaluation Metrics

The evaluation provides:

### Similarity Scores
- **Average Similarity**: Mean cosine similarity between query and all retrieved documents
- **Max Similarity**: Best match among retrieved documents
- **Results Count**: Number of documents retrieved

### Comparison Results
- **Winner**: Which model performed better (Original, Finetuned, or Tie)
- **Performance Improvement**: Percentage difference between models
- **Average Difference**: Absolute difference in similarity scores

### Detailed Breakdown
- Individual similarity scores for each retrieved document
- Ranked from 1 to K (based on top-k setting)

## Interpretation

- **Higher similarity** = Better semantic match between query and retrieved content
- **OpenAI embeddings** serve as a neutral reference (not biased toward either model)
- Useful for **objective comparison** of retrieval quality

## Example Output

```markdown
### 📊 OpenAI Embedding Evaluation

**Evaluation Model**: text-embedding-3-small
**Query**: 볼드모트는 누구인가?

| Metric | Original Model | Finetuned Model |
|--------|---------------|-----------------|
| Average Similarity | 0.8245 | 0.8567 |
| Max Similarity | 0.8921 | 0.9103 |
| Results Count | 5 | 5 |

#### 🏆 Evaluation Result
Winner: Finetuned Model 🎯
Performance Improvement: 3.91%
Average Difference: 0.0322
```

## Cost Considerations

- Uses OpenAI API (costs apply)
- `text-embedding-3-small`: $0.00002 per 1K tokens
- Example: Evaluating 5 documents ~500 chars each ≈ $0.0001
- Very affordable for occasional evaluation

## Troubleshooting

### Error: "OPENAI_API_KEY not found"
- Ensure `.env` file exists in project root
- Check that `OPENAI_API_KEY` is set correctly
- Restart the demo after updating `.env`

### Error: "Authentication failed"
- Verify your API key is valid
- Check you have credits in your OpenAI account

### Button Not Appearing
- Make sure you selected "Both" in Model Selection
- Perform a search before the button appears

## Technical Details

- **Embedding Model**: OpenAI `text-embedding-3-small` (1536 dimensions)
- **Similarity Metric**: Cosine similarity
- **Implementation**: `utils/evaluator.py`
- **API Client**: OpenAI Python SDK v1.0+

