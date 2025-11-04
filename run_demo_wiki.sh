#!/bin/bash

# Script to launch Gradio demo for Wiki RAG comparison

DB_PATH="./chroma_db"
FINETUNED_MODEL="./logs/tensorboard/run_20251103_083449/model"
ORIGINAL_COLLECTION="wiki_original"
FINETUNED_COLLECTION="wiki_finetuned"
DATA_TYPE="Wiki"

echo "========================================="
echo "Launching Wiki RAG Demo"
echo "========================================="
echo "Database: $DB_PATH"
echo "Collections:"
echo "  - Original: $ORIGINAL_COLLECTION"
echo "  - Finetuned: $FINETUNED_COLLECTION"
echo "Data Type: $DATA_TYPE"
echo ""
echo "Starting Gradio interface..."
echo "========================================="
echo ""

python3 demo/app.py \
    --db-path "$DB_PATH" \
    --finetuned-model "$FINETUNED_MODEL" \
    --original-collection "$ORIGINAL_COLLECTION" \
    --finetuned-collection "$FINETUNED_COLLECTION" \
    --data-type "$DATA_TYPE"
