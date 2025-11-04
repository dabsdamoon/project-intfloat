#!/bin/bash

# Script to build ChromaDB with wiki text chunks
# Builds databases for both original and finetuned models

set -e  # Exit on error

# Configuration
WIKI_FILE="data/text_cleaned.txt"
DB_PATH="./chroma_db"
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_CHUNKS=""  # Leave empty for all chunks, or set a number for testing

# Model paths
ORIGINAL_MODEL="intfloat/multilingual-e5-small"
FINETUNED_MODEL="./logs/tensorboard/run_20251103_083449/model"

# Collection names
ORIGINAL_COLLECTION="wiki_original"
FINETUNED_COLLECTION="wiki_finetuned"

echo "========================================="
echo "Building Wiki ChromaDB Collections"
echo "========================================="
echo "Wiki file: $WIKI_FILE"
echo "Chunk size: $CHUNK_SIZE"
echo "Chunk overlap: $CHUNK_OVERLAP"
echo "Database path: $DB_PATH"
echo ""

# Build command arguments
COMMON_ARGS="--mode wiki --wiki-file $WIKI_FILE --chunk-size $CHUNK_SIZE --chunk-overlap $CHUNK_OVERLAP --db-path $DB_PATH"

if [ -n "$MAX_CHUNKS" ]; then
    COMMON_ARGS="$COMMON_ARGS --max-chunks $MAX_CHUNKS"
    echo "⚠️  Limited to $MAX_CHUNKS chunks (for testing)"
fi

echo ""
echo "========================================="
echo "Step 1: Building Original Model Database"
echo "========================================="
python3 pipeline/build_database.py \
    $COMMON_ARGS \
    --model-path "$ORIGINAL_MODEL" \
    --model-type original \
    --collection-name "$ORIGINAL_COLLECTION"

echo ""
echo "========================================="
echo "Step 2: Building Finetuned Model Database"
echo "========================================="
python3 pipeline/build_database.py \
    $COMMON_ARGS \
    --model-path "$FINETUNED_MODEL" \
    --model-type finetuned \
    --collection-name "$FINETUNED_COLLECTION"

echo ""
echo "========================================="
echo "✅ All collections built successfully!"
echo "========================================="
echo "Database location: $DB_PATH"
echo "Collections:"
echo "  - $ORIGINAL_COLLECTION"
echo "  - $FINETUNED_COLLECTION"
echo ""
echo "To test retrieval, run:"
echo "  python3 pipeline/retriever.py --query '볼드모트는 누구인가?' --db-path $DB_PATH"
echo ""
