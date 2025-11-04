#!/bin/bash

# Test script to build a small wiki database and test retrieval
# Uses only 10 chunks for quick testing

set -e  # Exit on error

echo "========================================="
echo "Testing Wiki Pipeline (10 chunks)"
echo "========================================="
echo ""

# Configuration
WIKI_FILE="data/text_cleaned.txt"
DB_PATH="./chroma_db_test"
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_CHUNKS=10  # Small number for testing

ORIGINAL_MODEL="intfloat/multilingual-e5-small"
FINETUNED_MODEL="./logs/tensorboard/run_20251103_083449/model"

ORIGINAL_COLLECTION="wiki_original_test"
FINETUNED_COLLECTION="wiki_finetuned_test"

echo "Step 1: Building test database with $MAX_CHUNKS chunks..."
echo "========================================="

# Build original model collection
python3 pipeline/build_database.py \
    --mode wiki \
    --wiki-file "$WIKI_FILE" \
    --chunk-size $CHUNK_SIZE \
    --chunk-overlap $CHUNK_OVERLAP \
    --max-chunks $MAX_CHUNKS \
    --model-path "$ORIGINAL_MODEL" \
    --model-type original \
    --collection-name "$ORIGINAL_COLLECTION" \
    --db-path "$DB_PATH"

echo ""
echo "Step 2: Building finetuned model collection..."
echo "========================================="

# Build finetuned model collection
python3 pipeline/build_database.py \
    --mode wiki \
    --wiki-file "$WIKI_FILE" \
    --chunk-size $CHUNK_SIZE \
    --chunk-overlap $CHUNK_OVERLAP \
    --max-chunks $MAX_CHUNKS \
    --model-path "$FINETUNED_MODEL" \
    --model-type finetuned \
    --collection-name "$FINETUNED_COLLECTION" \
    --db-path "$DB_PATH"

echo ""
echo "========================================="
echo "✅ Test database built successfully!"
echo "========================================="
echo ""
echo "To clean up test database:"
echo "  rm -rf $DB_PATH"
echo ""
echo "To test the full pipeline with all chunks:"
echo "  ./run_build_wiki_database.sh"
echo ""
