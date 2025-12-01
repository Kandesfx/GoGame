#!/bin/bash
# Script để xử lý dataset theo từng năm

# Configuration
INPUT_DIR="data/raw_sgf"
OUTPUT_DIR="data/processed"
LABELED_DIR="data/datasets"
YEARS=(2015 2016 2017 2018 2019 2020 2021 2022 2023 2024)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Processing SGF files by year${NC}"
echo -e "${BLUE}========================================${NC}"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LABELED_DIR"

# Process each year
for YEAR in "${YEARS[@]}"; do
    echo -e "\n${GREEN}Processing year: $YEAR${NC}"
    echo -e "${YELLOW}----------------------------------------${NC}"
    
    # Step 1: Parse SGF files
    echo -e "${BLUE}Step 1: Parsing SGF files for $YEAR...${NC}"
    python scripts/parse_sgf_local.py \
        --input "$INPUT_DIR" \
        --output "$OUTPUT_DIR" \
        --year "$YEAR" \
        --board-sizes 9 13 19 \
        --workers 8 \
        --min-positions 10
    
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}⚠️  Warning: Parsing for $YEAR had errors${NC}"
    fi
    
    # Step 2: Generate labels for each board size
    for BOARD_SIZE in 9 13 19; do
        INPUT_FILE="$OUTPUT_DIR/positions_${BOARD_SIZE}x${BOARD_SIZE}_${YEAR}.pt"
        OUTPUT_FILE="$LABELED_DIR/labeled_${BOARD_SIZE}x${BOARD_SIZE}_${YEAR}.pt"
        
        if [ -f "$INPUT_FILE" ]; then
            echo -e "${BLUE}Step 2: Generating labels for ${BOARD_SIZE}x${BOARD_SIZE} ($YEAR)...${NC}"
            python scripts/generate_labels_local.py \
                --input "$INPUT_FILE" \
                --output "$OUTPUT_FILE" \
                --filter-handicap \
                --workers 8 \
                --batch-size 5000
            
            if [ $? -ne 0 ]; then
                echo -e "${YELLOW}⚠️  Warning: Label generation for ${BOARD_SIZE}x${BOARD_SIZE} ($YEAR) had errors${NC}"
            fi
        else
            echo -e "${YELLOW}⚠️  Skipping ${BOARD_SIZE}x${BOARD_SIZE} ($YEAR) - no input file${NC}"
        fi
    done
    
    echo -e "${GREEN}✅ Completed processing year: $YEAR${NC}"
done

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}All years processed!${NC}"
echo -e "${GREEN}========================================${NC}"

# Summary
echo -e "\n${BLUE}Summary:${NC}"
echo "Processed files are in: $OUTPUT_DIR"
echo "Labeled datasets are in: $LABELED_DIR"
echo ""
echo "To merge all years, use:"
echo "  python scripts/merge_datasets.py --input $LABELED_DIR --output $LABELED_DIR/merged"

