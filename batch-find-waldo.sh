#!/bin/bash

# Ensure correct usage
if [ $# -ne 2 ]; then
    echo "Usage: ./batch_find_waldo.sh <input_directory> <output_directory>"
    exit 1
fi

# Input and output directories
INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Loop through each image in the input directory
for image in "$INPUT_DIR"/*.jpg "$INPUT_DIR"/*.png "$INPUT_DIR"/*.jpeg; do
    # Skip if no images found
    [ -e "$image" ] || continue

    # Get image filename without extension
    filename=$(basename -- "$image")
    filename_no_ext="${filename%.*}"

    # Define output file path
    output_file="$OUTPUT_DIR/${filename_no_ext}_detected.jpg"

    # Run the Python script
    echo "Processing: $image → Saving to: $output_file"
    python3 scripts/find-waldo.py "$image" "$output_file"
done

echo "Batch processing completed! Results saved in: $OUTPUT_DIR"
