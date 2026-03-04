#!/usr/bin/env bash

DIR="$1"
EXT="$2"

# Check if directory exist
if [ ! -d "$DIR" ]; then
	echo "Error: Directory $DIR does not exist."
	exit 1
fi

# Find matching files
FILES=("$DIR"/*."$EXT")

# If not files found
if [ ! -e "${FILES[0]}" ]; then
	echo "No files with the extension $EXT found in $DIR."
	exit 0
fi

# Append filenames to summary.txt
OUTPUT="$DIR/summary.txt"
for file in "${FILES[@]}"; do
	basename "$file" >> "$OUTPUT"
done

# Success message
echo "File names with extension $EXT have been added to $OUTPUT."
