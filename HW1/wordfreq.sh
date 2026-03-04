#!/usr/bin/env bash

# == check argument count
if [ "$#" -ne 1 ];then
	echo "Error: missing argument"
	exit 1
fi

FILE="$1"

# Check if file exists
if [ ! -f "$FILE" ];then
	echo "Error: file not found"
	exit 1
fi

# Pring header
echo "Word Frequency"

# Process file
cat "$FILE" \
| tr 'A-Z' 'a-z' \
| tr -cs 'a-z' '\n' \
| sort \
| uniq -c \
| sort -k2,2 \
| awk '{print $2, $1}'
