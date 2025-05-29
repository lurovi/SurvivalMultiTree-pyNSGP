#!/bin/bash

# This .sh counts the (non-empty) lines in the input file.

# Check if a file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <file>"
    exit 1
fi

file=$1

# Check if the file exists
if [ ! -f "$file" ]; then
    echo "Error: File '$file' not found."
    exit 1
fi

# Count non-empty lines using awk
line_count=$(awk 'NF' "$file" | awk 'END {print NR}')

# Print the result
echo "Number of non-empty lines in '$file': $line_count"

