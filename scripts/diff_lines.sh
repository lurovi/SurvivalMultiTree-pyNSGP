#!/bin/bash

# This .sh takes a file 1 and a file 2 as inputs. It provides a file 3 which have the lines that are in file 1 but not in file 2. Empty lines are excluded.

# Check if three arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <file1> <file2> <output_file>"
    exit 1
fi

file1=$1
file2=$2
output_file=$3

# Check if the files exist
if [ ! -f "$file1" ] || [ ! -f "$file2" ]; then
    echo "Error: One or both input files do not exist."
    exit 1
fi

# Remove empty lines from both input files before processing
grep -v -E '^\s*$' "$file1" | grep -v -F -x -f <(grep -v -E '^\s*$' "$file2") > "$output_file"

echo "Differences saved in '$output_file'."

