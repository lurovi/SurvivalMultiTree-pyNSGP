#!/bin/bash

# Security checks
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <file1.txt> <file2.txt>"
    exit 1
fi

file1="$1"
file2="$2"

if [ ! -f "$file1" ]; then
    echo "Error: '$file1' does not exist or is not a file."
    exit 1
fi

if [ ! -f "$file2" ]; then
    echo "Error: '$file2' does not exist or is not a file."
    exit 1
fi

# Normalize input files: remove empty lines and sort them
cleaned1=$(grep -v '^[[:space:]]*$' "$file1" | sort)
cleaned2=$(grep -v '^[[:space:]]*$' "$file2" | sort)

# Use `comm` to check if there are differences
if comm -3 <(echo "$cleaned1") <(echo "$cleaned2") | grep -q .; then
    echo "The files have different content."
else
    echo "The files have the same content."
fi

