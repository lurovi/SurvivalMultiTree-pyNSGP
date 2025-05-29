#!/bin/bash

# This .sh applies the sequential .sh to all the (non-empty) lines of the provided input file as parameters.

# Check if the correct number of arguments is provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_file> <num_jobs>"
    exit 1
fi

input_file=$1
num_jobs=$2

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: File '$input_file' not found."
    exit 1
fi

# Check if num_jobs is a valid positive integer
if ! [[ "$num_jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: num_jobs must be a positive integer."
    exit 1
fi

start=$(date +%s)

# Filter out empty lines before processing
grep -v -E '^\s*$' "$input_file" | parallel --jobs "$num_jobs" --colsep ';' --ungroup ./scripts/run_main.sh {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14}

end=$(date +%s)
