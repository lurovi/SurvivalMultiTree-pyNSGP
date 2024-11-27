#!/bin/bash

start=$(date +%s)
parallel --jobs ${2} --colsep ',' --ungroup ./run_single_main.sh {1} {2} {3} {4} {5} {6} {7} :::: ${1}
end=$(date +%s)

