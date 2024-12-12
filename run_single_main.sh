#!/bin/bash

start=$(date +%s)
python3 main.py --method ${1} --seed ${2} --dataset ${3} --normalize ${4} --test_size ${5} --config ${6} --run_id ${7} --verbose ${8}
end=$(date +%s)

