#!/bin/bash

start=$(date +%s)
python3 main.py --method ${1} --seed ${2} --dataset ${3} --test_size ${4} --config ${5} --run_id ${6} --verbose ${7}
end=$(date +%s)

