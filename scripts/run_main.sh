#!/bin/bash

start=$(date +%s)

python3 main.py --"${1}" "${2}"  \
                --"${3}" "${4}" \
                --"${5}" "${6}" \
                --"${7}" "${8}" \
                --"${9}" "${10}" \
                --"${11}" "${12}" \
                --"${13}" "${14}" 

end=$(date +%s)
