#!/bin/bash

day=50
sed -i 's/--day_max .*/--day_max '${day}' \\/' \
    desc_files/sird_china.desc
for seed in {45..53}; do
    sed -i 's/%%SEED .*/%%SEED '${seed}'/' desc_files/sird_china.desc
    ./covid_ac solve --name sird_china_${day}_${seed}'.100gen' \
        --desc ./desc_files/sird_china.desc \
        --content ./models/sird.py \
        ./data/china.dat \
        ./models/utils/utils.py ./models/utils/config.py
done
