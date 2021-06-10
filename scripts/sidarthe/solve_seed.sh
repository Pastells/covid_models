#!/bin/bash

day=46
sed -i 's/--day_max .*/--day_max '${day}' \\/' \
    sidarthe_precise.desc
for seed in {42..51}; do
    sed -i 's/%%SEED .*/%%SEED '${seed}'/' sidarthe_precise.desc
    ./covid_ac solve --name sidarthe_${day}_${seed}'.500gen' \
        --desc ./desc_files/sidarthe.desc \
        --content ./sidarthe.py ./sidarthe.m ./empty.dat
done
