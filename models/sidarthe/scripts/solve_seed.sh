#!/bin/bash

day=35
sed -i 's/--day_max .*/--day_max '${day}' \\/' \
    sidarthe_precise.desc
for seed in {42..61}; do
    sed -i 's/%%SEED .*/%%SEED '${seed}'/' sidarthe_precise.desc
    ./covid_ac solve --name sidarthe_${day}_${seed}'.500gen' \
        --desc ./sidarthe_precise.desc \
        --content ./sidarthe.py ./sidarthe.m ./empty.dat
done
