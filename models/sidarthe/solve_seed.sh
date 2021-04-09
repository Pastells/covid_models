#!/bin/bash

for day in 42; do
    sed -i 's/--max_cost_day .*/--max_cost_day '${day}' \\/' \
        sidarthe_precise.desc
    for seed in {42..51}; do
        sed -i 's/%%SEED .*/%%SEED '${seed}'/' sidarthe_precise.desc
        ./covid_ac solve --name sidarthe_${day}_${seed}'.60gen' \
            --desc ./sidarthe_precise.desc \
            --content ./sidarthe.py ./sidarthe.m ./empty.dat
    done
done
