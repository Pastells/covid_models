#!/bin/bash

max_cost_day=46

results="/home/pol/Documents/iiia_udl/programs/results/ac"  # results folder
grep 500 results/ac/LIST.dat > models/sidarthe/list.dat  # filter saved list
cd models/sidarthe

for day in 42; do
    echo "seed gen train_cost test_av_cost" >> costs.dat
    for seed in {42..61}; do

        number=$(grep $day'_'$seed list.dat | awk '{print $1;}')
        # grep -B 34 "Generation Summary" $results/$number.stdout > temp.dat  # all generations
        grep -B 6 "Generation Summary 1 " $results/$number.stdout > temp.dat  # first generation
        grep -B 6 -A 30 "Winner has changed" $results/$number.stdout >> temp.dat  # only the ones improving
        gen_array=($(grep "Generation Summary" temp.dat | sed 's/^.*Summary //' | awk '{print $1;}'))
        grep command temp.dat | sed 's/^.*<seed> //' > params_temp.dat
        grep objective temp.dat | sed 's/^.*objective: //' > costs_temp.dat

        line=1
        for gen in ${gen_array[@]}; do
            echo $seed $gen | tee -a results.dat
            params=$(sed -n ${line}p params_temp.dat)
            cost=$(sed -n ${line}p costs_temp.dat)

            echo -n $seed $gen $cost" " >> costs.dat
            python sidarthe.py $params --print \
                --data empty.dat \
                --min_cost_day $((day+1)) --max_cost_day $max_cost_day \
                --seed $seed --timeout 100  >> results.dat
            ((line++))
        done
    done
done

# Create column names
str=""
for ((i=$((day+1)); i<=$max_cost_day; i++)); do str="$str H$i"; done
for ((i=$((day+1)); i<=$max_cost_day; i++)); do str="$str D$i"; done
for ((i=$((day+1)); i<=$max_cost_day; i++)); do str="$str R$i"; done
for ((i=$((day+1)); i<=$max_cost_day; i++)); do str="$str T$i"; done
echo "seed gen $str" > results.csv
# Merge every 5 lines
paste -d " "  - - - - -  < results.dat >> results.csv

rm list.dat temp.dat costs_temp.dat params_temp.dat results.dat
cd ../..
