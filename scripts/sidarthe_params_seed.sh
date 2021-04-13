#!/bin/bash

max_cost_day=46

results="/home/pol/Documents/iiia_udl/programs/results/ac"  # results folder
grep 200 results/ac/LIST.dat > models/sidarthe/list.dat  # filter saved list
cd models/sidarthe

for day in 42; do
    echo "seed gen train_cost test_cost" >> costs.dat
    for seed in {42..51}; do

        number=$(grep $day'_'$seed list.dat | awk '{print $1;}')
        # grep -B 34 "Generation Summary" $results/$number.stdout > temp.dat  # all generations
        grep -B 6 "Generation Summary 1 " $results/$number.stdout > temp.dat  # first generation
        grep -B 6 -A 30 "Winner has changed" $results/$number.stdout >> temp.dat  # only the ones improving
        gen_array=($(grep "Generation Summary" temp.dat | sed 's/^.*Summary //' | awk '{print $1;}'))
        grep command temp.dat | sed 's/^.*<seed> //' > params.dat
        grep objective temp.dat | sed 's/^.*objective: //' > costs_temp.dat

        line=1
        for gen in ${gen_array[@]}; do
            echo '# day' $day 'seed' $seed 'gen' $gen | tee -a results.dat
            params=$(sed -n ${line}p params.dat)
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
rm list.dat temp.dat costs_temp.dat
cd ../..
