#!/bin/bash
# List problems to get the result from the ones with name pattern
# day_seed
#  - day  : final day where cost was computed to optimize
#  - seed : automatic configurator seed

max_cost_day=46
./covid_ac list --start-from 1114 > list.dat

for day in 42; do
    echo '# min_day' $((day+1)) 'max_day' $max_cost_day >> costs.dat
    for seed in {42..51}; do
        echo '# day' $day 'seed' $seed | tee -a results.dat costs.dat params.dat

        number=$(grep $day'_'$seed list.dat | awk '{print $1;}')
        # number=$((1072+seed-42))
        ./covid_ac stdout $number | tail > temp.dat
        params=$(tail temp.dat | grep bin | sed 's/^[^-]*//')
        cost=$(tail temp.dat | grep objective | sed 's/^[^0-9]*//')

        echo '# ac cost =' $cost >> costs.dat
        echo $params >> params.dat
        python sidarthe.py $params --print \
            --data empty.dat --min_cost_day $((day+1)) --max_cost_day $max_cost_day \
            --seed $seed --timeout 100  >> results.dat

    done
    python mean_cost.py --min_cost_day $((day)) --max_cost_day $max_cost_day
    sed -i '/# [cnrt]/d;/^$/d' costs.dat
    sed -n 4p costs.dat
    mkdir day_$day
    mv results.dat costs.dat params.dat day_$day

done

rm list.dat temp.dat
