#!/bin/bash
# List problems to get the result from the ones with name pattern
# day_seed
#  - day  : final day where cost was computed to optimize
#  - seed : automatic configurator seed

day_max=46
./covid_ac list --start-from 1134 --count 10 > list.dat

for day in 42; do
    echo '# min_day' $((day+1)) 'max_day' $day_max >> costs.dat
    for seed in {42..51}; do
        echo '# day' $day 'seed' $seed | tee -a results.dat costs.dat params.dat

        number=$(grep $day'_'$seed list.dat | awk '{print $1;}')
        # number=$((1072+seed-42))
        ./covid_ac stdout $number | tail > temp.dat
        params=$(tail temp.dat | grep command | sed 's/^[^-]*//')
        cost=$(tail temp.dat | grep objective | sed 's/^[^0-9]*//')

        echo '# ac cost =' $cost >> costs.dat
        echo $params >> params.dat
        python sidarthe.py $params --print \
            --data empty.dat --day_min $((day+1)) --day_max $day_max \
            --seed $seed --timeout 100  >> results.dat

    done
    python mean_cost.py --day_min $((day)) --day_max $day_max 
    sed -i '/# [cnrt]/d;/^$/d' costs.dat
    sed -n 4p costs.dat
    mkdir day_$day
    mv results.dat costs.dat params.dat day_$day

done

rm list.dat temp.dat
