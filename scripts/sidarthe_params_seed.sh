#!/bin/bash

day=35
day_max=38
prefix="sidarthe"
n_gen=500
results="/home/pol/Documents/iiia_udl/programs/results/ac"  # results folder

if [[ -f "models/sidarthe/costs.dat" ]]; then
    echo "costs.dat exists"
    exit
fi

grep -E "$prefix.*$n_gen" results/ac/LIST.dat > models/sidarthe/.list.dat  # filter saved list
cd models/sidarthe || exit

echo "seed gen train_cost test_av_cost" > costs.dat
truncate -s 0 .results.dat  # create or empty
for seed in {42..43}; do

    number=$(grep $day'_'$seed .list.dat | awk '{print $1;}')
    # grep -B 34 "Generation Summary" $results/$number.stdout > .temp.dat  # all generations
    grep -B 6 "Generation Summary 1 " "$results/$number.stdout" > .temp.dat  # first generation
    grep -B 6 -A 30 "Winner has changed" "$results/$number.stdout" >> .temp.dat  # only the ones improving
    gen_array=($(grep "Generation Summary" .temp.dat | sed 's/^.*Summary //' | awk '{print $1;}'))
    grep command .temp.dat | sed 's/^.*<seed> //' > .params_temp.dat
    grep objective .temp.dat | sed 's/^.*objective: //' > .costs_temp.dat

    line=1
    for gen in "${gen_array[@]}"; do
        echo "$seed $gen" | tee -a .results.dat
        params=$(sed -n ${line}p .params_temp.dat)
        cost=$(sed -n ${line}p .costs_temp.dat)

        echo -n "$seed $gen $cost " >> costs.dat
        # program writes test cost and adds newline character
        python sidarthe.py "$params" --print \
            --data empty.dat \
            --day_min $((day+1)) --day_max $day_max \
            --seed $seed --timeout 100  >> .results.dat
        ((line++))
    done
done

# Create column names
str=""
for ((i=1; i<=$day_max; i++)); do str="$str H$i"; done
for ((i=1; i<=$day_max; i++)); do str="$str D$i"; done
for ((i=1; i<=$day_max; i++)); do str="$str R$i"; done
for ((i=1; i<=$day_max; i++)); do str="$str T$i"; done
echo "seed gen $str" > results.csv

# Deal with multiple line output from octave
sed -i -e '/Column/d' -e '/^$/d' .results.dat

# Merge lines
awk '/^[0-9]/{if (NR!=1)print "";}{printf $0}END{print "";}' .results.dat >> results.csv

rm .list.dat .temp.dat .costs_temp.dat .params_temp.dat .results.dat
cd ../..
