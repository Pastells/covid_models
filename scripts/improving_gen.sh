#!/bin/bash

# prints which generations improve and their cost

number=$1
results="results/ac"  # results folder
: ${1?"Usage: $0 number"}

# save output
# ./covid_ac stdout $number > $results/$number.stdout  # save output
./scripts/save_problem.sh $number

grep -B 6 "Generation Summary 1 " $results/$number.stdout > .temp.dat  # first generation
grep -B 6 -A 30 "Winner has changed" $results/$number.stdout >> .temp.dat  # only the ones improving
gen_array=($(grep "Generation Summary" .temp.dat | sed 's/^.*Summary //' | awk '{print $1;}'))
cost_array=($(grep objective .temp.dat | sed 's/^.*objective: //'))

echo "gen cost"
for ((i=1; i<=${#gen_array[@]}; i++)); do
    echo "${gen_array[i]} ${cost_array[i]}"
done
rm .temp.dat
