#!/bin/bash

# save a given number of problems to result folder
# saves stdout and get output, and adds number and name to list

results="results/ac"  # results folder
list="$results/LIST.dat"
: ${1?"Usage: $0 number"}

while [[ $# -gt 0 ]]; do
    number=$1
    shift
    # check if problem already exists
    if grep -q "^$number " $list; then
        echo "$number is already in list"
        read -p 'Do you want to overwrite it? [Y/n] ' answer
        if [[ ${answer,,} = y ]]; then
            sed -i '/^$number /d' $list
        else
            echo "Problem not saved"
            continue
        fi
    fi

    ./covid_ac stdout $number > $results/$number.stdout
    # check if problem exists in covid_ac list
    if ! [[ -s $results/$number.stdout ]]; then
        echo "Stdout empty, problem $number probably non-existent."
        rm $results/$number.stdout
        continue
    fi

    ./covid_ac get $number &> $results/$number.get

    name=$(grep Name $results/$number.get | awk  '{print $2}')
    echo "$number $name" >> $list
done

sort -nuo $list $list
