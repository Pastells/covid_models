#!/bin/bash
: ${1?"Usage: $0 number"}
number=$1
list="results/ac/LIST.dat"

# check if problem already exists
if grep -q "^$number " $list; then
    echo "$number is already in list"
    read -p 'Do you want to overwrite it? [Y/n] ' answer
    if [ ${answer,,} = y ]; then
        sed -i '/^$number /d' $list
    else
        echo "Problem not saved"
        exit
    fi
fi

./covid_ac stdout $number > results/ac/$number.stdout
./covid_ac get $number &> results/ac/$number.get

name=$(grep Name results/ac/$number.get | awk  '{print $2}')
echo $number $name >> $list
sort -no $list $list
