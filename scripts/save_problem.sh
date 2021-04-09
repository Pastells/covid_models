#!/bin/bash
: ${1?"Usage: $0 number"}
number=$1

./covid_ac stdout $number > results/ac/$number.stdout
./covid_ac get $number &> results/ac/$number.get

name=$(grep Name results/ac/$number.get | awk  '{print $2}')
echo $number $name >> results/ac/LIST.dat
