#!/bin/bash
: ${1?"Usage: $0 number"}
: ${2?"Usage: $0 name"}
number=$1
name=$2

sed -i "s/$number.*/$number $name/" results/ac/LIST.dat
sed -i "s/Name.*/Name: $name/" results/ac/$number.get
