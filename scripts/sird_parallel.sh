#!/bin/bash

: ${1?"Usage: $0 number n_seeds"}
: ${2?"Usage: $0 $1 n_seeds"}

# china
case $1 in
    1)
    echo "china"
    python -m models sird-parallel --plot --mc_nseed $2 \
        --data data/china.dat --day_min 0 --day_max 54 \
        --initial_infected 430 --initial_recovered 10 --initial_dead 15 \
        --n 83300 --beta 0.329035 --delta 0.0384 --theta 0.08072917 \
        --plot --save china
    ;;
# china*
    2)
    echo "china*"
    python -m models sird-parallel --plot --mc_nseed $2 \
        --data data/china.dat --day_min 0 --day_max 28 \
        --initial_infected 999 --initial_recovered 10 --initial_dead 17 \
        --n 79200 --beta 0.263736 --delta 0.021 --theta 0.142857 \
        --plot --save china_star
    ;;
# italy
    3)
    echo "italy"
    python -m models sird-parallel --plot --mc_nseed $2 \
        --data data/italy.dat --day_min 20 --day_max 54 \
        --n 41300 --beta 0.32627 --delta 0.0376 --theta 0.4335106 \
        --plot --save italy
    echo $@
    ;;
esac
