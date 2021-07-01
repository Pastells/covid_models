#!/bin/bash

var_beta="--beta"
var_delta="--delta"
var_n="--n"
other=""

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --beta*)
        shift
        while [[ $1 != --* ]] && [[ $# -gt 0 ]]; do
            var_beta+=" $1"
            shift
        done
        ;;
        --delta*)
        shift
        while [[ $1 != --* ]] && [[ $# -gt 0 ]]; do
            var_delta+=" $1"
            shift
        done
        ;;
        --n*)
        shift
        while [[ $1 != --* ]] && [[ $# -gt 0 ]]; do
            var_n+=" $1"
            shift
        done
        ;;
        *)
        other+=" $1"
        shift
        while [[ $1 != --* ]] && [[ $# -gt 0 ]]; do
            other+=" $1"
            shift
        done
        ;;
    esac
done

# To run locally uncomment and move line below (remember to add --section_days)
# python ./models/sir_erlang_sections.py \
./venv/bin/python -u ./models/sir_erlang_sections.py \
    $var_beta $var_delta $var_n $other
