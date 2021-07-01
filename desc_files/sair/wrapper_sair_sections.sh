#!/bin/bash

var_beta="--beta"
var_beta_a="--beta_a"
var_delta="--delta"
var_delta_a="--delta_a"
var_alpha="--alpha"
var_n="--n"
other=""

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --beta_a*)
        shift
        while [[ $1 != --* ]] && [[ $# -gt 0 ]]; do
            var_beta_a+=" $1"
            shift
        done
        ;;
        --beta*)
        shift
        while [[ $1 != --* ]] && [[ $# -gt 0 ]]; do
            var_beta+=" $1"
            shift
        done
        ;;
        --delta_a*)
        shift
        while [[ $1 != --* ]] && [[ $# -gt 0 ]]; do
            var_delta_a+=" $1"
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
        --alpha*)
        shift
        while [[ $1 != --* ]] && [[ $# -gt 0 ]]; do
            var_alpha+=" $1"
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
# python ./models/sair_erlang_sections.py \
./venv/bin/python -u ./models/sair_erlang_sections.py \
    $var_beta $var_beta_a $var_delta $var_delta_a $var_alpha $var_n $other
