#!/bin/bash

var_beta="--beta"
var_beta="--beta"
var_delta_a="--delta_a"
var_delta_a="--delta_a"
var_alpha="--alpha"
var_n="--n"
other=""

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --beta1*)
        shift
        while [[ $1 != --* ]] && [[ $# -gt 0 ]]; do
            var_beta+=" $1"
            shift
        done
        ;;
        --beta2*)
        shift
        while [[ $1 != --* ]] && [[ $# -gt 0 ]]; do
            var_beta_a+=" $1"
            shift
        done
        ;;
        --delta1*)
        shift
        while [[ $1 != --* ]] && [[ $# -gt 0 ]]; do
            var_delta+=" $1"
            shift
        done
        ;;
        --delta2*)
        shift
        while [[ $1 != --* ]] && [[ $# -gt 0 ]]; do
            var_delta_a+=" $1"
            shift
        done
        ;;
        --epsilon*)
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

./scenario/venv/bin/python -u ./scenario/models/seir_erlang_sections.py \
    $var_beta $var_beta_a $var_delta $var_delta_a $var_alpha $var_n $other
