#!/bin/bash

var_beta1="--beta1"
var_beta2="--beta2"
var_delta1="--delta1"
var_delta2="--delta2"
var_epsilon="--epsilon"
var_n="--n"
other=""

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --beta1*)
        shift
        while [[ $1 != --* ]] && [[ $# -gt 0 ]]; do
            var_beta1+=" $1"
            shift
        done
        ;;
        --beta2*)
        shift
        while [[ $1 != --* ]] && [[ $# -gt 0 ]]; do
            var_beta2+=" $1"
            shift
        done
        ;;
        --delta1*)
        shift
        while [[ $1 != --* ]] && [[ $# -gt 0 ]]; do
            var_delta1+=" $1"
            shift
        done
        ;;
        --delta2*)
        shift
        while [[ $1 != --* ]] && [[ $# -gt 0 ]]; do
            var_delta2+=" $1"
            shift
        done
        ;;
        --epsilon*)
        shift
        while [[ $1 != --* ]] && [[ $# -gt 0 ]]; do
            var_epsilon+=" $1"
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

echo ""
echo ""
python seir_erlang_sections.py \
    $var_beta1 $var_beta2 $var_delta1 $var_delta2 $var_epsilon $var_n $other
