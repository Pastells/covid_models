#!/bin/bash

var_beta="--beta"
var_delta="--delta"
var_n="--n"
other=""

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --beta*)
    var_beta+=" $2"
    shift # past argument
    shift # past value
    ;;
    --delta*)
    var_delta+=" $2"
    shift
    shift
    ;;
    --n*)
    var_n+=" $2"
    shift
    shift
    ;;
    *)
    other+=" $1"
    shift
    while [[ $1 != --* ]] && [[ $# -gt 0 ]]
        do other+=" $1"
        shift
    done
    ;;
esac
done

./scenario/venv/bin/python -u ./scenario/src/sir_erlang_sections.py \
    $var_beta $var_delta $var_n $other
