#!/bin/bash

# Executed like:
# sir.py -h | ./generate_desc.sh
# not working for _sections models

params=false
while read line; do
    if [[ "$params" = true ]]; then
        if [[ $line = "" ]]; then
            break
        else
            words=( $line )
            var_type=${words[1]}
            case $var_type in
                int)
                    var_type="integer"
                ;;
                float)
                    var_type="real"
                ;;
                *)
                    var_type="categorical"
                ;;
            esac
            echo ${words[0]} $var_type ${words[-1]}
        fi
    fi
    if [[ $line == parameters:* ]]; then
        params=true
    fi
done

