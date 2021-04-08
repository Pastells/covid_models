#!/usr/bin/env bash

CURRENT_DIR=$(dirname $(realpath ${BASH_SOURCE[0]}))
PROJECT_DIR=$(realpath ${CURRENT_DIR}/..)

python -u ${PROJECT_DIR}/models/sird/configurable.py \
--data ${PROJECT_DIR}/data/china.dat \
--seed 42 \
--timeout 100 \
--day_min 0 \
--day_max 54 \
--mc_nseed 100 \
--D_0 5 --I_0 422 --R_0 4 --beta 0.33176312428241916 --delta 0.04121699838204111 --n 85226 --theta 0.07162823540904227
