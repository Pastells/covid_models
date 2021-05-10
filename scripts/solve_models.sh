#!/bin/bash

country=italy
day_min=0
day_max=50

cd desc_files
for f in sir.desc sair.desc sir_erlang.desc sair_erlang.desc net_sir.desc; do
    sed -i -e "s/\.\/data.*/\.\/data\/${country}.dat $(seq -s ' ' 42 46)/" \
        -e 's/--day_min .*/--day_min '${day_min}' \\/' \
        -e 's/--day_max .*/--day_max '${day_max}' \\/' $f
done
cd ..
exit


# SIR
./covid_ac solve --name ${country}_sir \
    --desc ./desc_files/sair.desc --content ./data/${country}.dat \
    ./models/utils/utils.py ./models/utils/config.py \
    ./models/sair.py

# SAIR
./covid_ac solve --name ${country}_sair \
    --desc ./desc_files/sair.desc --content ./data/${country}.dat \
    ./models/utils/utils.py ./models/utils/config.py \
    ./models/sair.py

# SIR Erlang
./covid_ac solve --name ${country}_sir_erlang \
    --desc ./desc_files/sir_erlang.desc --content ./data/${country}.dat \
    ./models/utils/utils.py ./models/utils/config.py \
    ./models/sir_erlang.py

# SAIR Erlang
./covid_ac solve --name ${country}_sair_erlang \
    --desc ./desc_files/sair_erlang.desc --content ./data/${country}.dat \
    ./models/utils/utils.py ./models/utils/config.py \
    ./models/sair_erlang.py

# Network SIR
./covid_ac solve --name ${country}_net_sir \
    --desc ./desc_files/net_sir.desc --content ./data/${country}.dat \
    ./models/utils/utils.py ./models/utils/config.py \
    ./models/net_sir.py ./models/event_driven/fast_sir.py ./models/utils/utils_net.py
