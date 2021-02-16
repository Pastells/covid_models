#!/bin/bash
./covid_ac_meta solve \
    --name arenas_01 \
    --desc ./desc_files/arenas.desc \
    --content ./data/spain.dat \
    ./models/arenas/src/markov_aux.jl \
    ./models/arenas/src/markov.jl \
    ./models/arenas/src/MMCAcovid19.jl \
    ./models/arenas/data.jld \
    ./models/arenas/run.jl \
    ./models/arenas/Manifest.toml \
    ./models/arenas/Project.toml
