#!/bin/bash
./covid_ac_fix_solve solve \
    --name arenas_i_no_under_rep \
    --desc ./desc_files/arenas.desc \
    --content ./data/spain.dat \
    ./models/arenas/data.jld \
    ./models/arenas/run.jl \
    ./models/arenas/Manifest.toml \
    ./models/arenas/Project.toml
