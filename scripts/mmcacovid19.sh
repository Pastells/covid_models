#!/bin/bash
./covid_ac_fix_solve solve \
    --name arenas_i_no_under_rep \
    --desc ./desc_files/mmcacovid19.desc \
    --content ./data/spain.dat \
    ./models/mmcacovid19/data.jld \
    ./models/mmcacovid19/run.jl \
    ./models/mmcacovid19/Manifest.toml \
    ./models/mmcacovid19/Project.toml
