#!/bin/bash
python sir_erlang_sections.py \
--delta 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 \
--beta 0.5 0.4 0.3 0.5 0.4 0.3 0.5 0.4 0.3 \
--n 10000 15000 20000 25000 30000 1000000 20000 150000 100000 \
--section_days 0 10 20 30 40 50 60 70 80 90 \
--k_inf 1 \
--k_rec 2 \
--mc_nseed=10 \
--plot \
