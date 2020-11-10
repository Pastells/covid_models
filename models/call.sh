#!/bin/bash
python \
sir_erlang_sections.py \
--plot \
--nseed=10 \
--delta 0.2 0.2 0.2 \
--beta 0.5 0.4 0.3 \
--k_inf 1 \
--k_rec 2 \
--n 10000 15000 20000 \
--section_days 0 20 40 100
