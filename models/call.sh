#!/bin/bash
python sir_erlang_sections.py \
--delta 0.2 0.2 \
--beta 0.5 2.5 \
--n 10000 15000 \
--section_days 0 10 20 \
--k_inf 1 \
--k_rec 2 \
--mc_nseed=3 \
