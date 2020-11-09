#!/bin/bash
python \
sir_erlang_steps.py \
--plot \
--nseed=10 \
--delta 0.2 0.2 0.2 \
--beta 0.5 0.5 0.5 \
--k_inf 2 \
--k_rec 1 \
--n 10000 13000 16000 \
--section_days 0 20 40 100
