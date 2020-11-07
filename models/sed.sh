#!/bin/bash
sed -i "s/S/s/g" $1
sed -i "s/E/e/g" $1
sed -i "s/I/i/g" $1
sed -i "s/R/r/g" $1
sed -i "s/s0/s_0/g" $1
sed -i "s/e0/e_0/g" $1
sed -i "s/i0/i_0/g" $1
sed -i "s/r0/r_0/g" $1
sed -i "s/T_steps/t_steps/g" $1
sed -i "s/N/n/g" $1

