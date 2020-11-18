#!/bin/bash
sed -i "s/ s / S /g" $1
sed -i "s/ s,/ S,/g" $1
sed -i "s/ s_/ S_/g" $1
sed -i "s/ s0/ S0/g" $1
sed -i "s/ s\[/ S\[/g" $1

sed -i "s/ e / E /g" $1
sed -i "s/ e,/ E,/g" $1
sed -i "s/ e_/ E_/g" $1
sed -i "s/ e0/ E0/g" $1
sed -i "s/ e\[/ E\[/g" $1

sed -i "s/ i / I /g" $1
sed -i "s/ i,/ I,/g" $1
sed -i "s/ i_/ I_/g" $1
sed -i "s/(i_/(I_/g" $1
sed -i "s/ i0/ I0/g" $1
sed -i "s/ i\[/ I\[/g" $1

sed -i "s/ r / R /g" $1
sed -i "s/ r,/ R,/g" $1
sed -i "s/ r_/ R_/g" $1
sed -i "s/ r0/ R0/g" $1
sed -i "s/ r\[/ R\[/g" $1

sed -i "s/ N,/ n,/g" $1
sed -i "s/ N / n /g" $1

