(* ::Package:: *)

BeginPackage["Sird`"];

SirdEquations::usage = "ODE system for the SIRD model";
SolveSird::usage = "";
OptFunc::usage = "";
Score::usage = "";

Begin["Private`"];

(* match pattern, when SirdEquations[...] is found, replace with the equations *)
(* rates = {infection rate, recovery rate, dead rate} *)
SirdEquations[rates_List, n_, initialInfected_, initialRecovered_, initialDeaths_] := {
	Derivative[1][Succeptible][t] == -rates[[1]] Infected[t] Succeptible[t],
	Derivative[1][Infected][t] == -(rates[[3]] + rates[[2]]) Infected[t]
		+ rates[[1]] Infected[t] Succeptible[t],
	Derivative[1][Recovered][t] == rates[[2]] Infected[t],
	Derivative[1][Deaths][t] == rates[[3]] Infected[t],
	Succeptible[0] == n - initialInfected - initialRecovered - initialDeaths,
	Infected[0] == initialInfected,
	Recovered[0] == initialRecovered,
	Deaths[0] == initialDeaths
};

SolveSird[ode_, tmin_, tmax_] := NDSolve[ode, {Deaths, Infected, Recovered, Succeptible}, {t, tmin, tmax}]

OptFunc[data_, ode_] := Module[{nsol1, nsol2, nsol3, infteo, deadteo, recoteo, phi,
		nsucc, ninf, ndead, nreco,
		y1valtemp, y2valtemp, y3valtemp, y4valtemp,
		days, tmax, param, nsolall},
	{days, nsucc, ninf, ndead, nreco} = Transpose[data];
	y1valtemp = nsucc /. {0 -> 1};
	y2valtemp = ninf /. {0 -> 1};
	y3valtemp = ndead /. {0 -> 1};
	y4valtemp = nreco /. {0 -> 1};
	tmax = Max[days];

	nsolall = SolveSird[ode, 0, tmax];

	nsol1 = Infected[t] /. nsolall[[1]];
	nsol2 = Deaths[t] /. nsolall[[1]];
	nsol3 = Recovered[t] /. nsolall[[1]];

	infteo = Evaluate[nsol1] /. t->days;
	deadteo = Evaluate[nsol2] /. t->days;
	recoteo = Evaluate[nsol3] /. t->days;

	phi = Total[((infteo-ninf) / y2valtemp) ^ 2
		+ ((deadteo - ndead) / y3valtemp) ^ 2
		+ ((recoteo - nreco) / y4valtemp) ^ 2
	]
];

Score[p1_ ? NumberQ, p2_ ? NumberQ, p3_ ? NumberQ, data_, n_,
		initialInfected_, initialRecovered_, initialDeaths_] :=
	Sird`OptFunc[data,
		Sird`SirdEquations[{p1, p2, p3},
			n, initialInfected, initialRecovered, initialDeaths]
	];

End[];

EndPackage[];
