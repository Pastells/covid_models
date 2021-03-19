Epidemic models in python to be callibrated to describe the Covid-19 pandemic.

Current models:

- Stochastic mean-field SIR (with exponential or Erlang distributed times)
- Stochastic mean-field SIRD (including a parallel version)
- Stochastic mean-field SEIR (with exponential or Erlang distributed times)
- Stochastic SIR/SEIR with a network of choice (ER or BA) as a population

All have a version with sections also available.

For the mean-field ones: beta, beta_a, delta, delta_a, alpha and n are all
changed continuously using a tanh for the network ones n changes abruptly for
now

Additionally, the following models have been adapted from the literature:

- SIDARTHE model (https://github.com/dungltr/sidarthe)
- MMCAcovid19 (https://github.com/jtmatamalas/MMCAcovid19.jl)
