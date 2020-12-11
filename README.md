Epidemic models in python to be callibrated to describe the Covid-19 pandemic.

Current models:

- Stochastic mean-field SIR (with exponential or Erlang distributed times)
- Stochastic mean-field SEIR (with exponential or Erlang distributed times)
- Stochastic SIR/SEIR with a network of choice (ER or BA) as a population
- All have a version with sections also available:
    for the mean-field ones: beta1/2, delta1/2, epsilon and n are all changed continuously using a tanh
    for the network ones n changes abruptly for now
