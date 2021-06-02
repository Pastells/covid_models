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

### How to run the models

To list all the available models, use:

```shell script
python -m models --help
```

To execute a model, use:

```shell script
python -m models <model> [model args]
```

To see the arguments for a specific model (p.e. sird), use:

```shell script
python -m models sird --help
```

### Auto-configurable models:

The following models are ready to be used with Optilog:

- SAIR
  - [x] SAIR
  - [ ] Network SAIR
  - [ ] Network SAIR with sections
  - [ ] Erlang SAIR
  - [ ] Erlang SAIR with sections
- SEAIR
  - [x] SEAIR
- SIDARTHE
  - [ ] Sidarthe 1 (todo: more descriptive name?)
  - [ ] Sidarthe 2 (todo: more descriptive name?)
  - [ ] Sidarthe comp (todo: more descriptive name?)
- MMCAcovid19 (todo)
- SIR (TODO: fix net-sir-sections [when running 2 sections in cost])
  - [x] SIR
  - [x] Network SIR
  - [x] Network SIR with sections
  - [x] Erlang SIR
  - [x] Erlang SIR with sections
- SIRD (TODO: remove old sird.py)
  - [x] SIRD
