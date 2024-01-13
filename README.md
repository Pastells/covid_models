# Epidemic Models

Epidemic models in python to be callibrated to describe the Covid-19 pandemic.
These models were used to test EpidemicGGA.

- Paper (open access) -
  [ePyDGGA: automatic configuration for fitting epidemic curves](https://www.nature.com/articles/s41598-023-43958-2)
- Documentation: https://ulog.udl.cat/static/doc/epidemic-gga/html/index.html

Algorithms used:

- ODEs - deterministic continuous mean-field
- Discrete - deterministic discrete mean-field
- Gillespie - Stochastic mean-field
- [Event-driven algorithm](https://link.springer.com/book/10.1007/978-3-319-50806-1)
  (fast-sir variant)
- Event-driven algorithm (with complex network). An ER or BA network can be
  used.

Stochastic models use exponential or Erlang distributed times (Erlang models).

Implemented models:

- SIR
  - ODEs (Erlang) (with sections)
  - Gillespie (Erlang) (with sections)
  - Fast (with sections)
  - Network (with sections)
- SIRD
  - Discrete ODEs
  - Gillespie
  - Gillespie parallelized version
- SAIR
  - Gillespie (Erlang) (with sections)
  - ODEs (with sections)
  - Fast (with sections)
  - Network (with sections)
- SEAIR
  - Gillespie

Additionally, the following models have been adapted from the literature:

- [SIDARTHE model](https://github.com/dungltr/sidarthe) (Matlab)
- SIDARTHE with sections (Python)
- [MMCAcovid19](https://github.com/jtmatamalas/MMCAcovid19.jl) (Julia)
- [SEIPAHRF](https://doi.org/10.1016/j.chaos.2020.109846)

For the mean-field ones: beta, beta_a, delta, delta_a, alpha and n are all
changed continuously using a tanh, for the network ones n changes abruptly.

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

- SIR
  - [x] SIR (Gillespie + ODEs)
  - [x] SIR with sections (ODEs)
  - [x] Network SIR
  - [x] Network SIR with sections
  - [x] Erlang SIR
  - [x] Erlang SIR with sections
- SIRD
  - [x] SIRD
- SAIR
  - [x] SAIR
  - [x] Network SAIR
  - [x] Network SAIR with sections
  - [x] Erlang SAIR
  - [x] Erlang SAIR with sections
- SEAIR
  - [x] SEAIR
- [x] SIDARTHE (Original in Matlab)
- [x] SIDARTHE with sections (Python)
