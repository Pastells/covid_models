using ArgParse, JLD, CSV, DataFrames
push!(LOAD_PATH, "./src/")
using MMCAcovid19



## -----------------------------------------------------------------------------
## Parsing
## -----------------------------------------------------------------------------

function parsing()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--beta_i"
            arg_type = Real
            default = 0.075
            help = "infectivity of symptomatic"
        "--eta"
            arg_type = Real
            default = 2.444
            help = "exposed days"
        "--alpha_y"
            arg_type = Real
            default = 5.671
            help = "asymptomatic days (young)"
        "--alpha_mo"
            arg_type = Real
            default = 2.756
            help = "asymptomatic days (mid and old)"
        "--mu_mo"
            arg_type = Real
            default = 3.915
            help ="infectious days (mid and old)"
        "--phi"
            arg_type = Real
            default = 0.174
            help = "household permeability"
        "--delta"
            arg_type = Real
            default = 0.207
            help = "social distancing"
        "--data"
            arg_type = String
            default = "/home/pol/Documents/iiia_udl/programs/data/spain.dat"
            help = "file with time series"
        "--day_min"
            arg_type = Int
            default = 1
            help="first day to consider of the data series"
        "--day_max"
            arg_type = Int
            default = 50
            help="last day to consider of the data series"
        "--save"
            action = :store_true
            help = "specify to save data"
    end

    return parse_args(s)
end

try
    global args
    args = parsing()
catch e
    println("Error parsing arguments:\n$e")
    println("GGA CRASHED ", 1e20)
    exit()
end

## -----------------------------------------------------------------------------
## Population parameters
## -----------------------------------------------------------------------------

# number of strata
G = 3

## .............................................................................
## from INE
## .............................................................................

# Load inputs generated with data.jl
try
    global nᵢᵍ, edgelist, Rᵢⱼ,sᵢ, M
    nᵢᵍ = load("data.jld", "n")
    edgelist = load("data.jld", "edgelist")
    Rᵢⱼ = load("data.jld", "R")
    sᵢ = load("data.jld", "s")
    M = length(sᵢ)
catch e
    println("Error reading data.jld:\n$e")
    println("GGA CRASHED ", 1e20)
    exit()
end

# number of regions (municipalities)

#=
M = 5
# populations for region and stratta
nᵢᵍ = [ 4995.0   9875.0  14970.0   30010.0   40326.0
       30107.0  59630.0  90009.0  179745.0  239983.0
       15145.0  29827.0  45086.0   90266.0  120026.0]

# region surface
sᵢ = [10.6, 23.0, 26.6, 5.7, 61.6]

edgelist = [1  1; 1  2; 1  3; 1  5; 2  1; 2  2; 2  3; 2  4;
            2  5; 3  1; 3  2; 3  3; 3  4; 3  5; 4  1; 4  3;
            4  4; 4  5; 5  1; 5  2; 5  3; 5  5]

# Mobility matrix
#   assumption Rᵢⱼ = δᵢⱼ for Y and O
Rᵢⱼ = [0.3288; 0.0905; 0.0995; 0.4812; 0.3916; 0.2213; 0.1052; 0.2775;
       0.0044; 0.0233; 0.5205; 0.0117; 0.0807; 0.3638; 0.5156; 0.0579;
       0.0218; 0.4047; 0.3081; 0.2862; 0.0621; 0.3436]

println(typeof(nᵢᵍ))
println(typeof(Rᵢⱼ))
println(typeof(edgelist))
println(typeof(sᵢ))
println(size(nᵢᵍ))
println(size(Rᵢⱼ))
println(size(edgelist))
println(size(sᵢ))
=#

## .............................................................................

# contacts matrix
C = [0.5980 0.3849 0.0171
     0.2440 0.7210 0.0350
     0.1919 0.5705 0.2376]

# average total number of contacts
kᵍ = [11.8, 13.3, 6.6]
# average number of contacts at home
kᵍ_h = [3.15, 3.17, 3.28]
# average number of contacts at wor
kᵍ_w = [1.72, 5.18, 0.0]

# mobility factor
# pᵍ = [0.0, 1.0, 0.05]
pᵍ = [1.0, 1.0, 1.0]

# density factor [km²]
ξ = 0.01
# average household size
σ = 2.5

population = Population_Params(G, M, nᵢᵍ, kᵍ, kᵍ_h, kᵍ_w, C, pᵍ, edgelist, Rᵢⱼ, sᵢ, ξ, σ)

## -----------------------------------------------------------------------------
## Epidemic parameters
## -----------------------------------------------------------------------------

βᴵ = args["beta_i"]
βᴬ = 0.5 * βᴵ  # assumed
ηᵍ = ones(3) / args["eta"]
αᵍ = [1/args["alpha_y"], 1/args["alpha_mo"], 1/args["alpha_mo"]]
μᵍ = [1, 1/args["mu_mo"], 1/args["mu_mo"]]  # Y assumed

# direct death probability
θᵍ = [0.0, 0.008, 0.047]
# ICU probability
γᵍ = [0.0003, 0.003, 0.026]
# predeceased rate
ζᵍ = ones(3) / 7.084
# prehospitalized in ICU rate
λᵍ = ones(3) / 4.084
# fatality probability in ICU
ωᵍ = ones(3) * 0.3
# death rate in ICU
ψᵍ = ones(3) / 7.0
# ICU discharge rate
χᵍ = ones(3) / 21.0

# days to compute
T = args["day_max"]-args["day_min"] + 1

epi_params = Epidemic_Params(βᴵ, βᴬ, ηᵍ, αᵍ, μᵍ, θᵍ, γᵍ,
                             ζᵍ, λᵍ, ωᵍ, ψᵍ, χᵍ, G, M, T)

## -----------------------------------------------------------------------------
## Initialization of the epidemics
## -----------------------------------------------------------------------------

E₀ = zeros(G, M)

A₀ = zeros(G, M)
A₀[2, 5] = 2.0
A₀[3, 3] = 1.0

I₀ = zeros(G, M)
I₀[2, 5] = 1.0

set_initial_infected!(epi_params, population, E₀, A₀, I₀)

## -----------------------------------------------------------------------------
## Epidemic spreading with multiple containments
## -----------------------------------------------------------------------------

# application of containment days
tᶜs = [30, 60, 90, 120]

# mobility reduction from INE
κ₀s = [0.65, 0.75, 0.65, 0.55]

ϕs = [0.174, 0.174, 0.174, 0.174]

# social distancing
δs = [0.207, 0.207, 0.207, 0.207]


## -----------------------------------------------------------------------------
## Run model
## -----------------------------------------------------------------------------
try
    # add verbose = true for prints during execution
    run_epidemic_spreading_mmca!(epi_params, population, tᶜs, κ₀s, ϕs, δs)
catch e
    println("Error while running the model:\n$e")
    println("GGA CRASHED ", 1e20)
    exit()
end


## -----------------------------------------------------------------------------
## Cost and save
## -----------------------------------------------------------------------------

# Cost function
try
    data = CSV.read(args["data"], DataFrame)
    data = data[setdiff(args["day_min"]:args["day_max"]), :]
    cost_function(epi_params, population, "IRD", data)
catch e
    println("Error reading data file:\n$e")
    println("GGA CRASHED ", 1e20)
    exit()
end

# save results
if args["save"]
    # Output path and suffix for results files
    output_path = "/home/pol/Documents/iiia_udl/programs/models/arenas/output"
    suffix = "run01"

    # Store compartments
    store_compartment(epi_params, population, "S", suffix, output_path)
    store_compartment(epi_params, population, "E", suffix, output_path)
    store_compartment(epi_params, population, "A", suffix, output_path)
    store_compartment(epi_params, population, "I", suffix, output_path)
    store_compartment(epi_params, population, "PH", suffix, output_path)
    store_compartment(epi_params, population, "PD", suffix, output_path)
    store_compartment(epi_params, population, "HR", suffix, output_path)
    store_compartment(epi_params, population, "HD", suffix, output_path)
    store_compartment(epi_params, population, "R", suffix, output_path)
    store_compartment(epi_params, population, "D", suffix, output_path)
end


# Optional kernel length
# τ = 21

# Calculate effective reproduction number R
# Rᵢᵍ_eff, R_eff = compute_R_eff(epi_params, population, τ)

# Calculate and store effective reproduction number R
# store_R_eff(epi_params, population, suffix, output_path; τ=τ)
