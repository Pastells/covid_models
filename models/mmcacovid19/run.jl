## -----------------------------------------------------------------------------
## Parsing
## -----------------------------------------------------------------------------
using ArgParse
# more imports below to have faster --help

function parsing()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--beta_i"
            arg_type = Float64
            default = 0.075
            help = "infectivity of symptomatic"
        "--eta"
            arg_type = Float64
            default = 2.444
            help = "exposed days"
        "--alpha_y"
            arg_type = Float64
            default = 5.671
            help = "asymptomatic days (young)"
        "--alpha_mo"
            arg_type = Float64
            default = 2.756
            help = "asymptomatic days (mid and old)"
        "--mu_mo"
            arg_type = Float64
            default = 3.915
            help ="infectious days (mid and old)"
        "--delta"
            arg_type = Float64
            default = 0.207
            help = "social distancing"
        "--phi"
            arg_type = Float64
            default = 0.174
            help= "household permeability"
        "--mobility_data"
            arg_type = String
            default = "data.jld"
            help = "file with julia variables from ine data"
        "--data"
            arg_type = String
            default = "/home/pol/Documents/iiia_udl/programs/data/spain.dat"
            help = "file with time series"
        "--day_min"
            arg_type = Int
            default = 36
            help="first day to consider of the data series"
        "--day_max"
            arg_type = Int
            default = 116
            help="last day to consider of the data series"
        "--save"
            action = :store_true
            help = "specify to save data"
        "--seed"
            arg_type = Int
            default = 1
            help="seed for the automatic configuration"
        "--timeout"
            arg_type = Int
            default = 1200
            help="timeout for the automatic configuration"
    end

    return parse_args(s)
end

## -------------------------------------------------

try
    global args
    args = parsing()
    args["mobility_data"] = normpath(joinpath(@__FILE__,"..",args["mobility_data"]))
catch e
    println("Error parsing arguments:\n$e")
    println("GGA CRASHED ", 1e20)
    rethrow()
end

using JLD, CSV, DataFrames, Printf, MMCAcovid19
import Random
Random.seed!(1)

### ----------------------------------------------------------------------------
### COST FUNCTIONS
### ----------------------------------------------------------------------------

"""
    cost_function(epi_params::Epidemic_Params,
                      population::Population_Params,
                      compartment::Char
                      data::DataFrame)

Compute the cost function comparing to real data

# Arguments

- `epi_params::Epidemic_Params`: Structure that contains all epidemic parameters
  and the epidemic spreading information.
- `population::Population_Params`: Structure that contains all the parameters
  related with the population.
- `compartment::String`: String indicating the compartment, one of: `"I"`, `"D"`, `"IRD"`
- `data::DataFrame`: Contains real data
"""
function cost_function(epi_params::Epidemic_Params,
        population::Population_Params,
        compartment::String,
        data)

    M = population.M
    G = population.G
    T = epi_params.T

    # Init. dataframe
    df = DataFrame()
    df.strata = repeat(1:G, outer = T * M)
    df.patch = repeat(1:M, inner = G, outer = T)
    df.time = repeat(1:T, inner = G * M)

    # Store number of cases

    df.S = reshape(epi_params.ρˢᵍ .* population.nᵢᵍ, G * M * T)
    df.E = reshape(epi_params.ρᴱᵍ .* population.nᵢᵍ, G * M * T)
    df.A = reshape(epi_params.ρᴬᵍ .* population.nᵢᵍ, G * M * T)
    df.I = reshape(epi_params.ρᴵᵍ .* population.nᵢᵍ, G * M * T)
    df.PD = reshape(epi_params.ρᴾᴰᵍ .* population.nᵢᵍ, G * M * T)
    df.PH = reshape(epi_params.ρᴾᴴᵍ .* population.nᵢᵍ, G * M * T)
    df.HR = reshape(epi_params.ρᴴᴿᵍ .* population.nᵢᵍ, G * M * T)
    df.HD = reshape(epi_params.ρᴴᴰᵍ .* population.nᵢᵍ, G * M * T)
    df.R = reshape(epi_params.ρᴿᵍ .* population.nᵢᵍ, G * M * T)
    df.D = reshape(epi_params.ρᴰᵍ .* population.nᵢᵍ, G * M * T)

    # CSV.write("output/output_pre.csv", df)
    # Group by day (sum of all patches and strata)
    select!(df, Not([:strata, :patch]))
    gd = groupby(df, :time)
    list = names(df)
    filter!(e->e≠"time",list)
    df = combine(gd, list .=> sum)

    # Dismiss initial transient,
    # start at first day with more than 100 total infected
    index = findfirst(df.I_sum.>1)
    # index = findfirst(df.I_sum.>100)
    # if non exist don't split data
    index == nothing ? index = 1 : nothing
    println("Index = ", index)

    df = df[setdiff(index:end), :]
    select!(df, Not([:time]))


    # Add data:
    # Scale infected for underreporting
    underreporting = 89.4
    df.data_inf = data."#infected"[1 : end - index + 1] * 100/(100-underreporting)
    df.data_rec = data.recovered[1 : end - index + 1]
    df.data_dead = data.dead[1 : end - index + 1]
    # date after time column
    insertcols!(df, 1, :date => data.date[1 : end - index + 1])


    # Turn recovered and dead from cumulative to daily
    for row in length(df.data_dead):-1:2
        df.data_rec[row] -= df.data_rec[row-1]
        df.data_dead[row] -= df.data_dead[row-1]
    end


    # Compute different costs per day

    df.cost_I = (df.I_sum-df.data_inf).^2
    df.cost_R = (df.R_sum-df.data_rec).^2
    df.cost_D = (df.D_sum-df.data_dead).^2
    df.cost_IRD = (df.I_sum-df.data_inf).^2 + (df.R_sum-df.data_rec).^2 +
    (df.D_sum-df.data_dead).^2

    # Add all days or take maximum for chosen cost
    if compartment == "I"
        cost = sum(df.cost_I)
        # cost = maximum(df.cost_I)
    elseif compartment == "R"
        cost = sum(df.cost_R)
    elseif compartment == "D"
        cost = sum(df.cost_D)
    elseif compartment == "IRD"
        cost = sum(df.cost_IRD)
    else
        error("compartment option in cost_function not correct")
    end


    @printf("GGA SUCCESS %.2f\n", cost/1e6)
    out_file = normpath(joinpath(@__FILE__,"..", "output/output.csv"))
    CSV.write(out_file, df)
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
    nᵢᵍ = load(args["mobility_data"], "n")
    edgelist = load(args["mobility_data"], "edgelist")
    Rᵢⱼ = load(args["mobility_data"], "R")
    sᵢ = load(args["mobility_data"], "s")
    M = length(sᵢ)
catch e
    println("Error reading data.jld:\n$e")
    println("GGA CRASHED ", 1e20)
    rethrow()
end

# contacts matrix
C = [0.5980 0.3849 0.0171
     0.2440 0.7210 0.0350
     0.1919 0.5705 0.2376]

# average total number of contacts
kᵍ = [11.8, 13.3, 6.6]
# average number of contacts at home
kᵍ_h = [3.15, 3.17, 3.28]
# average number of contacts at work
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
I₀ = zeros(G, M)

# Random seeds
mov_areas = rand(1:M, 50)
I₀[2, mov_areas] .= 1.0
A₀[2, mov_areas] .= 5.0

# Madrid
A₀[2, 1515] = 50.0
I₀[2, 1515] = 10.0
A₀[3, 1516] = 50.0

# Barcelona
A₀[2, 445] = 50.0
I₀[2, 445] = 10.0
A₀[3, 446] = 50.0


set_initial_infected!(epi_params, population, E₀, A₀, I₀)

## -----------------------------------------------------------------------------
## Epidemic spreading with multiple containments
## -----------------------------------------------------------------------------

# application of containment days
# starting at day 36
tᶜs = [17, 33, 47, 66, 74] + 0*ones(Int, 5)


# mobility reduction from INE
κ₀s = [0.31, 0.25, 0.32, 0.36, 0.45]

ϕs = ones(5) * args["phi"]

# social distancing
δs = ones(5) * args["delta"]


## -----------------------------------------------------------------------------
## Run model
## -----------------------------------------------------------------------------
try
    # add verbose = true for prints during execution
    run_epidemic_spreading_mmca!(epi_params, population, tᶜs, κ₀s, ϕs, δs)
catch e
    println("Error while running the model:\n$e")
    println("GGA CRASHED ", 1e20)
    rethrow()
end


## -----------------------------------------------------------------------------
## Cost and save
## -----------------------------------------------------------------------------

# Cost function
try
    data = CSV.read(args["data"], DataFrame)
    data = data[setdiff(args["day_min"]:args["day_max"]), :]
    cost_function(epi_params, population, "I", data)
catch e
    println("Error reading data file or computing cost:\n$e")
    println("GGA CRASHED ", 1e20)
    rethrow()
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
