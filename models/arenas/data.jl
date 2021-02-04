"""
Create nᵢᵍ, edgelist, Rᵢⱼ and sᵢ
"""

using CSV, DataFrames, PyCall
@pyimport shapefile


# --------------------------------
# Population for mobility area
# --------------------------------
# Removed million . for total beforehand
# Changes 1rst column name to Areas and 2nd to Tipo

df1 = CSV.read("input/poblacion_areas_movilidad.csv", DataFrame)
select!(df1, Not([:Tipo, :Periodo]))

# Here I turn into integer
transform!(df1, :Total => ByRow(x -> trunc(Int,1000*x)) => :Total)

# Remove first row (total)
df1 = df1[setdiff(1:end, 1), :]

# Add numbering row
insert!(df1, 3, 1, :num)
for i in 1:length(df1[1])
    df1[i, :].num = i
end


# --------------------------------
# Mobility between areas
# --------------------------------

# Turned into utf-8 and removed spaces before ; beforehand
df2 = CSV.read("input/flujos0.csv", DataFrame)
select!(df2, Not([:CELDA_ORIGEN, :CELDA_DESTINO, :Column6]))

# Left join
df2 = leftjoin(df2, df1, on = :NOMBRE_CELDA_ORIGEN => :Areas)
rename!(df2, Dict(:num => :orig))
df2 = leftjoin(df2, select(df1, Not(:Total)), on = :NOMBRE_CELDA_DESTINO => :Areas)
rename!(df2, Dict(:num => :dest))

# Remove names
select!(df2, Not([:NOMBRE_CELDA_ORIGEN, :NOMBRE_CELDA_DESTINO]))

# Remove mobility between areas not in df1
df2 = dropmissing(df2)

# Turn flux into float
df2[:FLUJO] = Float64.(df2[:FLUJO])


# --------------------------------
# Generate needed arrays
# --------------------------------

orig = df1.orig[1]
sum = 0
Rᵢⱼ = Float64[]
edgelist = []
for i in 1:length(df2[1])
    global sum, orig
    # Divide flux by origin population (M, Y and O don't move assumption)
    flux = df2[i, :].FLUJO /= (df2[i, :].Total * .562)

    if orig != df2[i, :].orig
        push!(edgelist, [orig, orig])
        push!(Rᵢⱼ, 1-sum)
        orig = df2[i, :].orig
        sum = flux
    else
        sum += flux
        push!(edgelist, [orig, df2[i, :].dest])
        push!(Rᵢⱼ, flux)
    end
end
push!(edgelist, [orig, orig])
push!(Rᵢⱼ, 1-sum)

# Reshape edgelist from array of arrays into 2d array
_edgelist = hcat(edgelist...)'
edgelist = zeros(Int64, 74441, 2)
edgelist[:, :] = _edgelist[:, :]


# --------------------------------
# Surface are of the mobility areas
# --------------------------------

sf = shapefile.Reader("shapefiles/celdas_marzo_2020.shp")
df = DataFrame(Areas = String[], Surface = Float64[])
for i in 0:3213
    push!(df, [sf.record(i)[5], sf.record(i)[7]/1e6])
end

# Add surface to df1
df1 = leftjoin(df1, df, on = :Areas)
# Deal with missing surfaces
df1.Surface = replace(df1.Surface, missing => 1)
sᵢ = df1.Surface

# Population number
nᵢᵍ = ones(3, 3215) # can't be set to zero -> NaNs

# approximate % taken from INE
# Y ∈ [0, 24]
# M ∈ [25, 64]
# O ∈ [65, ∞)

nᵢᵍ[1, :] = df1.Total * .244
nᵢᵍ[2, :] = df1.Total * .562
nᵢᵍ[3, :] = df1.Total * .194

# Save to file
using JLD
dict = Dict("n" => nᵢᵍ, "edgelist" => edgelist, "R" => Rᵢⱼ, "s" => sᵢ)
save("data.jld", dict)
nᵢᵍ = load("data.jld", "n")
edgelist = load("data.jld", "edgelist")
Rᵢⱼ = load("data.jld", "R")
sᵢ = load("data.jld", "s")
