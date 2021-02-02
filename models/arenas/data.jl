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

df1 = CSV.read("poblacion_areas_movilidad.csv", DataFrame)
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

# Removed spaces before ; beforehand
df2 = CSV.read("flujos_unicode.csv", DataFrame)
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

orig = 1833
sum = 0
Rᵢⱼ = Float64[]
edgelist = []
for i in 1:length(df2[1])
    global sum, orig
    # Divide flux by origin population
    flux = df2[i, :].FLUJO /= df2[i, :].Total

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
nᵢᵍ[2, :] = df1.Total

# Save to file
using JLD
dict = Dict("n" => nᵢᵍ, "edgelist" => edgelist, "R" => Rᵢⱼ, "s" => sᵢ)
save("input.jld", dict)
nᵢᵍ = load("input.jld", "n")
edgelist = load("input.jld", "edgelist")
Rᵢⱼ = load("input.jld", "R")
sᵢ = load("input.jld", "s")
