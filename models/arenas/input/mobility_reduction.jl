"""
Compute mobility reduction of given days with baseline
"""
using CSV, DataFrames
df = CSV.read("flujos0.csv", DataFrame)

a1 = CSV.read("Flujos+15_O-D_M1_16MAR.csv", DataFrame)
a2 = CSV.read("Flujos+15_O-D_M1_30MAR.csv", DataFrame)
a3 = CSV.read("Flujos+15_O-D_M1_13ABR.csv", DataFrame)
a4 = CSV.read("Flujos+15_O-D_M1_27ABR.csv", DataFrame)
a5 = CSV.read("Flujos+15_O-D_M1_11MAY.csv", DataFrame)

println(sum(a1.FLUJO)/sum(df.FLUJO))
println(sum(a2.FLUJO)/sum(df.FLUJO))
println(sum(a3.FLUJO)/sum(df.FLUJO))
println(sum(a4.FLUJO)/sum(df.FLUJO))
println(sum(a5.FLUJO)/sum(df.FLUJO))
