################################################
### An implementation of a corrected and enhanced model of the
## original nonconvex model for piecewise linear regresssion
### appearing in
# Goldberg, N., Kim, Y., Leyffer, S. et al. Adaptively refined dynamic program for linear spline regression. Comput Optim Appl 58, 523–541 (2014). https://doi.org/10.1007/s10589-014-9647-y
#
# Implemented by N. Goldberg 12/2019-3/2020 to use the Couenne solver,
# Matthew J. Turner is acknowledged for his implementation of the original model using the Juniper solver

using JuMP
using AmplNLWriter
using LinearAlgebra
using CSV
using Plots
using StatsBase

m_regions = 3
#f = CSV.read("test2.csv")
f = CSV.read("titanium.csv")
#f = CSV.read("slump_test.data_fa.csv")
#f = CSV.read("nhtemp.csv")

m = Model(with_optimizer(AmplNLWriter.Optimizer, "couenne"))
# m = Model(() -> AmplNLWriter.Optimizer("couenne"))

# Need these lines for either CSV import implementation
p = sortperm(.X)   # sort needed for new constraints in place of breakpoint ordering
x = f.X[p]
y = f.Y[p]
n = length(x)
freqs = countmap(x)
nn = length(freqs)
xx = fill(0.0,nn)
yy = fill(0.0,nn)
global kk = 1
global ii = 1
while ii <= n
    global kk
    global ii
    freq = freqs[x[ii]]
    if freq > 1
        xx[kk] = x[ii]
        yy[kk] = mean(y[ii:ii+freq-1])
        freqs[x[ii]] = 0
        ii = ii+freq-1
        kk = kk+1
    elseif freq == 1
        xx[kk] = x[ii]
        yy[kk] = y[ii]
        kk = kk+1
    end
    ii = ii + 1
end
x = xx
y = yy

n_points = length(x)
print("\n*****************************\n")
print("After averaging y of duplicate x, dataset has n=",n_points)
#show(IOContext(stdout, :limit=>false), MIME"text/plain"(),[x y])
print("\n*****************************\n")
# Model specifications

# println(x)
# println(y)
# Derive constants M1 and M2
max_M1 = 0
MM1 = fill(0.0,n_points)
MM2 = fill(0.0,n_points)
MM3 = fill(0.0,n_points)

for i = 1:n_points
    for j = 1:n_points
        if x[j]-x[i] > MM1[i]
            MM1[i] = x[j]-x[i]
        end
        if x[i]-x[j] > MM2[i]
            MM2[i] = x[i]-x[j]
        end
        for k = 1:n_points
            res = 0
            if j !=(i)
                global res
                res = abs(y[k]-y[j]-(y[i]-y[j])/(x[i]-x[j])*(x[k]-x[j]))
            end
            if res >= max_M1
                global max_M1 = res
            end
            if res >= MM3[k]
                MM3[k] = res
            end
        end
    end
end
M1 = max_M1
# println(M1)
minI = argmin(x)
maxI = argmax(x)
xMin = x[minI]
xMax = x[maxI]
M2 = abs(xMax-xMin)
#println("Constant M1=", M1)
#println("Constant M2=", M2)
#println(MM1)
#println(MM2)
#println(MM3)

# Define optimization variables
@variable(m, E[1:n_points])
@variable(m, B[1:m_regions])
@variable(m, b[1:m_regions, 1:2])
@variable(m, 0 <= phi[1:n_points, 1:m_regions] <= 1, Int)

# Define optimization constraints
@constraint(m, constb_l[i = 1:n_points, j = 1:m_regions], -(y[i] - (B[j]*x[i]+b[j,2])) <= E[i] + MM3[i]*phi[i,j])
@constraint(m, constb_r[i = 1:n_points, j = 1:m_regions], (y[i] - (B[j]*x[i]+b[j,2])) <= E[i] +  MM3[i]*phi[i,j])
@constraint(m, constc[i = 1:n_points], sum(phi[i,j] for j = 1:m_regions) == m_regions-1)
@constraint(m, constd[i = 1:n_points, j = 1:m_regions], -MM2[i]*phi[i,j] <= b[j,1]-x[i])
@constraint(m, conste[i = 1:n_points, j = 2:m_regions], -MM1[i]*phi[i,j] <= x[i] - b[j-1,1])
@constraint(m, constf[j = 1:m_regions-1], b[j,1]*(B[j] - B[j+1]) == b[j+1,2] - b[j,2])
@constraint(m, constg[j=1:m_regions-1], b[j,1] <= b[j+1,1] )
#@constraint(m, consth[j = 1:m_regions], sum(phi[i,j] for i = 1:n_points) <= n_points-1)
@constraint(m, consti, b[m_regions,1] <= xMax)
@constraint(m, constj, b[1,1] >= xMin)

#@constraint(m, constk[i=1:n_points-1,j=2:m_regions-1], phi[i,j]+phi[i,j+1]-1 <= phi[i+1,j+1])
#@constraint(m, constl[i=1:n_points-1], phi[i,1] <= phi[i+1,1])
#@constraint(m, constm[i=1:n_points-1], phi[i+1,m_regions] <= phi[i,m_regions])

# Optimization objective
@objective(m, Min, sum((E[i])^2 for i = 1:n_points))

@time begin
optimize!(m)
end
println("Optimal Objective Function value: ", JuMP.objective_value(m))
println(JuMP.termination_status(m))
println("Optimal Solutions:")

for i in 1:m_regions
    println("B $i:", value(B[i]))
    println("b $i,1:", value(b[i,1]), "  b $i,2:", value(b[i,2]))
    for j in 1:n_points
        print("Phi $j,$i: ", value(phi[j,i]) , ";")
    end
    println("")
end

# Plot the results
gr()
plot(x,y,seriestype = :scatter,label="")#,ylims=(-.2,1.05))

lx = [value(b[j,1]) for j = 1:m_regions-1]
lx = prepend!(lx, xMin)
lx = append!(lx, xMax)
ly = [value(B[j])*lx[j+1]+value(b[j,2]) for j = 1:m_regions-1]
ly = prepend!(ly, value(B[1])*lx[1]+value(b[1,2]))
ly = append!(ly, value(B[m_regions])*lx[m_regions+1]+value(b[m_regions, 2]))
#println(lx, "ly", ly)
plot!(lx, ly,label="", linewidth = 3)
