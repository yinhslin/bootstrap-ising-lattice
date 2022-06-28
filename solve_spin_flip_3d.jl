#=
    Numerically solve the spin-flip equations at given value of J by QR decomposition.

    float 
        String specifying precision, e.g. Float64, Float64x2, Float64x4.

    rev
        Whether to reverse the ordering of spin configs.
        The algorithm below solves earlier ones in terms of later ones, so if original ordering is from simple to complex.
        Reversing the ordering chooses the independent spin configs to be the simple ones.
=#


using LinearAlgebra
# using SparseArrays, SuiteSparse
using MultiFloats
# using DoubleFloats
using DelimitedFiles
using MatrixMarket
# using ZChop
# using JLD2


#=
    Command line.
=#
# float = ARGS[1]
# jnum = parse(Int64,ARGS[2])
# jden = parse(Int64,ARGS[3])
# rev = parse(Bool,ARGS[4])


#=
    REPL.
=#
float = "Float64"
jnum = 4
jden = 10
rev = false


#=
    Path to save results.
=#
j = string(jnum/jden)
if rev
    path = "QR_" * float * "_Rev-J" * j * "/"
else
    path = "QR_" * float * "-J" * j * "/"
end
if !ispath(path)
	mkdir(path)
end
println(path)


#=
    Precision setting.
=#
MyFloat = eval(Symbol(float))
if MyFloat==Float64 || MyFloat==Float64x1
    setprecision(BigFloat,64)
elseif MyFloat==Double64 || MyFloat==Float64x2
    setprecision(BigFloat,128)
else
    setprecision(BigFloat,256) 
end
J = BigFloat(jnum) / BigFloat(jden)


#=
    Construct matrix representing spin-flip equations.
=#
A2 = (-245 + 270 * cosh(4*J) - 27 * cosh(8*J) + 2 * cosh(12*J))/720
A4 = (28 - 39 * cosh(4*J) + 12 * cosh(8*J) - cosh(12*J))/1152
A6 = (-10 + 15 * cosh(4*J) - 6 * cosh(8*J) + cosh(12*J))/23040
B1 = (45 * J * sinh(4*J) - 9 * J * sinh(8*J) + J * sinh(12*J))/(60 * J)
B3 = -((13 * J * sinh(4*J) - 8 * J * sinh(8*J) + J * sinh(12*J))/(192 * J))
B5 = (5 * J * sinh(4*J) - 4 * J * sinh(8*J) + J * sinh(12*J))/(3840 * J)
coef = [A2,A4,A6,B1,B3,B5,1]
coef = MyFloat.(coef)

A = sum([ coef[i] * MatrixMarket.mmread("A3D/A3D-" * string(i) * ".mtx") for i in 1:7 ])
B = sum([ coef[i] * MatrixMarket.mmread("B3D/B3D-" * string(i) * ".mtx") for i in 1:7 ])

b = vec(B)
if rev
    A = reverse(A, dims=2)
end
A = hcat(A,b)


#=
    Dense QR decomposition with pivoting.

    Built-in QR does not allow multiprecision sparse matrices, so for multiprecision, turn A from sparse to dense.
=#
A = Matrix(A) # Turns A from sparse to dense.
@time AQR = qr(A, Val(true)) # Val(true) turns on pivoting.
diagR = abs.(diag(AQR.R))
dep = findall(x -> abs(x)>1e-5, diagR)
ind = filter(x -> x ∉ dep, 1:size(A)[2])
varIdx = vcat(dep,ind)
p = AQR.p[varIdx]


#=
    Sparse QR decomposition with pivoting.

    Built-in QR only allows at most double precision.

    TODO:  Find workaround for multiprecision sparse matrices.
=#
# @time AQR = qr(A)
# diagR = abs.(diag(AQR.R))
# dep = findall(x -> abs(x)>1e-5, diagR)
# ind = filter(x -> x ∉ dep, 1:size(A)[2])
# varIdx = vcat(dep,ind)
# p = sortperm(AQR.rpivinv)[varIdx]


#=
    Row reduction.
=#
X = AQR.R[dep,dep]
Y = AQR.R[dep,ind]
@time sol = X \ Y
# sol = zchop.(sol, 10^-12)


#=
    Save solutions.

    TODO: Find more efficient way to store coefficient matrix.
=#
open(path * "sol_coefs.csv", "w") do io
    writedlm(io, sol, ',')
end

# Permutation of spin configs by pivoting.
open(path * "var_idx.csv", "w") do io
    writedlm(io, p, ',')
end

# @save path * "sol.jld2" sol p


#=
    Checks.
=#
# Verify the absense of overflow numbers which signify numerical instability.
println(maximum(abs.(sol)))

# Verify that solution is in kernel of matrix representing spin-flip equations.
println(maximum(abs.(vec(A[:,p] * vcat(sol,-Matrix(1I,630,630)))))) # Dense
# println(maximum(abs.(vec(A * vcat(sol,-Matrix(1I,630,630))[sortperm(p),:])))) # Sparse