#=
    Represent each spin config as a cubic BitArray of size L.
    Assuming L is at most 5, each BitArray can be mapped to an Int128 since 5^3 < 128.
    Perform translations and octahedral symmetry operations, and construct a dictionary canonMap as a map from arbitrary spin configs within the domain to canonical (primitive) spin configs.
=#


#=
    Specify domain of sites.
=#

# 1-5-13-5-1
# domain = [[-2, 0, 0], [-1, 0, 0], [-1, -1, 0], [-1, 1, 0], [-1, 0, -1], [-1, 0, 1], [0, -2, 0], [0, -1, -1], [0, -1, 0], [0, -1, 1], [0, 0, -2], [0, 0, -1], [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, -1], [0, 1, 0], [0, 1, 1], [0, 2, 0], [1, 0, 0], [1, -1, 0], [1, 1, 0], [1, 0, -1], [1, 0, 1], [2, 0, 0]]

# 15551
domain = [[-2, 0, 0], [-1, 0, 0], [-1, -1, 0], [-1, 1, 0], [-1, 0, -1], [-1, 0, 1], [0, -1, 0], [0, 0, -1], [0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, -1, 0], [1, 1, 0], [1, 0, -1], [1, 0, 1], [2, 0, 0]]

# 1551
# domain = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1], [0, 0, 2]]

# 151
# domain = [[-1, 0, 0], [0, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1], [1, 0, 0]]


#=
    Shift domain such that all coordinates are natural numbers
=#
domain = hcat(domain...)
domain = [domain[i,:] .- minimum(domain[i,:]) .+ 1 for i in 1:3]
domain = Matrix(hcat(domain...))
L = maximum(vec(domain))
domain = mapslices(x->[x],domain,dims=2)[:]


#=
    Int representation of a BitArray with 1 at a single site.
=#
function unit(x,y,z)
    return Int128(1) << ((x-1) * L^2 + (y-1) * L + (z-1))
end


#=
    Generate all spin configs.
=#
spinConfigs = [Int128(0)]
for site in domain
    global spinConfigs;
    append!(spinConfigs, spinConfigs .+ unit(site[1],site[2],site[3]))
end
filter!(x->((count_ones(x) & 1)==0), spinConfigs)
spinConfigsDict = Dict{Int128,Bool}(s => false for s in spinConfigs)


#=
    Int representation of BitArray.
=#
function int(ba)
    if length(ba.chunks)==1
        return Int128(ba.chunks[1])
    else
        return Int128(ba.chunks[1]) + (Int128(ba.chunks[2]) << 64)
    end
end


#=
    Turn Int into BitArray of various dimensions.
=#
function bitTen3(n)
    tmp = BitArray(undef, L, L, L)
    if (n >> 64) == 0
        tmp.chunks[1] = n
    else
        tmp.chunks = [n & (UInt128(2)^64-1), n >> 64]
    end
    return tmp
end

function bitMat(n)
    tmp = BitArray(undef, L, L)
    tmp.chunks[1] = n
    return tmp
end

function bitVec(n)
    tmp = BitArray(undef, L)
    tmp.chunks[1] = n
    return tmp
end


#=
    Int representation of left or right slice as a BitArray of 1 dim lower.
=#
function left(n)
    return n & (2^(L^2)-1)
end

function right(n)
    return (n >> (L^2*(L-1))) & (2^(L^2)-1)
end


#=
    Octahedral symmetry actions.
=#
function cyc(n)
    return int(permutedims(bitTen3(n),(2,3,1)))
end

function tp(n)
    return int(permutedims(bitTen3(n),(2,1,3)))
end

function rev(n)
    return int(reverse(bitTen3(n), dims=1))
end

function rot(n)
    return rev(tp(n))
end

function revall(n)
    return int(reverse(bitTen3(n), dims=:))
end


#=
    Find canonical representative under translations.
=#
function translationCanonize(n)
    if n==0
        return 0
    end

    m = n
    while left(m) == 0
        m = m >> L^2
    end
    m = cyc(m)
    while left(m) == 0
        m = m >> L^2
    end
    m = cyc(m)
    while left(m) == 0
        m = m >> L^2
    end
    return cyc(m)
end


#=
    Compute orbit under octahedral group, parameterized as coset × S3.
=#
function s3(n)
    m = n
    res = [m,tp(m)]
    for i in 1:2
        m = cyc(m)
        append!(res,[m,tp(m)])
    end
    return res
end

function coset(n)
    m = n
    res = [m,revall(m)]
    for i in 1:3
        m = rot(m)
        append!(res,[m,revall(m)])
    end
    return res
end

function octahedral(n)
    res = []
    for m in s3(n)
        append!(res,coset(m))
    end
    return unique(res)
end


#=
    Find canonical representative under translations and octahedral symmetry.
=#
function canonize(n)
    if n==0
        return 0
    end
    oct = octahedral(n)
    return minimum(translationCanonize.(oct))
end


#=
    Given a spin config represented by Int n, canonize it to can, compute its orbit under translations, and set dictionary canonMap[orbit] = can.
=#
function translationOrbit(canonMap, n)
    if n==0
        canonMap[0] = 0
        return 0
    end

    can = translationCanonize(n)

    m = can
    orb1 = [cyc(m)]
    while right(m) == 0
        m = m << L^2
        append!(orb1, cyc(m))
    end
    for p1 in orb1
        q1 = p1
        orb2 = [cyc(q1)]
        while right(q1) == 0
            q1 = q1 << L^2
            append!(orb2, cyc(q1))
        end
        for p2 in orb2
            q2 = p2
            if haskey(spinConfigsDict,cyc(q2))
                canonMap[cyc(q2)] = can
            end
            while right(q2) == 0
                q2 = q2 << L^2
                if haskey(spinConfigsDict,cyc(q2))
                    canonMap[cyc(q2)] = can
                end
            end
        end
    end
end


#=
    Given a spin config represented by Int n, canonize it to can, compute its orbit under translations and octahedral symmetry, and set dictionary canonMap[orbit] = can.
=#
function orbit(canonMap, n)
    if n==0
        canonMap[0] = 0
        return 0
    end

    can = canonize(n)
    oct = octahedral(n)

    for m in oct
        orb1 = [cyc(m)]
        while right(m) == 0
            m = m << L^2
            append!(orb1, cyc(m))
        end
        for p1 in orb1
            q1 = p1
            orb2 = [cyc(q1)]
            while right(q1) == 0
                q1 = q1 << L^2
                append!(orb2, cyc(q1))
            end
            for p2 in orb2
                q2 = p2
                tmp = cyc(q2)
                if haskey(spinConfigsDict,tmp)
                    canonMap[tmp] = can
                end
                while right(q2) == 0
                    q2 = q2 << L^2
                    tmp = cyc(q2)
                    if haskey(spinConfigsDict,tmp)
                        canonMap[tmp] = can
                    end
                end
            end
        end
    end
end


#=
    Scan over spin configs to produce canonMap under translations.
=#
function genTranslationCanonical(canonMap)
    progress = 0
    for s in spinConfigs
        if !haskey(canonMap,s)
            translationOrbit(canonMap,s)
        end

        newProgress = 100 * length(keys(canonMap)) / length(spinConfigs)

        if newProgress - progress > 1
            progress = newProgress
            println(Int(round(progress)), '%')
        end

        if length(keys(canonMap)) == length(spinConfigs)
            break
        end
    end
end


#=
    Scan over spin configs to produce canonMap under translations and octahedral symmetry.
=#
function genCanonical(canonMap)
    progress = 0
    for s in spinConfigs
        if !haskey(canonMap,s)
            orbit(canonMap,s)
        end

        newProgress = 100 * length(keys(canonMap)) / length(spinConfigs)

        if newProgress - progress > 1
            progress = newProgress
            println(Int(round(progress)), '%')
        end

        if length(keys(canonMap)) == length(spinConfigs)
            break
        end
    end
end


canonMap = Dict{Int128,Int128}()
# @time genTranslationCanonical(canonMap)
@time genCanonical(canonMap)
println(length(unique(collect(values(canonMap)),dims=1)))