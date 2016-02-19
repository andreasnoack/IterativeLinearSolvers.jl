module IterativeLinearSolvers

using ToeplitzMatrices

import Base.LinAlg: BlasReal, A_mul_B!

export cgs, cg, gmres

typealias Preconditioner{T} Union{AbstractMatrix{T}, Factorization{T}}

function cg{T<:BlasReal}(A::AbstractMatrix{T},
    x::AbstractVector{T},
    b::AbstractVector{T},
    M::Preconditioner{T},
    max_it::Integer,
    tol::Real)
#  -- Iterative template routine --
#     Univ. of Tennessee and Oak Ridge National Laboratory
#     October 1, 1993
#     Details of this algorithm are described in "Templates for the
#     Solution of Linear Systems: Building Blocks for Iterative
#     Methods", Barrett, Berry, Chan, Demmel, Donato, Dongarra,
#     Eijkhout, Pozo, Romine, and van der Vorst, SIAM Publications,
#     1993. (ftp netlib2.cs.utk.edu; cd linalg; get templates.ps).
#
#  [x, error, iter, flag] = cg(A, x, b, M, max_it, tol)
#
# cg.m solves the symmetric positive definite linear system Ax=b
# using the Conjugate Gradient method with preconditioning.
#
# input   A        REAL symmetric positive definite matrix
#         x        REAL initial guess vector
#         b        REAL right hand side vector
#         M        REAL preconditioner matrix
#         max_it   INTEGER maximum number of iterations
#         tol      REAL error tolerance
#
# output  x        REAL solution vector
#         error    REAL error norm
#         iter     INTEGER number of iterations performed
#         flag     INTEGER: 0 = solution found to tolerance
#                           1 = no convergence given max_it

    n = length(b)
    flag = 0                                 # initialization
    iter = 0

    bnrm2 = norm(b)
    if bnrm2 == 0.0 bnrm2 = one(T) end

    local ρ₁
    z = zeros(T, n)
    q = zeros(T, n)
    p = zeros(T, n)
    # r = copy(b)
    # A_mul_B!(-one(T),A,x,one(T),r)
    r = b - A*x
    error = norm(r)/bnrm2
    if error < tol
        return
    end

    for iter = 1:max_it                       # begin iteration

        z[:] = r
        A_ldiv_B!(M, z)
        # z[:] = M\r
        ρ = dot(r,z)

        if iter > 1                       # direction vector
            β = ρ/ρ₁
            for l = 1:n
                p[l] = z[l] + β*p[l]
            end
        else
            p[:] = z
        end

        # A_mul_B!(one(T),A,p,zero(T),q)
        q[:] = A*p
        α = ρ / dot(p,q)
        for l = 1:n
            x[l] += α*p[l]                    # update approximation vector
            r[l] -= α*q[l]                    # compute residual
        end

        error = norm(r)/bnrm2                     # check convergence
        if error <= tol break end

        ρ₁ = ρ

    end

    if error > tol flag = 1 end                 # no convergence
    return x, error, iter, flag
end

function cgs{T<:BlasReal}(A::AbstractMatrix{T},
    x::AbstractVector{T},
    b::AbstractVector{T},
    M::Preconditioner{T},
    max_it::Integer,
    tol::Real)
#  -- Iterative template routine --
#     Univ. of Tennessee and Oak Ridge National Laboratory
#     October 1, 1993
#     Details of this algorithm are described in "Templates for the
#     Solution of Linear Systems: Building Blocks for Iterative
#     Methods", Barrett, Berry, Chan, Demmel, Donato, Dongarra,
#     Eijkhout, Pozo, Romine, and van der Vorst, SIAM Publications,
#     1993. (ftp netlib2.cs.utk.edu; cd linalg; get templates.ps).
#
#  [x, error, iter, flag] = cgs(A, x, b, M, max_it, tol)
#
# cgs.m solves the linear system Ax=b using the
# Conjugate Gradient Squared Method with preconditioning.
#
# input   A        REAL matrix
#         x        REAL initial guess vector
#         b        REAL right hand side vector
#         M        REAL preconditioner
#         max_it   INTEGER maximum number of iterations
#         tol      REAL error tolerance
#
# output  x        REAL solution vector
#         error    REAL error norm
#         iter     INTEGER number of iterations performed
#         flag     INTEGER: 0 = solution found to tolerance
#                           1 = no convergence given max_it

    iter = 0                               # initialization
    flag = 0

    n = length(b)
    bnrm2 = norm(b)
    if bnrm2 == 0.0 bnrm2 = one(T) end

    u = zeros(T, n)
    p = zeros(T, n)
    p̂ = zeros(T, n)
    q = zeros(T, n)
    û = zeros(T,n)
    v̂ = zeros(T, n)
    ρ = zero(T)
    ρ₁ = ρ
    r = copy(b)
    A_mul_B!(-one(T),A,x,one(T),r)
    # r = b - A*x
    error = norm(r)/bnrm2

    if error < tol return x, error, iter, flag end

    r_tld = copy(r)

    for iter = 1:max_it                    # begin iteration

        ρ = dot(r_tld,r)
        if ρ == 0.0 break end

        if iter > 1                     # direction vectors
            β = ρ/ρ₁
            for l = 1:n
                u[l] = r[l] + β*q[l]
                p[l] = u[l] + β*(q[l] + β*p[l])
            end
        else
            u[:] = r
            p[:] = u
        end

        p̂[:] = p
        A_ldiv_B!(M, p̂)
        # p̂[:] = M\p
        A_mul_B!(one(T),A,p̂,zero(T),v̂)    # adjusting scalars
        # v̂[:] = A*p̂
        α = ρ/dot(r_tld,v̂)
        for l = 1:n
            q[l] = u[l] - α*v̂[l]
            û[l] = u[l] + q[l]
        end
        A_ldiv_B!(M, û)
        # û[:] = M\û

        for l = 1:n
            x[l] += α*û[l]                 # update approximation
        end

        A_mul_B!(-α,A,û,one(T),r)
        # r[:] -= α*(A*û)
        error = norm(r)/bnrm2           # check convergence
        if error <= tol break end

        ρ₁ = ρ

    end

    if error <= tol                      # converged
        flag = 0
    elseif ρ == 0.0                  # breakdown
        flag = -1
    else                                    # no convergence
        flag = 1
    end
    return x, error, iter, flag
end

function gmres{T<:BlasReal}(A::Any,
    b::AbstractVector{T},
    x::AbstractVector{T} = randn(length(b)),
    M::Any = eye(Diagonal{T}, length(b)),
    restrt::Integer = length(b) |> t -> min(t - 1, int(10log10(t))),
    max_it::Integer = 1000, tol::Real = eps())
#  -- Iterative template routine --
#     Univ. of Tennessee and Oak Ridge National Laboratory
#     October 1, 1993
#     Details of this algorithm are described in "Templates for the
#     Solution of Linear Systems: Building Blocks for Iterative
#     Methods", Barrett, Berry, Chan, Demmel, Donato, Dongarra,
#     Eijkhout, Pozo, Romine, and van der Vorst, SIAM Publications,
#     1993. (ftp netlib2.cs.utk.edu; cd linalg; get templates.ps).
#
# [x, error, iter, flag] = gmres( A, x, b, M, restrt, max_it, tol )
#
# gmres.m solves the linear system Ax=b
# using the Generalized Minimal residual ( GMRESm ) method with restarts .
#
# input   A        REAL nonsymmetric positive definite matrix
#         x        REAL initial guess vector
#         b        REAL right hand side vector
#         M        REAL preconditioner matrix
#         restrt   INTEGER number of iterations between restarts
#         max_it   INTEGER maximum number of iterations
#         tol      REAL error tolerance
#
# output  x        REAL solution vector
#         error    REAL error norm
#         iter     INTEGER number of iterations performed
#         flag     INTEGER: 0 = solution found to tolerance
#                           1 = no convergence given max_it

    iter = 0                                         # initialization
    flag = 0

    bnrm2 = norm(b)
    if bnrm2 == 0.0 bnrm2 = one(T) end

    r = copy(b)
    A_mul_B!(-one(T), A, x, one(T), r)
    # r = b - A*x
    A_ldiv_B!(M, r)
    # r[:] = M\r
    error = norm(r) / bnrm2
    if error < tol return x, error, iter, flag end

    n = length(x)                               # initialize workspace
    m = restrt
    i = 1
    V = zeros(n,m+1)
    H = zeros(m+1,m)
    cs = zeros(m)
    sn = zeros(m)
    s = zeros(n)
    w = zeros(n)

    for iter = 1:max_it                              # begin iteration
        r[:] = b
        A_mul_B!(-one(T),A,x,one(T),r)
        # r[:] = M\(b - A*x)
        A_ldiv_B!(M, r)
        nrmr = norm(r)
        for l = 1:n
            V[l,1] = r[l]/nrmr
        end
        fill!(s,zero(T))
        s[1] = nrmr
        for i = 1:m                                   # construct orthonormal
            A_mul_B!(one(T),A,sub(V,1:n,i),zero(T),w)
            # w[:] = A*sub(V,1:n,i)
            A_ldiv_B!(M, w)                              # basis using Gram-Schmidt
            # w[:] = M\(A*sub(V,1:n,i))
            for k = 1:i
                H[k,i] = dot(w, sub(V,1:n,k))
                for l = 1:n
                    w[l] -= H[k,i]*V[l,k]
                end
            end
            H[i+1,i] = norm(w)
            for l = 1:n
                V[l,i+1] = w[l]/H[i+1,i]
            end
            for k = 1:i-1                              # apply Givens rotation
                temp = cs[k]*H[k,i] + sn[k]*H[k+1,i]
                H[k+1,i] = -sn[k]*H[k,i] + cs[k]*H[k+1,i]
                H[k,i] = temp
            end
            cs[i], sn[i] = rotmat(H[i,i], H[i+1,i]) # form i-th rotation matrix
            temp = cs[i]*s[i]                       # approximate residual norm
            s[i+1] = -sn[i]*s[i]
            s[i] = temp
            H[i,i] = cs[i]*H[i,i] + sn[i]*H[i+1,i]
            H[i+1,i] = zero(T)
            error = abs(s[i+1])/bnrm2
            if error <= tol                        # update approximation
                y = H[1:i,1:i] \ s[1:i]                 # and exit
                for lo = 1:n
                    for li = 1:i
                        x[lo] += V[lo,li]*y[li]
                    end
                end
                break
            end
        end

        if error <= tol break end
        y = H[1:m,1:m]\s[1:m]
        for lo = 1:n
            for li = 1:m
                x[lo] += V[lo,li]*y[li]               # update approximation
            end
        end
        r[:] = b
        A_mul_B!(-one(T),A,x,one(T),r)
        A_ldiv_B!(M, r)
        # r[:] = M\(b - A*x)
        s[i+1] = norm(r)
        error = s[i+1]/bnrm2                          # check convergence
        if error <= tol break end
    end

    if error > tol flag = 1 end                 # converged
    return x, error, iter, flag
end

function rotmat{T<:Real}(a::T, b::T)
#
# Compute the Givens rotation matrix parameters for a and b.
#
    if b == 0
        return one(T), zero(T)
    elseif abs(b) > abs(a)
        temp = a / b
        s = one(T) / sqrt(one(T) + temp*temp)
        return temp * s, s
    else
        temp = b / a
        c = one(T) / sqrt( one(T) + temp*temp)
        return c, temp * c
    end
end

A_mul_B!(α::Number, f::Function, x::AbstractVector, β::Number, y::AbstractVector) = y[:] = α*f(x) + β*y
end
