module IterativeLinearSolvers

using ToeplitzMatrices

import Base.LinAlg: BlasReal

export cgs, cg, gmres

function cg{T<:BlasReal}(A::AbstractMatrix{T}, x::AbstractVector{T}, b::AbstractVector{T}, M::AbstractMatrix{T}, max_it::Integer, tol::Real)
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
    if bnrm2 == 0.0 bnrm2 = 1.0 end

    r = copy(b)
    z = zeros(T, n)
    q = zeros(T, n)
    p = zeros(T, n)
    # A_mul_B!(-one(T),A,x,one(T),r)
    r[:] -= A*x
    error = norm(r)/bnrm2
    if error < tol return end

    for iter = 1:max_it                       # begin iteration

        z[:] = r
        solve!(M, z)
        rho = dot(r,z)

        if iter > 1                       # direction vector
            beta = rho/rho_1
            for l = 1:n
                p[l] = z[l] + beta*p[l]
            end
        else
            p[:] = z
        end

        # A_mul_B!(one(T),A,p,zero(T),q)
        q[:] = A*p
        alpha = rho / dot(p,q)
        for l = 1:n
            x[l] += alpha*p[l]                    # update approximation vector
            r[l] -= alpha*q[l]                    # compute residual
        end
                        
        error = norm(r)/bnrm2                     # check convergence
        if error <= tol break end 

        rho_1 = rho

    end

    if error > tol flag = 1 end                 # no convergence
    return x, error, iter, flag
end

function cgs{T<:BlasReal}(A::AbstractMatrix{T}, x::AbstractVector{T}, b::AbstractVector{T}, M::AbstractMatrix{T}, max_it::Integer, tol::Real)
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
    if bnrm2 == 0.0 bnrm2 = 1.0 end

    r = copy(b)
    u = zeros(T, n)
    p = zeros(T, n)
    p_hat = zeros(T, n)
    q = zeros(T, n)
    u_hat = zeros(T,n)
    v_hat = zeros(T, n)
    rho = 0.0
    # A_mul_B!(-one(T),A,x,one(T),r)
    r[:] -= A*x
    error = norm(r)/bnrm2

    if error < tol return x, error, iter, flag end

    r_tld = copy(r)

    for iter = 1:max_it                    # begin iteration

        rho = dot(r_tld,r)
        if rho == 0.0 break end

        if iter > 1                     # direction vectors
            beta = rho/rho_1
            for l = 1:n
                u[l] = r[l] + beta*q[l]
                p[l] = u[l] + beta*(q[l] + beta*p[l])
            end
        else
            u[:] = r
            p[:] = u
        end

        p_hat[:] = p
        solve!(M, p_hat)
        # A_mul_B!(one(T),A,p_hat,zero(T),v_hat)    # adjusting scalars
        v_hat[:] = A*p_hat
        alpha = rho/dot(r_tld,v_hat)
        for l = 1:n
            q[l] = u[l] - alpha*v_hat[l]
            u_hat[l] = u[l] + q[l]
        end
        solve!(M, u_hat)

        for l = 1:n
            x[l] += alpha*u_hat[l]                 # update approximation
        end

        # A_mul_B!(-alpha,A,u_hat,one(T),r)
        r[:] -= alpha*(A*u_hat)
        error = norm(r)/bnrm2           # check convergence
        if error <= tol break end

        rho_1 = rho

    end

    if error <= tol                      # converged
        flag = 0
    elseif rho == 0.0                  # breakdown
        flag = -1
    else                                    # no convergence
        flag = 1
    end
    return x, error, iter, flag
end

function gmres{T<:BlasReal}(A::AbstractMatrix{T}, x::AbstractVector{T}, b::AbstractVector{T}, M::AbstractMatrix{T}, restrt::Integer, max_it::Integer, tol::Real)
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
    if bnrm2 == 0.0 bnrm2 = 1.0 end

    r = copy(b)
    # A_mul_B!(-one(T), A, x, one(T), r)
    r[:] -= A*x
    solve!(M, r)
    error = norm(r) / bnrm2
    if error < tol return x, error, iter, flag end

    n = size(A, 1)                                  # initialize workspace
    m = restrt
    i = 1
    V = zeros(n,m+1)
    H = zeros(m+1,m)
    cs = zeros(m)
    sn = zeros(m)
    s = zeros(n)
    w = zeros(n)

    for iter = 1:max_it                              # begin iteration
        r[:] = b - A*x
        # A_mul_B!(-one(T),A,x,one(T),r)
        solve!(M, r)
        nrmr = norm(r)
        V[:,1] = r/nrmr
        fill!(s,zero(T))
        s[1] = nrmr
        for i = 1:m                                   # construct orthonormal
            # A_mul_B!(one(T),A,sub(V,1:n,i),zero(T),w)
            w[:] = A*sub(V,1:n,i)
            solve!(M, w)                              # basis using Gram-Schmidt
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
        r[:] = b - A*x                                     # compute residual
        # A_mul_B!(-one(T),A,x,one(T),r)
        solve!(M, r)
        s[i+1] = norm(r)
        error = s[i+1]/bnrm2                          # check convergence
        if error <= tol break end
    end

    if error > tol flag = 1 end                 # converged
    return x, error, iter, flag
end

function rotmat(a::Real, b::Real)
#
# Compute the Givens rotation matrix parameters for a and b.
#
    if b == 0.0
        return 1.0, 0.0
    elseif abs(b) > abs(a)
        temp = a / b
        s = 1.0 / sqrt(1.0 + temp*temp)
        return temp * s, s
    else
        temp = b / a
        c = 1.0 / sqrt( 1.0 + temp*temp)
        return c, temp * c
    end
end
end