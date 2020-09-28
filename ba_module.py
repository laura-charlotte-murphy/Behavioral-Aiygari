import numpy as np
import matplotlib.pyplot as plt
import numba

def agrid(amax,N,amin=0,pivot=0.1):
    """
    Grid with a+pivot evenly log-spaced between amin+pivot and amax+pivot
    """
    a = np.geomspace(amin+pivot,amax+pivot,N) - pivot
    a[0] = amin # make sure *exactly* equal to amin
    return a

def up(c,gamma=1):
    """first derivative of a CRRA utility function: c^(1 - gamma) / (1 - gamma)

    Parameters
    ----------
    c       : array (float), consumption levels to be evaluated
    gamma   : float, risk aversion parameter
    """
    return c**(-gamma)

def up_inv(c,gamma=1):
    """inverse of the first derivative of a CRRA utility function: c^(1 - gamma) / (1 - gamma)

    Parameters
    ----------
    c       : array (float), consumption levels to be evaluated
    gamma   : float, risk aversion parameter
    """

    return c**(-1/gamma)

def interpolate_coord(x, xq):
    """Get representation xqi, xqpi of xq interpolated against x:
    xq = xqpi * x[xqi] + (1-xqpi) * x[xqi+1]

    Parameters
    ----------
    x    : array (n), ascending data points
    xq   : array (nq), ascending query points

    Returns
    ----------
    xqi  : array (nq), indices of lower bracketing gridpoints
    xqpi : array (nq), weights on lower bracketing gridpoints
    """
    nxq, nx = xq.shape[0], x.shape[0]
    xqi = np.empty(nxq, dtype=np.int64)
    xqpi = np.empty(nx)

    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx - 2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi + 1]

        xqpi[xqi_cur] = (x_high - xq_cur) / (x_high - x_low)
        xqi[xqi_cur] = xi
    return xqi, xqpi

def stationary(Pi, pi_seed=None, tol=1E-11, maxit=10_000):
    """Find invariant distribution of a Markov chain by iteration.

    Parameters
    ----------
    Pi          : array (n x n), markov transition matrix
    pi_seed     : array (n), initial guess for stationary distribution
    tol         : (float), maximum error distance before
    maxit       : (float), maximum number of iterations will loop before giving up

    Returns
    ----------
    pi          : array (n), invariant distribution
    """
    if pi_seed is None:
        pi = np.ones(Pi.shape[0]) / Pi.shape[0]
    else:
        pi = pi_seed

    for it in range(maxit):
        pi_new = pi @ Pi
        if np.max(np.abs(pi_new - pi)) < tol:
            break
        pi = pi_new
    else:
        raise ValueError(f'No convergence after {maxit} forward iterations!')
    pi = pi_new

    return pi

def ppol(p,l,alpha):
    """Returns t+1 beliefs as a function of t beliefs and t+1 labour state

    Parameters
    ----------
    p          : array, grid of possible beliefs
    l          : array, possible labour states
    alpha      : (float), maximum error distance before

    """
    return np.array((1 / (1 + alpha))*(l[:,np.newaxis] + alpha * p[np.newaxis,:]))

@numba.njit
def forward_iterate_p(D, Pi_T, pplus_i, pplus_pi):

    Dplus = Pi_T @ D
    Dnew = np.zeros_like(D)
    for si in range(D.shape[0]):
        for ai in range(D.shape[1]):
            i = pplus_i[si, ai]
            pi = pplus_pi[si, ai]
            d = Dplus[si, ai]

            Dnew[si, i] += d * pi
            Dnew[si, i + 1] += d * (1 - pi)

    # second step: update using transpose of Markov matrix for exogenous state z
    # take Pi_T itself as input for memory efficiency
    return Dnew

def ss_dist_p(Pi, aplus_i, aplus_pi, verbose=True):
    # start by getting stationary distribution of s
    pi = stationary(Pi)

    # need to initialize joint distribution of (s, a), assume uniform on a
    nA = aplus_i.shape[1]
    D = np.outer(pi, np.full(nA, 1 / nA))

    # Pi.T is a "view" on Pi with the wrong memory order, copy this to get right memory order (faster operations)
    Pi_T = Pi.T.copy()

    # now iterate forward until convergence
    for it in range(100_000):
        Dnew = forward_iterate_p(D, Pi_T, aplus_i, aplus_pi)

        # only check convergence every 20 iterations for efficiency
        if it % 20 == 0 and np.max(np.abs(Dnew - D)) < 1E-10:
            if verbose:
                print(f'Convergence after {it} forward iterations!')
            break
        D = Dnew
    else:
        raise ValueError(f'No convergence after {maxit} forward iterations!')

    return D

def exp_cplus(cplus,pplus_i,pplus_pi):
    cplusp = np.zeros_like(cplus)
    for ip in range(pplus_i.shape[1]):
        for si in range(cplus.shape[0]):
            pi = pplus_pi[si,ip]
            i = pplus_i[si,ip]
            cplusp[si,ip,:] = pi * cplus[si,i,:] + (1-pi) * cplus[si,i+1,:]
    return cplusp

def backward_iterate(cplus, up, up_inv, beta, pigrid, pplus_i, pplus_pi, r, y, a):
    # step one: get consumption on endogenous gridpoints
    cendog = np.zeros([pigrid.shape[1],a.shape[0]])
    cplusp = exp_cplus(cplus, pplus_i, pplus_pi)
    for ip in range(pigrid.shape[1]):
        cendog[ip, :] = up_inv(beta * (1 + r) * pigrid[:,ip] @ up(cplusp[:, ip, :]))

    cendog = np.repeat(cendog[np.newaxis,:,:],y.shape[0],axis=0)

    # step two: solve for consumption on regular gridpoints implied by Euler equation
    coh = y[:, np.newaxis] + (1 + r) * a
    coh = np.repeat(coh[:, :, np.newaxis], pigrid.shape[1], axis=2)
    coh = np.swapaxes(coh, 1, 2)

    c = np.empty_like(cendog)
    for si in range(y.shape[0]):
        for ip in range(cendog.shape[1]):
            c[si,ip, :] = np.interp(coh[si, ip, :], cendog[si, ip, :] + a, cendog[si, ip, :])

    # step three: enforce a_+ >= amin, assuming amin is lowest gridpoint of a
    aplus = coh - c
    aplus[aplus < a[0]] = a[0]
    c = coh - aplus

    return c, aplus

def ss_policy(up, up_inv, beta, pigrid, pplus_i, pplus_pi, r, y, a, verbose=True):
    # guess initial value for consumption function as 10% of cash on hand
    coh = y[:, np.newaxis] + (1 + r) * a
    coh = np.repeat(coh[:, :, np.newaxis], pigrid.shape[1], axis=2)
    coh = np.swapaxes(coh, 1, 2)

    c = 0.1 * coh

    # iterate until convergence
    for it in range(2000):
        c, aplus = backward_iterate(c, up, up_inv, beta, pigrid, pplus_i, pplus_pi, r, y, a)

        if it % 10 == 1 and np.max(np.abs(c - cold)) < 1E-9:
            if verbose:
                print(f'Took {it} iterations!')
            return c, aplus

        cold = c

def ss_dist(Pi, aplus_i, aplus_pi, pplus_i, pplus_pi, verbose=True):
    # start by getting stationary distribution of s
    pi = stationary(Pi)

    # need to initialize joint distribution of (s, a), assume uniform on a
    nA = aplus_i.shape[2]
    nP = pplus_i.shape[1]
    nY = Pi.shape[0]
    D = np.ones([nP, nY, nA]) * (1 / (nA*nY*nP))

    # Pi.T is a "view" on Pi with the wrong memory order, copy this to get right memory order (faster operations)
    Pi_T = Pi.T.copy()

    # now iterate forward until convergence
    for it in range(100_000):
        Dnew = forward_iterate4(D, Pi_T, aplus_i, aplus_pi, pplus_i, pplus_pi)

        # only check convergence every 20 iterations for efficiency
        if it % 20 == 0 and np.max(np.abs(Dnew - D)) < 1E-10:
            if verbose:
                print(f'Convergence after {it} forward iterations!')
            break
        D = Dnew
    else:
        raise ValueError(f'No convergence after {maxit} forward iterations!')

    return D

@numba.njit
def forward_iterate1(D, aplus_i, aplus_pi):
    Dnew = np.zeros_like(D)
    for ip in range(D.shape[0]):
        for si in range(D.shape[1]):
            for ai in range(D.shape[2]):
                i = aplus_i[si, ip, ai]
                pi = aplus_pi[si, ip, ai]
                d = D[ip, si, ai]

                Dnew[ip, si, i] += d * pi
                Dnew[ip, si, i + 1] += d * (1 - pi)

    # second step: update using transpose of Markov matrix for exogenous state z
    # take Pi_T itself as input for memory efficiency
    return Dnew

def forward_iterate2(Dnew,Pi_T):
    return np.einsum('ij,hjk->hik', Pi_T, Dnew)

@numba.njit
def forward_iterate3(Dnew, pplus_i, pplus_pi):
    Dplus = np.zeros_like(Dnew)

    for ip in range(Dplus.shape[0]):
        for si in range(Dplus.shape[1]):
            for ai in range(Dplus.shape[2]):
                i = pplus_i[si, ip]
                pi = pplus_pi[si, ip]
                d = Dnew[ip, si, ai]

                Dplus[i, si, ai] += d * pi
                Dplus[i + 1, si, ai] += d * (1 - pi)

    # second step: update using transpose of Markov matrix for exogenous state z
    # take Pi_T itself as input for memory efficiency
    return Dplus

def forward_iterate4(D,Pi_T,aplus_i,aplus_pi,pplus_i,pplus_pi):
    Dnew = forward_iterate1(D, aplus_i, aplus_pi)
    Dnew2 = forward_iterate2(Dnew,Pi_T)

    return forward_iterate3(Dnew2, pplus_i, pplus_pi)
