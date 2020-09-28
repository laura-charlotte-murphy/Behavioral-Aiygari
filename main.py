import numpy as np
import numba
import matplotlib.pyplot as plt
from ba_module import ppol, interpolate_coord, ss_dist_p, agrid, ss_policy, up, up_inv, ss_dist, forward_iterate1, forward_iterate2, forward_iterate3, forward_iterate4

# Calibration
beta = 1 - 0.08/4
r = 0.01/4
w = 3
b = w*0.25

N_y = 2
N_p = 201
N_a = 500

# Note: I double checked this grid is wide enough for all the different income processes to work
a = agrid(amin=0, amax=150, N=N_a)
p = np.linspace(0,1,N_p)
pigrid = np.array([1 - p, p])
y = np.array([b,w])
l = np.array([0,1])

p_true = 0.85
Pi = np.array([(1 - p_true, p_true),(1 - p_true, p_true)])
Pi_T = Pi.T

# Setting up the beliefs process
alpha = 9
#pplus gives you a N_y x N_p matrix where if you HAD belief index ip and you GOT labour shock yi then your
# updated beliefs are pplus(yi,ip)
pplus = ppol(p,l,alpha)

pplus_i = np.empty(pplus.shape, dtype=np.int64)
pplus_pi = np.empty(pplus.shape)

for s in range(N_y):
    pplus_i[s,:], pplus_pi[s,:] = interpolate_coord(p, pplus[s, :])

# Looking at SS distribution of beliefs:
Dp = ss_dist_p(Pi, pplus_i, pplus_pi, verbose=True)
plt.plot(p,Dp.sum(axis=0))
plt.show()

# Looking at policy functions
c, aplus = ss_policy(up, up_inv, beta, pigrid, pplus_i, pplus_pi, r, y, a)

# Looking at SS distribution of beliefs:
plt.plot(a, aplus[1,200,:]-a, label=f'p={p[200]:.2f}', linewidth=4)
plt.plot(a, aplus[1,180,:]-a, label=f'p={p[180]:.2f}', linewidth=4)
plt.plot(a, aplus[1,100,:]-a, label=f'p={p[100]:.2f}', linewidth=4)
plt.plot(a, aplus[1,50,:]-a, label=f'p={p[50]:.2f}', linewidth=4)
plt.plot(a, aplus[1,0,:]-a, label=f'p={p[0]:.2f}', linewidth=4)
plt.legend(); plt.show()

aplus_i = np.empty(aplus.shape, dtype=np.int64)
aplus_pi = np.empty(aplus.shape)
for s in range(N_y):
    for ip in range(N_p):
        aplus_i[s,ip,:], aplus_pi[s,ip,:] = interpolate_coord(a, aplus[s, ip, :])

D = ss_dist(Pi, aplus_i, aplus_pi, pplus_i, pplus_pi, verbose=True)
D_a = D.sum(axis=0)
D_a = D_a.sum(axis=0)

plt.plot(a[0:200],D_a[0:200])
plt.show()