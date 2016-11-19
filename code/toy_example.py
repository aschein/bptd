import numpy as np
import numpy.random as rn
from IPython import embed
from bptd import BPTD

V = 10  # number of actors
A = 4   # number of action types
T = 5   # number of time steps

Y = rn.poisson(10, size=(V, V, A, T))  # toy example of a count tensor of size V x V x A x T
Y[np.identity(V).astype(bool)] = 0     # the diagonal is always assumed undefined (set to 0)

C = 4   # number of communities of actors
K = 2   # number of communities of actors
R = 2   # number of regimes of time steps

model = BPTD(n_regimes=R,
             n_communities=C,
             n_topics=K,
             n_iter=1000,        # how many Gibbs sampling iterations
             verbose=True)       # whether to printout information each iteration

model.fit(Y)

Theta_VC = model.Theta_VC         # actor-community factor matrix, size V X C
Phi_AK = model.Phi_AK             # action-topic factor matrix, size A x K
Psi_TR = model.Psi_TR             # time-regime factor matrix, size T x R
Lambda_RKCC = model.Lambda_RKCC   # core tensor, size R x K x C x C

recon = model.reconstruct()               # model reconstruction of the training data Y
idx = ~np.identity(V).astype(bool)        # off-diagonal indices
mae = np.abs(Y[idx] - recon[idx]).mean()  # mean absolute error on training data

embed()
