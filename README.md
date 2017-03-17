# Bayesian Poisson Tucker decomposition
Source code for the paper: [Bayesian Poisson Tucker Decomposition for Learning the Structure of International Relations] (http://people.cs.umass.edu/~aschein/ScheinZhouBleiWallach2016_paper.pdf) by Aaron Schein, Mingyuan Zhou, David M. Blei, and Hanna Wallach, in ICML 2016.

The MIT License (MIT)

Copyright (c) 2016 Aaron Schein

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## What's included:

* [bptd.py](https://github.com/aschein/bptd/blob/master/code/bptd.py): The main code file.  Implements Gibbs sampling inference for BPTD.
* [sampling.pyx](https://github.com/aschein/bptd/blob/master/code/sampling.pyx): Sampling methods including Cython implementation of compositional allocation.
* [csample.h](https://github.com/aschein/bptd/blob/master/code/csample.cpp): Header file for a C++ implementation of sampling for the CRT distribution.
* [csample.cpp](https://github.com/aschein/bptd/blob/master/code/csample.cpp): C++ implementation of sampling for the CRT distribution.
* [setup.py](https://github.com/aschein/bptd/blob/master/code/setup.py): Setup file for compiling Cython.
* [toy_example.py](https://github.com/aschein/bptd/blob/master/code/toy_example.py): Toy example of using the model.

## Dependencies:

* numpy
* scipy
* argparse
* path
* scikit-learn
* scikit-tensor
* cython

## Example usage:
```
import numpy as np
import numpy.random as rn
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


