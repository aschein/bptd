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
* [setup.py](https://github.com/aschein/bptd/blob/master/code/setup.py): Setup file for compiling Cython.

## Dependencies:

* numpy
* scipy
* argparse
* path
* scikit-learn
* scikit-tensor
* cython
