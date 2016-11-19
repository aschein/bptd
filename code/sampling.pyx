# distutils: language = c++
# distutils: libraries = stdc++
# distutils: library_dirs = /usr/local/lib
# distutils: sources = csample.cpp
# distutils: extra_compile_args = -O3 -w -std=c++0x -fopenmp
# distutils: extra_link_args = -fopenmp
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

from cython cimport boundscheck, cdivision, nonecheck, wraparound
from cython.parallel import parallel, prange

from openmp cimport omp_get_max_threads, omp_get_thread_num

from libc.stdlib cimport malloc, free
from libc.math cimport log
import numpy as np
cimport numpy as np
import numpy.random as rn


cdef extern from "csample.h":
    double sample_uniform() nogil

cpdef int get_omp_num_threads():
    # This might not be kosher
    cdef int num_threads = omp_get_max_threads()
    return num_threads

###############################################################################
#                                  Allocation                                 #
###############################################################################


cdef int searchsorted(double val, double[::1] arr, int imax) nogil:
    cdef:
        int imin, imid
    imin = 0
    while (imin < imax):
        imid = (imin + imax) / 2
        if arr[imid] < val:
            imin = imid + 1
        else:
            imax = imid
    return imin


cpdef comp_allocate(unsigned int[::1] vals_P,
                    unsigned int[:, ::1] subs_P4,
                    double [:, ::1] Theta_s_VC,
                    double [:, ::1] Theta_r_VC,
                    double [:, ::1] Phi_AK,
                    double [:, ::1] Psi_TR,
                    double [:, :, :, ::1] Lambda_RKCC,
                    unsigned int[:, ::1] Y_s_VC,
                    unsigned int[:, ::1] Y_r_VC,
                    unsigned int[:, ::1] Y_TR,
                    unsigned int[:, ::1] Y_AK,
                    unsigned int[:, :, :, ::1] Y_RKCC,
                    size_t num_threads):

    cdef:
        size_t P, V, A, T, R, K, C
        np.intp_t p, i, j, a, t, r, c1, c2, k, thread_num
        unsigned int y, _
        double theta_s, theta_r, phi, psi, lam, norm, u
        double summand, summand_r, summand_rk, summand_rkc, summand_rkcc
        
        double [:, ::1] cdf_HR
        double [:, :, ::1] cdf_HRK
        double [:, :, :, ::1] cdf_HRKC
        double [:, :, :, :, ::1] cdf_HRKCC

        unsigned int[:, :, ::1] Y_s_HVC
        unsigned int[:, :, ::1] Y_r_HVC
        unsigned int[:, :, ::1] Y_HAK
        unsigned int[:, :, ::1] Y_HTR
        unsigned int[:, :, :, :, ::1] Y_HRKCC

    P = vals_P.size
    T, R = Psi_TR.shape[0], Psi_TR.shape[1]
    V, C = Theta_s_VC.shape[0], Theta_s_VC.shape[1]
    A, K = Phi_AK.shape[0], Phi_AK.shape[1]

    # TODO: Use standard 'threadlocal' Cython variables.
    cdf_HR = np.zeros((num_threads, R))
    cdf_HRK = np.zeros((num_threads, R, K))
    cdf_HRKC = np.zeros((num_threads, R, K, C))
    cdf_HRKCC = np.zeros((num_threads, R, K, C, C))
    Y_s_HVC = np.zeros((num_threads, V, C), dtype=np.uint32)
    Y_r_HVC = np.zeros((num_threads, V, C), dtype=np.uint32)
    Y_HAK = np.zeros((num_threads, A, K), dtype=np.uint32)
    Y_HTR = np.zeros((num_threads, T, R), dtype=np.uint32)
    Y_HRKCC = np.zeros((num_threads, R, K, C, C), dtype=np.uint32)

    Y_s_VC[:, :] = 0
    Y_r_VC[:, :] = 0
    Y_AK[:, :] = 0
    Y_TR[:, :] = 0
    Y_RKCC[:, :, :, :] = 0

    with nogil:
        for p in prange(P, schedule='static', num_threads=num_threads):
            thread_num = omp_get_thread_num()
            i = subs_P4[p, 0]
            j = subs_P4[p, 1]
            a = subs_P4[p, 2]
            t = subs_P4[p, 3]

            summand_r = 0
            for r in xrange(R):
                psi = Psi_TR[t, r]

                summand_rk = 0
                for k in xrange(K):
                    phi = Phi_AK[a, k]
                    
                    summand_rkc = 0
                    for c1 in xrange(C):
                        theta_s = Theta_s_VC[i, c1]

                        summand_rkcc = 0
                        for c2 in xrange(C):
                            theta_r = Theta_r_VC[j, c2]
                            lam = Lambda_RKCC[r, k, c1, c2]
                            summand_rkcc = summand_rkcc + (lam * theta_r)
                            cdf_HRKCC[thread_num, r, k, c1, c2] = summand_rkcc

                        summand_rkc = summand_rkc + (theta_s * summand_rkcc)
                        cdf_HRKC[thread_num, r, k, c1] = summand_rkc

                    summand_rk = summand_rk + (phi * summand_rkc)
                    cdf_HRK[thread_num, r, k] = summand_rk

                summand_r = summand_r + (psi * summand_rk)
                cdf_HR[thread_num, r] = summand_r

            y = vals_P[p]
            for _ in xrange(y):
                u = sample_uniform() 
                norm = cdf_HR[thread_num, R-1]
                r = searchsorted(u * norm, cdf_HR[thread_num], R-1)

                u = sample_uniform() 
                norm = cdf_HRK[thread_num, r, K-1]
                k = searchsorted(u * norm, cdf_HRK[thread_num, r], K-1)

                u = sample_uniform()
                norm = cdf_HRKC[thread_num, r, k, C-1]
                c1 = searchsorted(u * norm, cdf_HRKC[thread_num, r, k], C-1)

                u = sample_uniform()
                norm = cdf_HRKCC[thread_num, r, k, c1, C-1]
                c2 = searchsorted(u * norm, cdf_HRKCC[thread_num, r, k, c1], C-1)

                Y_s_HVC[thread_num, i, c1] += 1
                Y_r_HVC[thread_num, j, c2] += 1
                Y_HAK[thread_num, a, k] += 1
                Y_HTR[thread_num, t, r] += 1
                Y_HRKCC[thread_num, r, k, c1, c2] += 1

    reduce_sources(Y_s_VC, Y_r_VC, Y_TR, Y_AK, Y_RKCC, Y_s_HVC, Y_r_HVC, Y_HTR, Y_HAK, Y_HRKCC, num_threads)


cpdef reduce_sources(unsigned int[:, ::1] Y_s_VC,
                     unsigned int[:, ::1] Y_r_VC,
                     unsigned int[:, ::1] Y_TR,
                     unsigned int[:, ::1] Y_AK,
                     unsigned int[:, :, :, ::1] Y_RKCC,
                     unsigned int[:, :, ::1] Y_s_HVC,
                     unsigned int[:, :, ::1] Y_r_HVC,
                     unsigned int[:, :, ::1] Y_HTR,
                     unsigned int[:, :, ::1] Y_HAK,
                     unsigned int[:, :, :, :, ::1] Y_HRKCC,
                     size_t num_threads):
    cdef:
        size_t V, A, T, R, K, C
        np.intp_t i, j, a, t, r, c1, c2, k, thread_num
    
    V, C = Y_s_VC.shape[0], Y_s_VC.shape[1]
    A, K = Y_AK.shape[0], Y_AK.shape[1]
    T, R = Y_TR.shape[0], Y_TR.shape[1]
    
    Y_s_VC[:, :] = 0
    Y_r_VC[:, :] = 0
    Y_AK[:, :] = 0
    Y_TR[:, :] = 0
    Y_RKCC[:, :, :, :] = 0

    with nogil:
        for thread_num in xrange(num_threads):
            for i in prange(V, schedule='static', num_threads=num_threads):
                for c1 in xrange(C):
                    Y_s_VC[i, c1] += Y_s_HVC[thread_num, i, c1]
                    Y_r_VC[i, c1] += Y_r_HVC[thread_num, i, c1]
            for a in prange(A, schedule='static', num_threads=num_threads):
                for k in xrange(K):
                    Y_AK[a, k] += Y_HAK[thread_num, a, k]
            for r in prange(R, schedule='static', num_threads=num_threads):
                for t in xrange(T):
                    Y_TR[t, r] += Y_HTR[thread_num, t, r]
                for k in xrange(K):
                    for c1 in xrange(C):
                        for c2 in xrange(C):
                            Y_RKCC[r, k, c1, c2] += Y_HRKCC[thread_num, r, k, c1, c2]


###############################################################################
#                     Chinese Restaurant Table (CRT) distribution             #
###############################################################################

cdef extern from "csample.h":
    unsigned int sample_crt (const unsigned int m,
                             const double r) nogil

cpdef unsigned int _crt(unsigned int m, double r):
    return sample_crt(m, r) 

cpdef _vec_crt(unsigned int[::1] m_I, double[::1] r_I, unsigned int[::1] l_I):
    cdef size_t I = m_I.shape[0]
    assert r_I.shape[0] == I
    assert l_I.shape[0] == I

    cdef np.intp_t i
    with nogil:
        for i in prange(I, schedule='static'):
            l_I[i] = sample_crt(m_I[i], r_I[i])

cpdef unsigned int _sumcrt(unsigned int[::1] m_I, double[::1] r_I):
    cdef size_t I = m_I.size
    assert r_I.size == I

    cdef:
        np.intp_t i
        unsigned int l

    l = 0
    with nogil:
        for i in xrange(I):
           l += sample_crt(m_I[i], r_I[i])
    return l

cpdef unsigned int _par_sumcrt(unsigned int[::1] m_I, double[::1] r_I, size_t num_threads):
    cdef size_t I = m_I.size
    assert r_I.size == I

    cdef:
        np.intp_t i, thread_num
        unsigned int l
        unsigned int [::1] L_H 

    L_H = np.zeros(num_threads, dtype=np.uint32)

    with nogil:
        for i in prange(I, schedule='static', num_threads=num_threads):
            thread_num = omp_get_thread_num()
            L_H[thread_num] += sample_crt(m_I[i], r_I[i])

        l = 0
        for h in xrange(num_threads):
            l += L_H[h]
    return l

def crt(m, r, out=None):
    """
    Sample from a Chinese Restaurant Table (CRT) distribution [1].

    l ~ CRT(m, r) can be sampled as the sum of indep. Bernoullis:

            l = \sum_{n=1}^m Bernoulli(r/(r + n-1))

    where m >= 0 is integer and r >=0 is real.

    This method broadcasts the parameters m, r if ndarrays are given.
    Also will parallelize if multiple inputs are given.

    No PyRNG needed.  Randomness comes from rand() in stdlib.h.
    ----------
    m : int or ndarray of ints
    r : float or ndarray of floats
    out : ndarray, optional
          Must be same shape as m or r.
    Returns
    -------
    l : int or ndarray of ints, the sample from the CRT

    References
    ----------
    [1] M. Zhou & L. Carin. Negative Binomial Count and Mixture Modeling. 
        In IEEE (2012).
    """
    if np.isscalar(m) and np.isscalar(r):
        assert m >= 0
        assert r >= 0
        assert out is None
        return np.uint32(_crt(np.uint32(m), float(r)))  # why is _crt returning longs?

    if isinstance(m, np.ndarray) and np.isscalar(r):
        assert (m >= 0).all()
        assert r >= 0
        shp = m.shape
        m_I = m
        if m_I.dtype != np.uint32:
            m_I = m_I.astype(np.uint32)
        if len(shp) > 1:
            m_I = m_I.ravel()
        I = m_I.size
        r_I = r * np.ones(I)

    elif np.isscalar(m) and isinstance(r, np.ndarray):
        assert m >= 0
        assert (r >= 0).all()
        shp = r.shape
        r_I = r
        if r_I.dtype != float:
            r_I = r_I.astype(float)
        if len(shp) > 1:
            r_I = r_I.ravel()
        I = r_I.size
        m_I = m * np.ones(I, dtype=np.uint32)

    elif isinstance(m, np.ndarray) and isinstance(r, np.ndarray):
        assert (m >= 0).all()
        assert (r >= 0).all()
        assert m.shape == r.shape
        shp = m.shape
        m_I = m
        if m_I.dtype != np.uint32:
            m_I = m_I.astype(np.uint32)
        r_I = r
        if r_I.dtype != float:
            r_I = r_I.astype(float)
        if len(shp) > 1:
            m_I = m_I.ravel()
            r_I = r_I.ravel()

    l_I = out
    if (l_I is None) or (l_I.dtype != np.uint32) or (len(shp) > 1):
        l_I = np.empty_like(m_I, dtype=np.uint32)

    _vec_crt(m_I, r_I, l_I)

    if out is not None:
        if len(shp) > 1:
            out[:] = l_I.reshape(shp)
        elif out.dtype != np.uint32:
            out[:] = l_I
        return out
    return l_I.reshape(shp)


def sumcrt(m, r, num_threads=1):
    """
    Sample a sum of independent CRTs.

    Avoids creating an extra array before summing. Possibly unnecessary.
    ----------
    m : int or ndarray of ints
    r : float or ndarray of floats

    Returns
    -------
    l : int, the sample of the sum of CRTs
    """
    if np.isscalar(m) and np.isscalar(r):  # crt is a special case
        assert m >= 0
        assert r >= 0
        return _crt(np.uint32(m), float(r))

    if isinstance(m, np.ndarray) and np.isscalar(r):
        assert (m >= 0).all()
        assert r >= 0
        shp = m.shape
        m_I = m
        if m_I.dtype != np.uint32:
            m_I = m_I.astype(np.uint32)
        if len(shp) > 1:
            m_I = m_I.ravel()
        I = m_I.size
        r_I = r * np.ones(I)

    elif np.isscalar(m) and isinstance(r, np.ndarray):
        assert m >= 0
        assert (r >= 0).all()
        shp = r.shape
        r_I = r
        if r_I.dtype != float:
            r_I = r_I.astype(float)
        if len(shp) > 1:
            r_I = r_I.ravel()
        I = r_I.size
        m_I = m * np.ones(I, dtype=np.uint32)

    elif isinstance(m, np.ndarray) and isinstance(r, np.ndarray):
        assert (m >= 0).all()
        assert (r >= 0).all()
        assert m.shape == r.shape
        shp = m.shape
        m_I = m
        if m_I.dtype != np.uint32:
            m_I = m_I.astype(np.uint32)
        r_I = r
        if r_I.dtype != float:
            r_I = r_I.astype(float)
        if len(shp) > 1:
            m_I = m_I.ravel()
            r_I = r_I.ravel()

    if num_threads > 1:
        return _par_sumcrt(m_I, r_I, num_threads)
    else:
        return _sumcrt(m_I, r_I)


###############################################################################
#                            Gamma distribution                               #
###############################################################################


MIN_GAMMA_SHAPE = 1e-5
MIN_GAMMA_SCALE = 1e-5
MIN_GAMMA_SAMPLE = 1e-300


def sample_gamma(shp, sca, size=None):
    if isinstance(shp, np.ndarray):
        shp = np.clip(shp, a_min=MIN_GAMMA_SHAPE, a_max=None, out=shp)
    else:
        shp = np.clip(shp, a_min=MIN_GAMMA_SHAPE, a_max=None)

    if isinstance(sca, np.ndarray):
        sca = np.clip(sca, a_min=MIN_GAMMA_SCALE, a_max=None, out=sca)
    else:
        sca = np.clip(sca, a_min=MIN_GAMMA_SCALE, a_max=None)
    return np.clip(rn.gamma(shp, sca, size=size), a_min=MIN_GAMMA_SAMPLE, a_max=None)
