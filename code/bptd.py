import sys
import time

import numpy as np
import numpy.random as rn
import sktensor as skt
import scipy.stats as st

from copy import deepcopy
from collections import defaultdict
from sklearn.base import BaseEstimator

from utils import sample_gamma
from update import get_omp_num_threads, update_sources_compositional_2, update_sources_compositional_sparse_2, crt, sumcrt, update_sources_compositional_sparse_3

MAX_THREADS = get_omp_num_threads()


STATE_VARS = ['Lambda_SKCC',
              'Theta_NC',
              'Phi_AK',
              'Psi_TS',
              'W_d_C',
              'W_a_C',
              'W_K',
              'W_S',
              'A_N',
              'beta',
              'b',
              'd']


class BPTD(BaseEstimator):
    """Bayesian non-parametric Poisson Tucker decomposition...

    ...for dynamic multiplex directed networks.
    """
    def __init__(self, n_compressions=3, n_communities=25, n_topics=5, e=0.1, f=0.1, gam=None,
                 n_iter=1000, schedule={}, verbose=True, n_threads=MAX_THREADS, eps=1e-300):
        self.n_compressions = n_compressions
        self.n_communities = n_communities
        self.n_topics = n_topics
        self.e = e
        self.f = f
        self.gam = gam
        self.n_iter = n_iter
        self.schedule = defaultdict(int, schedule)
        self.verbose = verbose
        self.n_threads = n_threads
        self.total_iter = 0
        self.eps = eps

    def get_state(self):
        state = {s: np.copy(getattr(self, s)) for s in STATE_VARS if hasattr(self, s)}
        state['Y_SKCC'] = self.Y_SKCC
        return state

    def set_state(self, state):
        # assert all(s in STATE_VARS for s in state.keys())
        for s in STATE_VARS:
            assert s in state.keys()
        N, C = state['Theta_NC'].shape
        A, K = state['Phi_AK'].shape
        T, S = state['Psi_TS'].shape
        self.n_actors = N
        self.n_actions = A
        self.n_timesteps = T
        self.n_compressions = S
        self.n_communities = C
        self.n_topics = K
        for s in state.keys():
            setattr(self, s, deepcopy(state[s]))

    def reconstruct(self, partial_state={}, subs=None):
        Lambda_SKCC = self.Lambda_SKCC
        if 'Lambda_SKCC' in partial_state.keys():
            Lambda_SKCC = partial_state['Lambda_SKCC']

        Theta_NC = self.Theta_NC
        if 'Theta_NC' in partial_state.keys():
            Theta_NC = partial_state['Theta_NC']

        Phi_AK = self.Phi_AK
        if 'Phi_AK' in partial_state.keys():
            Phi_AK = partial_state['Phi_AK']

        Psi_TS = self.Psi_TS
        if 'Psi_TS' in partial_state.keys():
            Psi_TS = partial_state['Psi_TS']

        assert Lambda_SKCC.shape[0] == Psi_TS.shape[1]
        assert Lambda_SKCC.shape[1] == Phi_AK.shape[1]
        assert Lambda_SKCC.shape[2] == Theta_NC.shape[1]
        assert Lambda_SKCC.shape[3] == Theta_NC.shape[1]

        N = Theta_NC.shape[0]
        Lambda_CCKS = np.transpose(Lambda_SKCC, (2, 3, 1, 0))
        rates_CCKT = np.einsum('cdks,ts->cdkt', Lambda_CCKS, Psi_TS)
        rates_CCAT = np.einsum('cdkt,ak->cdat', rates_CCKT, Phi_AK)
        rates_CNAT = np.einsum('cdat,jd->cjat', rates_CCAT, Theta_NC)
        rates_NNAT = np.einsum('cjat,ic->ijat', rates_CNAT, Theta_NC)
        rates_NNAT[np.identity(N).astype(bool)] = 0

        if subs is not None:
            return rates_NNAT[subs]
        return rates_NNAT

    def _check_params(self):
        N = self.n_actors
        A = self.n_actions
        T = self.n_timesteps
        S = self.n_compressions
        K = self.n_topics
        C = self.n_communities
        assert self.Lambda_SKCC.shape == (S, K, C, C)
        assert self.Phi_AK.shape == (A, K)
        assert self.Theta_NC.shape == (N, C)
        assert self.Psi_TS.shape == (T, S)
        assert self.W_a_C.shape == (C,)
        assert self.W_d_C.shape == (C,)
        assert self.W_K.shape == (K,)
        assert self.W_S.shape == (S,)
        assert self.A_N.shape == (N,)
        for k in STATE_VARS:
            if hasattr(self, k):
                assert np.isfinite(getattr(self, k)).all()

    def partial_fit(self, partial_state, data, mask=None, initialized=False):
        assert all(s in STATE_VARS for s in partial_state.keys())
        data = self._init_data(data, mask)
        if not initialized:
            self._init_latent_params()

        for k, v in partial_state.iteritems():
            assert k in STATE_VARS
            setattr(self, k, v)
            self.schedule[k] = None
        self._check_params()

        self._update(data, mask)
        return self

    def fit(self, data, mask=None, initialized=False):
        data = self._init_data(data, mask)
        if not initialized:
            self._init_latent_params()
        self._update(data, mask)
        return self

    def score(self, data, subs):
        recon = self.reconstruct(subs=subs)
        return st.poisson.logpmf(data, recon).mean()

    def _init_data(self, data, mask=None):
        if isinstance(data, np.ndarray):
            data = skt.sptensor(data.nonzero(),
                                data[data.nonzero()],
                                data.shape)
        assert isinstance(data, skt.sptensor)
        assert data.ndim == 4
        assert data.shape[0] == data.shape[1]
        N, A, T = data.shape[1:]
        self.n_actors = N
        self.n_actions = A
        self.n_timesteps = T

        if mask is not None:
            assert isinstance(mask, np.ndarray)
            assert (mask.ndim == 2) or (mask.ndim == 3)
            assert mask.shape[-2:] == (N, N)
            assert np.issubdtype(mask.dtype, np.integer)

        return data

    def _init_latent_params(self):
        N = self.n_actors
        A = self.n_actions
        T = self.n_timesteps
        S = self.n_compressions
        C = self.n_communities
        K = self.n_topics

        if self.gam is None:
            self.gam = (0.1 ** (1. / 4)) * (S + K + C + C)
            print 'Setting gam to: %f' % self.gam
        self.beta = 1.

        self.W_S = sample_gamma(self.gam / (S + K + C + C), 1. / self.beta, size=S)
        self.W_K = sample_gamma(self.gam / (S + K + C + C), 1. / self.beta, size=K)
        self.W_d_C = sample_gamma(self.gam / (S + K + C + C), 1. / self.beta, size=C)
        self.W_a_C = sample_gamma(self.gam / (S + K + C + C), 1. / self.beta, size=C)

        self.d = 1.
        shp_SKCC = np.ones((S, K, C, C))
        shp_SKCC[:] = np.outer(self.W_d_C, self.W_d_C)
        shp_SKCC[:, :, np.identity(C).astype(bool)] = self.W_a_C * self.W_d_C
        shp_SKCC *= self.W_K[None, :, None, None]
        shp_SKCC *= self.W_S[:, None, None, None]
        self.Lambda_SKCC = sample_gamma(shp_SKCC, 1. / self.d)
        self.Psi_TS = sample_gamma(self.e, 1. / self.f, size=(T, S))
        self.Phi_AK = np.ones((A, K))
        self.Phi_AK[:, :] = rn.dirichlet(self.e * np.ones(A), size=K).T
        self.A_N = np.ones(N) * self.e
        self.b = 1.
        self.Theta_NC = np.ones((N, C))

    def _update(self, data, mask=None):
        vals_P = data.vals.astype(np.uint32)
        subs_P4 = np.asarray(zip(*data.subs), dtype=np.uint32)
        P = vals_P.size

        N = self.n_actors
        A = self.n_actions
        T = self.n_timesteps
        S = self.n_compressions
        C = self.n_communities
        K = self.n_topics

        Lambda_SKCC = self.Lambda_SKCC
        Theta_NC = self.Theta_NC
        Phi_AK = self.Phi_AK
        Psi_TS = self.Psi_TS
        W_d_C = self.W_d_C
        W_a_C = self.W_a_C
        W_K = self.W_K
        W_S = self.W_S
        A_N = self.A_N
        beta = self.beta
        b = self.b
        d = self.d

        # Hyperparameters
        if self.gam is None:
            self.gam = (0.1 ** (1. / 4)) * (S + K + C + C)
            print 'Setting gam to: %f' % self.gam
        gam = self.gam
        e = self.e
        f = self.f
        eta_A = np.ones(A) * e

        Y_s_NC = self.Y_s_NC = np.zeros((N, C), np.uint32)
        Y_r_NC = self.Y_r_NC = np.zeros((N, C), np.uint32)
        Y_AK = self.Y_AK = np.zeros((A, K), np.uint32)
        Y_TS = self.Y_TS = np.zeros((T, S), np.uint32)
        Y_SKCC = self.Y_SKCC = np.zeros((S, K, C, C), np.uint32)

        Y_2S = np.ones(2 * S, dtype=np.uint32)
        tmp_2S = np.ones(2 * S)

        # Latent CRT sources
        L_NC = np.zeros((N, C), dtype=np.uint32)
        L_K = np.zeros(K, dtype=np.uint32)
        L_S = np.zeros(S, dtype=np.uint32)
        H_N = np.zeros(N, dtype=np.uint32)

        # Masks for treating diagonals
        int_diag_CC = np.identity(C)
        int_off_CC = np.abs(1 - int_diag_CC)
        bool_diag_CC = int_diag_CC.astype(bool)

        if mask is None:
            mask = np.abs(1 - np.identity(N).astype(int))

        zeta_SCC = np.zeros((S, C, C))
        if mask.ndim == 2:
            mask_NN = mask
            zeta_SCC[:] = np.dot(Theta_NC.T, np.dot(mask_NN, Theta_NC))
            zeta_SCC *= Psi_TS.sum(axis=0)[:, None, None]
        else:
            mask_TNN = mask
            zeta_TNC = np.einsum('tij,jd->tid', mask_TNN, Theta_NC)
            zeta_TCC = np.einsum('tid,ic->tcd', zeta_TNC, Theta_NC)
            zeta_SCC[:] = np.einsum('tcd,ts->scd', zeta_TCC, Psi_TS)

        Lambda_SCC = Lambda_SKCC.sum(axis=1)
        Lambda_KCC = Lambda_SKCC.sum(axis=0)
        Lambda_CC = Lambda_KCC.sum(axis=0)

        shp_SKCC = np.ones((S, K, C, C))
        shp_SKCC[:] = np.outer(W_d_C, W_d_C)
        shp_SKCC[:, :, bool_diag_CC] = W_a_C * W_d_C
        shp_SKCC *= W_K[None, :, None, None]
        shp_SKCC *= W_S[:, None, None, None]

        schedule = self.schedule.copy()
        for k, v in schedule.items():
            if v is None:
                schedule[k] = np.inf

        curr_score = -np.inf
        if self.verbose:
            outstr = 'Starting' if self.total_iter == 0 else 'Restarting'
            print '%s inference...' % outstr

        for itn in xrange(self.n_iter):
            total_start = time.time()

            if schedule['Sources'] <= self.total_iter:

                subs_Q4 = np.asarray(zip(*np.where(self.Lambda_SKCC > self.eps))).astype(np.uint32)
                sparsity = subs_Q4.shape[0] / float(Lambda_SKCC.size)
                if self.verbose:
                    print 'Lambda sparsity: %f' % (sparsity)

                if (sparsity > 0.):
                    start = time.time()
                    update_sources_compositional_2(vals_P=vals_P,
                                                   subs_P4=subs_P4,
                                                   Theta_s_NC=Theta_NC,
                                                   Theta_r_NC=Theta_NC,
                                                   Phi_AK=Phi_AK,
                                                   Psi_TS=Psi_TS,
                                                   Lambda_SKCC=Lambda_SKCC,
                                                   Y_s_NC=Y_s_NC,
                                                   Y_r_NC=Y_r_NC,
                                                   Y_AK=Y_AK,
                                                   Y_TS=Y_TS,
                                                   Y_SKCC=Y_SKCC,
                                                   num_threads=self.n_threads)
                    end = time.time() - start
                    if self.verbose:
                        print '%f: sampling tokens compositionally' % end

                else:
                    start = time.time()
                    update_sources_compositional_sparse_3(vals_P=vals_P,
                                                          subs_P4=subs_P4,
                                                          subs_Q4=subs_Q4,
                                                          Theta_s_NC=Theta_NC,
                                                          Theta_r_NC=Theta_NC,
                                                          Phi_AK=Phi_AK,
                                                          Psi_TS=Psi_TS,
                                                          Lambda_SKCC=Lambda_SKCC,
                                                          Y_s_NC=Y_s_NC,
                                                          Y_r_NC=Y_r_NC,
                                                          Y_AK=Y_AK,
                                                          Y_TS=Y_TS,
                                                          Y_SKCC=Y_SKCC,
                                                          num_threads=self.n_threads,
                                                          eps=1e-300)
                    assert vals_P.sum() == Y_TS.sum()
                    end = time.time() - start
                    if self.verbose:
                        print '%f: alt sampling tokens sparsely' % end

            if schedule['Lambda_SKCC'] <= self.total_iter:
                start = time.time()

                shp_SKCC[:] = np.outer(W_d_C, W_d_C)
                shp_SKCC[:, :, bool_diag_CC] = W_a_C * W_d_C
                shp_SKCC *= W_K[None, :, None, None]
                shp_SKCC *= W_S[:, None, None, None]
                post_shp_SKCC = shp_SKCC + Y_SKCC

                if mask.ndim == 2:
                    mask_NN = mask
                    zeta_SCC[:] = np.dot(Theta_NC.T, np.dot(mask_NN, Theta_NC))
                    zeta_SCC *= Psi_TS.sum(axis=0)[:, None, None]
                else:
                    mask_TNN = mask
                    zeta_TNC = np.einsum('tij,jd->tid', mask_TNN, Theta_NC)
                    zeta_TCC = np.einsum('tid,ic->tcd', zeta_TNC, Theta_NC)
                    zeta_SCC = np.einsum('tcd,ts->scd', zeta_TCC, Psi_TS)
                post_rte_SKCC = d + zeta_SCC[:, None, :, :]

                Lambda_SKCC[:] = sample_gamma(post_shp_SKCC, 1. / post_rte_SKCC)
                end = time.time() - start
                if self.verbose:
                    print '%f: sampling lambda' % end

            if schedule['Psi_TS'] <= self.total_iter:
                start = time.time()

                Lambda_SCC[:] = Lambda_SKCC.sum(axis=1)
                if mask.ndim == 2:
                    mask_NN = mask
                    zeta_SCC[:] = np.dot(Theta_NC.T, np.dot(mask_NN, Theta_NC))
                    zeta_TS = (zeta_SCC * Lambda_SCC).sum(axis=(1, 2)).reshape((1, S))
                else:
                    mask_TNN = mask
                    zeta_TNC = np.einsum('tij,jd->tid', mask_TNN, Theta_NC)
                    zeta_TCC = np.einsum('tid,ic->tcd', zeta_TNC, Theta_NC)
                    zeta_TS = np.einsum('tcd,scd->ts', zeta_TCC, Lambda_SCC)

                post_shp_TS = e + Y_TS
                post_rte_TS = f + zeta_TS
                Psi_TS[:] = sample_gamma(post_shp_TS, 1. / post_rte_TS)

                end = time.time() - start
                if self.verbose:
                    print '%f: sampling psi' % end

            if schedule['Theta_NC'] <= self.total_iter:
                start = time.time()
                Lambda_SCC[:] = Lambda_SKCC.sum(axis=1)
                Psi_S = Psi_TS.sum(axis=0)

                if mask.ndim == 2:
                    zeta_CC = (Lambda_SCC * Psi_S[:, None, None]).sum(axis=0)

                    zeta_s_NC = np.dot(zeta_CC, np.dot(mask_NN, Theta_NC).T).T
                    zeta_r_NC = np.dot(np.dot(mask_NN.T, Theta_NC), zeta_CC)
                else:
                    zeta_TCC = np.einsum('scd,ts->tcd', Lambda_SCC, Psi_TS)
                    zeta_s_TCN = np.einsum('tij,jd->tid', mask_TNN, Theta_NC)
                    zeta_r_TCN = np.einsum('tij,ic->tcj', mask_TNN, Theta_NC)

                    zeta_s_NC = np.einsum('tcd,tid->ci', zeta_TCC, zeta_s_TCN).T
                    zeta_r_NC = np.einsum('tcd,tcj->dj', zeta_TCC, zeta_r_TCN).T
                zeta_NC = zeta_s_NC + zeta_r_NC

                post_shp_NC = A_N[:, None] + Y_s_NC + Y_r_NC
                post_rte_NC = b + zeta_NC

                Theta_NC[:, :] = sample_gamma(post_shp_NC, 1. / post_rte_NC)

                if mask.ndim == 2:
                    zeta_CC = np.einsum('scd,s->cd', Lambda_SCC, Psi_S)

                    zeta_s_NC = np.dot(zeta_CC, np.dot(mask_NN, Theta_NC).T).T
                    zeta_r_NC = np.dot(np.dot(mask_NN.T, Theta_NC), zeta_CC)
                else:
                    zeta_TCC = np.einsum('scd,ts->tcd', Lambda_SCC, Psi_TS)
                    zeta_s_TCN = np.einsum('tij,jd->tid', mask_TNN, Theta_NC)
                    zeta_r_TCN = np.einsum('tij,ic->tcj', mask_TNN, Theta_NC)

                    zeta_s_NC = np.einsum('tcd,tid->ci', zeta_TCC, zeta_s_TCN).T
                    zeta_r_NC = np.einsum('tcd,tcj->dj', zeta_TCC, zeta_r_TCN).T
                zeta_NC = zeta_s_NC + zeta_r_NC

                H_N[:] = 0
                for (i, c) in np.ndindex((N, C)):
                    H_N[i] += crt(Y_s_NC[i, c] + Y_r_NC[i, c], A_N[i])
                post_shp_N = e + H_N
                post_rte_N = f + np.log1p(zeta_NC / b).sum(axis=1)
                A_N[:] = sample_gamma(post_shp_N, 1. / post_rte_N)

                end = time.time() - start
                if self.verbose:
                    print '%f: sampling theta' % end

            if schedule['Phi_AK'] <= self.total_iter:
                start = time.time()
                for k in xrange(K):
                    Phi_AK[:, k] = rn.dirichlet(eta_A + Y_AK[:, k])

                end = time.time() - start
                if self.verbose:
                    print '%f: sampling phi' % end

            if any(schedule[s] <= self.total_iter for s in ['W_a_C', 'W_d_C']):
                start = time.time()
                w = W_K.sum()
                Y_SCC = Y_SKCC.sum(axis=1)

                if mask.ndim == 2:
                    zeta_SCC[:] = np.dot(Theta_NC.T, np.dot(mask_NN, Theta_NC))
                    zeta_SCC *= Psi_TS.sum(axis=0)[:, None, None]
                else:
                    zeta_TNC = np.einsum('tij,jd->tid', mask_TNN, Theta_NC)
                    zeta_TCC = np.einsum('tid,ic->tcd', zeta_TNC, Theta_NC)
                    zeta_SCC[:] = np.einsum('tcd,ts->scd', zeta_TCC, Psi_TS)
                zeta_SCC[:] = np.log1p(zeta_SCC / d)
                zeta_CC = w * (W_S[:, None, None] * zeta_SCC).sum(axis=0)

                for c in xrange(C):
                    Y_2S[:S] = Y_SCC[:, c, c]
                    m_a = sumcrt(Y_2S[:S], W_S * w * W_d_C[c] * W_a_C[c], num_threads=1)
                    zeta_a = W_d_C[c] * zeta_CC[c, c]
                    W_a_C[c] = sample_gamma(gam / (S + K + C + C) + m_a, 1. / (beta + zeta_a))

                    m_d = sumcrt(Y_2S[:S], W_S * w * W_d_C[c] * W_a_C[c], num_threads=1)
                    zeta_d = W_a_C[c] * zeta_CC[c, c]
                    for c2 in xrange(C):
                        if c == c2:
                            continue
                        Y_2S[:S] = Y_SCC[:, c, c2]
                        Y_2S[S:] = Y_SCC[:, c2, c]
                        tmp_2S[:S] = W_S * w * W_d_C[c] * W_d_C[c2]
                        tmp_2S[S:] = tmp_2S[:S]
                        m_d += sumcrt(Y_2S, tmp_2S, num_threads=1)
                        zeta_d += W_d_C[c2] * (zeta_CC[c, c2] + zeta_CC[c2, c])
                    W_d_C[c] = sample_gamma(gam / (S + K + C + C) + m_d, 1. / (beta + zeta_d))

                end = time.time() - start
                if self.verbose:
                    print '%f: sampling W_C' % end

            if schedule['W_K'] <= self.total_iter:
                start = time.time()
                shp_SCC = np.zeros((S, C, C))
                shp_SCC[:] = np.outer(W_d_C, W_d_C)
                shp_SCC[:, bool_diag_CC] = W_a_C * W_d_C
                shp_SCC *= W_S[:, None, None]
                shp_ = shp_SCC.ravel()

                for k in xrange(K):
                    L_K[k] = sumcrt(Y_SKCC[:, k].ravel(), shp_ * W_K[k], num_threads=1)

                if mask.ndim == 2:
                    zeta_SCC[:] = np.dot(Theta_NC.T, np.dot(mask_NN, Theta_NC))
                    zeta_SCC *= Psi_TS.sum(axis=0)[:, None, None]
                else:
                    zeta_TNC = np.einsum('tij,jd->tid', mask_TNN, Theta_NC)
                    zeta_TCC = np.einsum('tid,ic->tcd', zeta_TNC, Theta_NC)
                    zeta_SCC[:] = np.einsum('tcd,ts->scd', zeta_TCC, Psi_TS)
                zeta = (shp_SCC * np.log1p(zeta_SCC / d)).sum()

                post_shp_K = gam / (S + K + C + C) + L_K
                post_rte = beta + zeta

                W_K[:] = sample_gamma(post_shp_K, 1. / post_rte)

                end = time.time() - start
                if self.verbose:
                    print '%f: sampling W_K' % end

            if schedule['W_S'] <= self.total_iter:
                start = time.time()
                shp_KCC = np.zeros((K, C, C))
                shp_KCC[:] = np.outer(W_d_C, W_d_C)
                shp_KCC[:, bool_diag_CC] = W_a_C * W_d_C
                shp_KCC *= W_K[:, None, None]
                shp_CC = shp_KCC.sum(axis=0)
                shp_ = shp_KCC.ravel()

                for s in xrange(S):
                    L_S[s] = sumcrt(Y_SKCC[s].ravel(), shp_ * W_S[s], num_threads=1)

                if mask.ndim == 2:
                    zeta_SCC[:] = np.dot(Theta_NC.T, np.dot(mask_NN, Theta_NC))
                    zeta_SCC *= Psi_TS.sum(axis=0)[:, None, None]
                else:
                    zeta_TNC = np.einsum('tij,jd->tid', mask_TNN, Theta_NC)
                    zeta_TCC = np.einsum('tid,ic->tcd', zeta_TNC, Theta_NC)
                    zeta_SCC[:] = np.einsum('tcd,ts->scd', zeta_TCC, Psi_TS)
                zeta_S = (shp_CC * np.log1p(zeta_SCC / d)).sum(axis=(1, 2))

                post_shp_S = gam / (S + K + C + C) + L_S
                post_rte_S = beta + zeta_S

                W_S[:] = sample_gamma(post_shp_S, 1. / post_rte_S)

                end = time.time() - start
                if self.verbose:
                    print '%f: sampling W_S' % end

            if schedule['beta'] <= self.total_iter:
                start = time.time()
                post_shp = e + gam
                post_rte = f + W_S.sum() + W_K.sum() + W_a_C.sum() + W_d_C.sum()
                self.beta = beta = sample_gamma(post_shp, 1. / post_rte)
                end = time.time() - start
                if self.verbose:
                    print '%f: sampling beta' % end

            if schedule['b'] <= self.total_iter:
                start = time.time()
                post_shp = e + C * A_N.sum()
                post_rte = f + Theta_NC.sum()
                b = self.b = sample_gamma(post_shp, 1. / post_rte)
                end = time.time() - start
                if self.verbose:
                    print '%f: sampling b' % end

            if schedule['d'] <= self.total_iter:
                start = time.time()
                shp_CC = np.outer(W_d_C, W_d_C)
                shp_CC[bool_diag_CC] = W_a_C * W_d_C
                post_shp = e + W_S.sum() * W_K.sum() * shp_CC.sum()
                post_rte = f + Lambda_SKCC.sum()
                d = self.d = sample_gamma(post_shp, 1. / post_rte)
                end = time.time() - start
                if self.verbose:
                    print '%f: sampling d' % end

            if self.verbose:
                end = time.time() - total_start
                print 'ITERATION %d:\t\
                       Time %f:'\
                       % (self.total_iter, end)
                curr_score = score
             self.total_iter += 1
