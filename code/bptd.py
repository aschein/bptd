import sys
import time

import numpy as np
import numpy.random as rn
import sktensor as skt
import scipy.stats as st

from copy import deepcopy
from collections import defaultdict
from sklearn.base import BaseEstimator

from sampling import get_omp_num_threads, comp_allocate, crt, sumcrt, sample_gamma

MAX_THREADS = get_omp_num_threads()


STATE_VARS = ['Lambda_SKCC',
              'Theta_NC',
              'Phi_AK',
              'Psi_TS',
              'eta_d_C',
              'eta_a_C',
              'nu_K',
              'rho_S',
              'alpha_N',
              'beta',
              'delta',
              'zeta']


class BPTD(BaseEstimator):
    """Bayesian Poisson Tucker decomposition"""
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
        state = dict([(s, np.copy(getattr(self, s))) for s in STATE_VARS if hasattr(self, s)])
        state['Y_SKCC'] = self.Y_SKCC.copy()
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
        assert self.eta_a_C.shape == (C,)
        assert self.eta_d_C.shape == (C,)
        assert self.nu_K.shape == (K,)
        assert self.rho_S.shape == (S,)
        assert self.alpha_N.shape == (N,)
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
        self.zeta = 1.
        self.delta = 1.

        self.rho_S = sample_gamma(self.gam / (S + K + C + C), 1. / self.zeta, size=S)
        self.nu_K = sample_gamma(self.gam / (S + K + C + C), 1. / self.zeta, size=K)
        self.eta_d_C = sample_gamma(self.gam / (S + K + C + C), 1. / self.zeta, size=C)
        self.eta_a_C = sample_gamma(self.gam / (S + K + C + C), 1. / self.zeta, size=C)

        self.d = 1.
        shp_SKCC = np.ones((S, K, C, C))
        shp_SKCC[:] = np.outer(self.eta_d_C, self.eta_d_C)
        shp_SKCC[:, :, np.identity(C).astype(bool)] = self.eta_a_C * self.eta_d_C
        shp_SKCC *= self.nu_K[None, :, None, None]
        shp_SKCC *= self.rho_S[:, None, None, None]
        self.Lambda_SKCC = sample_gamma(shp_SKCC, 1. / self.d)
        self.Psi_TS = sample_gamma(self.e, 1. / self.f, size=(T, S))
        self.Phi_AK = np.ones((A, K))
        self.Phi_AK[:, :] = rn.dirichlet(self.e * np.ones(A), size=K).T
        self.alpha_N = np.ones(N) * self.e
        self.beta = 1.
        self.Theta_NC = np.ones((N, C))

    def _update(self, data, mask=None):
        vals_P = data.vals.astype(np.uint32)
        subs_P4 = np.asarray(zip(*data.subs), dtype=np.uint32)
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
        eta_d_C = self.eta_d_C
        eta_a_C = self.eta_a_C
        nu_K = self.nu_K
        rho_S = self.rho_S
        alpha_N = self.alpha_N
        beta = self.beta
        zeta = self.zeta
        delta = self.delta

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
        L_K = np.zeros(K, dtype=np.uint32)
        L_S = np.zeros(S, dtype=np.uint32)
        H_N = np.zeros(N, dtype=np.uint32)

        # Masks for treating diagonals
        int_diag_CC = np.identity(C)
        bool_diag_CC = int_diag_CC.astype(bool)

        if mask is None:
            mask = np.abs(1 - np.identity(N).astype(int))

        tmp_SCC = np.zeros((S, C, C))
        if mask.ndim == 2:
            mask_NN = mask
            tmp_SCC[:] = np.dot(Theta_NC.T, np.dot(mask_NN, Theta_NC))
            tmp_SCC *= Psi_TS.sum(axis=0)[:, None, None]
        else:
            mask_TNN = mask
            tmp_TNC = np.einsum('tij,jd->tid', mask_TNN, Theta_NC)
            tmp_TCC = np.einsum('tid,ic->tcd', tmp_TNC, Theta_NC)
            tmp_SCC[:] = np.einsum('tcd,ts->scd', tmp_TCC, Psi_TS)

        Lambda_SCC = Lambda_SKCC.sum(axis=1)

        shp_SKCC = np.ones((S, K, C, C))
        shp_SKCC[:] = np.outer(eta_d_C, eta_d_C)
        shp_SKCC[:, :, bool_diag_CC] = eta_a_C * eta_d_C
        shp_SKCC *= nu_K[None, :, None, None]
        shp_SKCC *= rho_S[:, None, None, None]

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
                start = time.time()
                comp_allocate(vals_P=vals_P,
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

            if schedule['Lambda_SKCC'] <= self.total_iter:
                start = time.time()

                shp_SKCC[:] = np.outer(eta_d_C, eta_d_C)
                shp_SKCC[:, :, bool_diag_CC] = eta_a_C * eta_d_C
                shp_SKCC *= nu_K[None, :, None, None]
                shp_SKCC *= rho_S[:, None, None, None]
                post_shp_SKCC = shp_SKCC + Y_SKCC

                if mask.ndim == 2:
                    mask_NN = mask
                    tmp_SCC[:] = np.dot(Theta_NC.T, np.dot(mask_NN, Theta_NC))
                    tmp_SCC *= Psi_TS.sum(axis=0)[:, None, None]
                else:
                    mask_TNN = mask
                    tmp_TNC = np.einsum('tij,jd->tid', mask_TNN, Theta_NC)
                    tmp_TCC = np.einsum('tid,ic->tcd', tmp_TNC, Theta_NC)
                    tmp_SCC = np.einsum('tcd,ts->scd', tmp_TCC, Psi_TS)
                post_rte_SKCC = delta + tmp_SCC[:, None, :, :]

                Lambda_SKCC[:] = sample_gamma(post_shp_SKCC, 1. / post_rte_SKCC)
                end = time.time() - start
                if self.verbose:
                    print '%f: sampling lambda' % end

            if schedule['Psi_TS'] <= self.total_iter:
                start = time.time()

                Lambda_SCC[:] = Lambda_SKCC.sum(axis=1)
                if mask.ndim == 2:
                    mask_NN = mask
                    tmp_SCC[:] = np.dot(Theta_NC.T, np.dot(mask_NN, Theta_NC))
                    tmp_TS = (tmp_SCC * Lambda_SCC).sum(axis=(1, 2)).reshape((1, S))
                else:
                    mask_TNN = mask
                    tmp_TNC = np.einsum('tij,jd->tid', mask_TNN, Theta_NC)
                    tmp_TCC = np.einsum('tid,ic->tcd', tmp_TNC, Theta_NC)
                    tmp_TS = np.einsum('tcd,scd->ts', tmp_TCC, Lambda_SCC)

                post_shp_TS = e + Y_TS
                post_rte_TS = f + tmp_TS
                Psi_TS[:] = sample_gamma(post_shp_TS, 1. / post_rte_TS)

                end = time.time() - start
                if self.verbose:
                    print '%f: sampling psi' % end

            if schedule['Theta_NC'] <= self.total_iter:
                start = time.time()
                Lambda_SCC[:] = Lambda_SKCC.sum(axis=1)
                Psi_S = Psi_TS.sum(axis=0)

                if mask.ndim == 2:
                    tmp_CC = (Lambda_SCC * Psi_S[:, None, None]).sum(axis=0)

                    tmp_s_NC = np.dot(tmp_CC, np.dot(mask_NN, Theta_NC).T).T
                    tmp_r_NC = np.dot(np.dot(mask_NN.T, Theta_NC), tmp_CC)
                else:
                    tmp_TCC = np.einsum('scd,ts->tcd', Lambda_SCC, Psi_TS)
                    tmp_s_TCN = np.einsum('tij,jd->tid', mask_TNN, Theta_NC)
                    tmp_r_TCN = np.einsum('tij,ic->tcj', mask_TNN, Theta_NC)

                    tmp_s_NC = np.einsum('tcd,tid->ci', tmp_TCC, tmp_s_TCN).T
                    tmp_r_NC = np.einsum('tcd,tcj->dj', tmp_TCC, tmp_r_TCN).T
                tmp_NC = tmp_s_NC + tmp_r_NC

                post_shp_NC = alpha_N[:, None] + Y_s_NC + Y_r_NC
                post_rte_NC = beta + tmp_NC

                Theta_NC[:, :] = sample_gamma(post_shp_NC, 1. / post_rte_NC)

                if mask.ndim == 2:
                    tmp_CC = np.einsum('scd,s->cd', Lambda_SCC, Psi_S)

                    tmp_s_NC = np.dot(tmp_CC, np.dot(mask_NN, Theta_NC).T).T
                    tmp_r_NC = np.dot(np.dot(mask_NN.T, Theta_NC), tmp_CC)
                else:
                    tmp_TCC = np.einsum('scd,ts->tcd', Lambda_SCC, Psi_TS)
                    tmp_s_TCN = np.einsum('tij,jd->tid', mask_TNN, Theta_NC)
                    tmp_r_TCN = np.einsum('tij,ic->tcj', mask_TNN, Theta_NC)

                    tmp_s_NC = np.einsum('tcd,tid->ci', tmp_TCC, tmp_s_TCN).T
                    tmp_r_NC = np.einsum('tcd,tcj->dj', tmp_TCC, tmp_r_TCN).T
                tmp_NC = tmp_s_NC + tmp_r_NC

                H_N[:] = 0
                for (i, c) in np.ndindex((N, C)):
                    H_N[i] += crt(Y_s_NC[i, c] + Y_r_NC[i, c], alpha_N[i])
                post_shp_N = e + H_N
                post_rte_N = f + np.log1p(tmp_NC / beta).sum(axis=1)
                alpha_N[:] = sample_gamma(post_shp_N, 1. / post_rte_N)

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

            if any(schedule[s] <= self.total_iter for s in ['eta_a_C', 'eta_d_C']):
                start = time.time()
                w = nu_K.sum()
                Y_SCC = Y_SKCC.sum(axis=1)

                if mask.ndim == 2:
                    tmp_SCC[:] = np.dot(Theta_NC.T, np.dot(mask_NN, Theta_NC))
                    tmp_SCC *= Psi_TS.sum(axis=0)[:, None, None]
                else:
                    tmp_TNC = np.einsum('tij,jd->tid', mask_TNN, Theta_NC)
                    tmp_TCC = np.einsum('tid,ic->tcd', tmp_TNC, Theta_NC)
                    tmp_SCC[:] = np.einsum('tcd,ts->scd', tmp_TCC, Psi_TS)
                tmp_SCC[:] = np.log1p(tmp_SCC / delta)
                tmp_CC = w * (rho_S[:, None, None] * tmp_SCC).sum(axis=0)

                for c in xrange(C):
                    Y_2S[:S] = Y_SCC[:, c, c]
                    m_a = sumcrt(Y_2S[:S], rho_S * w * eta_d_C[c] * eta_a_C[c], num_threads=1)
                    tmp_a = eta_d_C[c] * tmp_CC[c, c]
                    eta_a_C[c] = sample_gamma(gam / (S + K + C + C) + m_a, 1. / (zeta + tmp_a))

                    m_d = sumcrt(Y_2S[:S], rho_S * w * eta_d_C[c] * eta_a_C[c], num_threads=1)
                    tmp_d = eta_a_C[c] * tmp_CC[c, c]
                    for c2 in xrange(C):
                        if c == c2:
                            continue
                        Y_2S[:S] = Y_SCC[:, c, c2]
                        Y_2S[S:] = Y_SCC[:, c2, c]
                        tmp_2S[:S] = rho_S * w * eta_d_C[c] * eta_d_C[c2]
                        tmp_2S[S:] = tmp_2S[:S]
                        m_d += sumcrt(Y_2S, tmp_2S, num_threads=1)
                        tmp_d += eta_d_C[c2] * (tmp_CC[c, c2] + tmp_CC[c2, c])
                    eta_d_C[c] = sample_gamma(gam / (S + K + C + C) + m_d, 1. / (zeta + tmp_d))

                end = time.time() - start
                if self.verbose:
                    print '%f: sampling W_C' % end

            if schedule['nu_K'] <= self.total_iter:
                start = time.time()
                shp_SCC = np.zeros((S, C, C))
                shp_SCC[:] = np.outer(eta_d_C, eta_d_C)
                shp_SCC[:, bool_diag_CC] = eta_a_C * eta_d_C
                shp_SCC *= rho_S[:, None, None]
                shp_ = shp_SCC.ravel()

                for k in xrange(K):
                    L_K[k] = sumcrt(Y_SKCC[:, k].ravel(), shp_ * nu_K[k], num_threads=1)

                if mask.ndim == 2:
                    tmp_SCC[:] = np.dot(Theta_NC.T, np.dot(mask_NN, Theta_NC))
                    tmp_SCC *= Psi_TS.sum(axis=0)[:, None, None]
                else:
                    tmp_TNC = np.einsum('tij,jd->tid', mask_TNN, Theta_NC)
                    tmp_TCC = np.einsum('tid,ic->tcd', tmp_TNC, Theta_NC)
                    tmp_SCC[:] = np.einsum('tcd,ts->scd', tmp_TCC, Psi_TS)
                tmp = (shp_SCC * np.log1p(tmp_SCC / delta)).sum()

                post_shp_K = gam / (S + K + C + C) + L_K
                post_rte = zeta + tmp

                nu_K[:] = sample_gamma(post_shp_K, 1. / post_rte)

                end = time.time() - start
                if self.verbose:
                    print '%f: sampling nu_K' % end

            if schedule['rho_S'] <= self.total_iter:
                start = time.time()
                shp_KCC = np.zeros((K, C, C))
                shp_KCC[:] = np.outer(eta_d_C, eta_d_C)
                shp_KCC[:, bool_diag_CC] = eta_a_C * eta_d_C
                shp_KCC *= nu_K[:, None, None]
                shp_CC = shp_KCC.sum(axis=0)
                shp_ = shp_KCC.ravel()

                for s in xrange(S):
                    L_S[s] = sumcrt(Y_SKCC[s].ravel(), shp_ * rho_S[s], num_threads=1)

                if mask.ndim == 2:
                    tmp_SCC[:] = np.dot(Theta_NC.T, np.dot(mask_NN, Theta_NC))
                    tmp_SCC *= Psi_TS.sum(axis=0)[:, None, None]
                else:
                    tmp_TNC = np.einsum('tij,jd->tid', mask_TNN, Theta_NC)
                    tmp_TCC = np.einsum('tid,ic->tcd', tmp_TNC, Theta_NC)
                    tmp_SCC[:] = np.einsum('tcd,ts->scd', tmp_TCC, Psi_TS)
                tmp_S = (shp_CC * np.log1p(tmp_SCC / delta)).sum(axis=(1, 2))

                post_shp_S = gam / (S + K + C + C) + L_S
                post_rte_S = zeta + tmp_S

                rho_S[:] = sample_gamma(post_shp_S, 1. / post_rte_S)

                end = time.time() - start
                if self.verbose:
                    print '%f: sampling rho_S' % end

            if schedule['zeta'] <= self.total_iter:
                start = time.time()
                post_shp = e + gam
                post_rte = f + rho_S.sum() + nu_K.sum() + eta_a_C.sum() + eta_d_C.sum()
                self.zeta = zeta = sample_gamma(post_shp, 1. / post_rte)
                end = time.time() - start
                if self.verbose:
                    print '%f: sampling zeta' % end

            if schedule['beta'] <= self.total_iter:
                start = time.time()
                post_shp = e + C * alpha_N.sum()
                post_rte = f + Theta_NC.sum()
                beta = self.beta = sample_gamma(post_shp, 1. / post_rte)
                end = time.time() - start
                if self.verbose:
                    print '%f: sampling b' % end

            if schedule['delta'] <= self.total_iter:
                start = time.time()
                shp_CC = np.outer(eta_d_C, eta_d_C)
                shp_CC[bool_diag_CC] = eta_a_C * eta_d_C
                post_shp = e + rho_S.sum() * nu_K.sum() * shp_CC.sum()
                post_rte = f + Lambda_SKCC.sum()
                delta = self.delta = sample_gamma(post_shp, 1. / post_rte)
                end = time.time() - start
                if self.verbose:
                    print '%f: sampling d' % end

            if self.verbose:
                end = time.time() - total_start
                print 'ITERATION %d:\t\
                       Time %f:'\
                       % (self.total_iter, end)
            self.total_iter += 1
