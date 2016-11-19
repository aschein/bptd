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


STATE_VARS = ['Lambda_RKCC',
              'Theta_VC',
              'Phi_AK',
              'Psi_TR',
              'eta_d_C',
              'eta_a_C',
              'nu_K',
              'rho_R',
              'alpha_V',
              'beta',
              'delta',
              'zeta']


class BPTD(BaseEstimator):
    """Bayesian Poisson Tucker decomposition"""
    def __init__(self, n_regimes=3, n_communities=25, n_topics=5, e=0.1, f=0.1, gam=None,
                 n_iter=1000, schedule={}, verbose=True, n_threads=1, eps=1e-300):
        self.n_regimes = n_regimes
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
        state['Y_RKCC'] = self.Y_RKCC.copy()
        return state

    def set_state(self, state):
        # assert all(s in STATE_VARS for s in state.keys())
        for s in STATE_VARS:
            assert s in state.keys()
        V, C = state['Theta_VC'].shape
        A, K = state['Phi_AK'].shape
        T, R = state['Psi_TR'].shape
        self.n_actors = V
        self.n_actions = A
        self.n_timesteps = T
        self.n_regimes = R
        self.n_communities = C
        self.n_topics = K
        for s in state.keys():
            setattr(self, s, deepcopy(state[s]))

    def reconstruct(self, partial_state={}, subs=None):
        Lambda_RKCC = self.Lambda_RKCC
        if 'Lambda_RKCC' in partial_state.keys():
            Lambda_RKCC = partial_state['Lambda_RKCC']

        Theta_VC = self.Theta_VC
        if 'Theta_VC' in partial_state.keys():
            Theta_VC = partial_state['Theta_VC']

        Phi_AK = self.Phi_AK
        if 'Phi_AK' in partial_state.keys():
            Phi_AK = partial_state['Phi_AK']

        Psi_TR = self.Psi_TR
        if 'Psi_TR' in partial_state.keys():
            Psi_TR = partial_state['Psi_TR']

        assert Lambda_RKCC.shape[0] == Psi_TR.shape[1]
        assert Lambda_RKCC.shape[1] == Phi_AK.shape[1]
        assert Lambda_RKCC.shape[2] == Theta_VC.shape[1]
        assert Lambda_RKCC.shape[3] == Theta_VC.shape[1]

        V = Theta_VC.shape[0]
        Lambda_CCKR = np.transpose(Lambda_RKCC, (2, 3, 1, 0))
        rates_CCKT = np.einsum('cdkr,tr->cdkt', Lambda_CCKR, Psi_TR)
        rates_CCAT = np.einsum('cdkt,ak->cdat', rates_CCKT, Phi_AK)
        rates_CVAT = np.einsum('cdat,jd->cjat', rates_CCAT, Theta_VC)
        rates_VVAT = np.einsum('cjat,ic->ijat', rates_CVAT, Theta_VC)
        rates_VVAT[np.identity(V).astype(bool)] = 0

        if subs is not None:
            return rates_VVAT[subs]
        return rates_VVAT

    def _check_params(self):
        V = self.n_actors
        A = self.n_actions
        T = self.n_timesteps
        R = self.n_regimes
        K = self.n_topics
        C = self.n_communities
        assert self.Lambda_RKCC.shape == (R, K, C, C)
        assert self.Phi_AK.shape == (A, K)
        assert self.Theta_VC.shape == (V, C)
        assert self.Psi_TR.shape == (T, R)
        assert self.eta_a_C.shape == (C,)
        assert self.eta_d_C.shape == (C,)
        assert self.nu_K.shape == (K,)
        assert self.rho_R.shape == (R,)
        assert self.alpha_V.shape == (V,)
        for key in STATE_VARS:
            if hasattr(self, key):
                assert np.isfinite(getattr(self, key)).all()

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
        V, A, T = data.shape[1:]
        self.n_actors = V
        self.n_actions = A
        self.n_timesteps = T

        if mask is not None:
            assert isinstance(mask, np.ndarray)
            assert (mask.ndim == 2) or (mask.ndim == 3)
            assert mask.shape[-2:] == (V, V)
            assert np.issubdtype(mask.dtype, np.integer)

        return data

    def _init_latent_params(self):
        V = self.n_actors
        A = self.n_actions
        T = self.n_timesteps
        R = self.n_regimes
        C = self.n_communities
        K = self.n_topics

        if self.gam is None:
            self.gam = (0.1 ** (1. / 4)) * (R + K + C + C)
            print 'Setting gam to: %f' % self.gam
        self.zeta = 1.
        self.delta = 1.

        self.rho_R = sample_gamma(self.gam / (R + K + C + C), 1. / self.zeta, size=R)
        self.nu_K = sample_gamma(self.gam / (R + K + C + C), 1. / self.zeta, size=K)
        self.eta_d_C = sample_gamma(self.gam / (R + K + C + C), 1. / self.zeta, size=C)
        self.eta_a_C = sample_gamma(self.gam / (R + K + C + C), 1. / self.zeta, size=C)

        self.d = 1.
        shp_RKCC = np.ones((R, K, C, C))
        shp_RKCC[:] = np.outer(self.eta_d_C, self.eta_d_C)
        shp_RKCC[:, :, np.identity(C).astype(bool)] = self.eta_a_C * self.eta_d_C
        shp_RKCC *= self.nu_K[None, :, None, None]
        shp_RKCC *= self.rho_R[:, None, None, None]
        self.Lambda_RKCC = sample_gamma(shp_RKCC, 1. / self.d)
        self.Psi_TR = sample_gamma(self.e, 1. / self.f, size=(T, R))
        self.Phi_AK = np.ones((A, K))
        self.Phi_AK[:, :] = rn.dirichlet(self.e * np.ones(A), size=K).T
        self.alpha_V = np.ones(V) * self.e
        self.beta = 1.
        self.Theta_VC = np.ones((V, C))

    def _update(self, data, mask=None):
        vals_P = data.vals.astype(np.uint32)
        subs_P4 = np.asarray(zip(*data.subs), dtype=np.uint32)
        V = self.n_actors
        A = self.n_actions
        T = self.n_timesteps
        R = self.n_regimes
        C = self.n_communities
        K = self.n_topics

        Lambda_RKCC = self.Lambda_RKCC
        Theta_VC = self.Theta_VC
        Phi_AK = self.Phi_AK
        Psi_TR = self.Psi_TR
        eta_d_C = self.eta_d_C
        eta_a_C = self.eta_a_C
        nu_K = self.nu_K
        rho_R = self.rho_R
        alpha_V = self.alpha_V
        beta = self.beta
        zeta = self.zeta
        delta = self.delta

        # Hyperparameters
        if self.gam is None:
            self.gam = (0.1 ** (1. / 4)) * (R + K + C + C)
            print 'Setting gam to: %f' % self.gam
        gam = self.gam
        e = self.e
        f = self.f
        eta_A = np.ones(A) * e

        Y_s_VC = self.Y_s_VC = np.zeros((V, C), np.uint32)
        Y_r_VC = self.Y_r_VC = np.zeros((V, C), np.uint32)
        Y_AK = self.Y_AK = np.zeros((A, K), np.uint32)
        Y_TR = self.Y_TR = np.zeros((T, R), np.uint32)
        Y_RKCC = self.Y_RKCC = np.zeros((R, K, C, C), np.uint32)

        Y_2R = np.ones(2 * R, dtype=np.uint32)
        tmp_2R = np.ones(2 * R)

        # Latent CRT sources
        L_K = np.zeros(K, dtype=np.uint32)
        L_R = np.zeros(R, dtype=np.uint32)
        H_V = np.zeros(V, dtype=np.uint32)

        # Masks for treating diagonals
        int_diag_CC = np.identity(C)
        bool_diag_CC = int_diag_CC.astype(bool)

        if mask is None:
            mask = np.abs(1 - np.identity(V).astype(int))

        tmp_RCC = np.zeros((R, C, C))
        if mask.ndim == 2:
            mask_VV = mask
            tmp_RCC[:] = np.dot(Theta_VC.T, np.dot(mask_VV, Theta_VC))
            tmp_RCC *= Psi_TR.sum(axis=0)[:, None, None]
        else:
            mask_TVV = mask
            tmp_TVC = np.einsum('tij,jd->tid', mask_TVV, Theta_VC)
            tmp_TCC = np.einsum('tid,ic->tcd', tmp_TVC, Theta_VC)
            tmp_RCC[:] = np.einsum('tcd,ts->scd', tmp_TCC, Psi_TR)

        Lambda_RCC = Lambda_RKCC.sum(axis=1)

        shp_RKCC = np.ones((R, K, C, C))
        shp_RKCC[:] = np.outer(eta_d_C, eta_d_C)
        shp_RKCC[:, :, bool_diag_CC] = eta_a_C * eta_d_C
        shp_RKCC *= nu_K[None, :, None, None]
        shp_RKCC *= rho_R[:, None, None, None]

        schedule = self.schedule.copy()
        for k, v in schedule.items():
            if v is None:
                schedule[k] = np.inf

        if self.verbose:
            outstr = 'Starting' if self.total_iter == 0 else 'Restarting'
            print '%s inference...' % outstr

        for itn in xrange(self.n_iter):
            total_start = time.time()

            if schedule['Sources'] <= self.total_iter:
                start = time.time()
                comp_allocate(vals_P=vals_P,
                              subs_P4=subs_P4,
                              Theta_s_VC=Theta_VC,
                              Theta_r_VC=Theta_VC,
                              Phi_AK=Phi_AK,
                              Psi_TR=Psi_TR,
                              Lambda_RKCC=Lambda_RKCC,
                              Y_s_VC=Y_s_VC,
                              Y_r_VC=Y_r_VC,
                              Y_AK=Y_AK,
                              Y_TR=Y_TR,
                              Y_RKCC=Y_RKCC,
                              num_threads=self.n_threads)
                end = time.time() - start
                if self.verbose:
                    print '%f: sampling tokens compositionally' % end

            if schedule['Lambda_RKCC'] <= self.total_iter:
                start = time.time()

                shp_RKCC[:] = np.outer(eta_d_C, eta_d_C)
                shp_RKCC[:, :, bool_diag_CC] = eta_a_C * eta_d_C
                shp_RKCC *= nu_K[None, :, None, None]
                shp_RKCC *= rho_R[:, None, None, None]
                post_shp_RKCC = shp_RKCC + Y_RKCC

                if mask.ndim == 2:
                    mask_VV = mask
                    tmp_RCC[:] = np.dot(Theta_VC.T, np.dot(mask_VV, Theta_VC))
                    tmp_RCC *= Psi_TR.sum(axis=0)[:, None, None]
                else:
                    mask_TVV = mask
                    tmp_TVC = np.einsum('tij,jd->tid', mask_TVV, Theta_VC)
                    tmp_TCC = np.einsum('tid,ic->tcd', tmp_TVC, Theta_VC)
                    tmp_RCC = np.einsum('tcd,ts->scd', tmp_TCC, Psi_TR)
                post_rte_RKCC = delta + tmp_RCC[:, None, :, :]

                Lambda_RKCC[:] = sample_gamma(post_shp_RKCC, 1. / post_rte_RKCC)
                end = time.time() - start
                if self.verbose:
                    print '%f: sampling lambda' % end

            if schedule['Psi_TR'] <= self.total_iter:
                start = time.time()

                Lambda_RCC[:] = Lambda_RKCC.sum(axis=1)
                if mask.ndim == 2:
                    mask_VV = mask
                    tmp_RCC[:] = np.dot(Theta_VC.T, np.dot(mask_VV, Theta_VC))
                    tmp_TR = (tmp_RCC * Lambda_RCC).sum(axis=(1, 2)).reshape((1, R))
                else:
                    mask_TVV = mask
                    tmp_TVC = np.einsum('tij,jd->tid', mask_TVV, Theta_VC)
                    tmp_TCC = np.einsum('tid,ic->tcd', tmp_TVC, Theta_VC)
                    tmp_TR = np.einsum('tcd,rcd->tr', tmp_TCC, Lambda_RCC)

                post_shp_TR = e + Y_TR
                post_rte_TR = f + tmp_TR
                Psi_TR[:] = sample_gamma(post_shp_TR, 1. / post_rte_TR)

                end = time.time() - start
                if self.verbose:
                    print '%f: sampling psi' % end

            if schedule['Theta_VC'] <= self.total_iter:
                start = time.time()
                Lambda_RCC[:] = Lambda_RKCC.sum(axis=1)
                Psi_R = Psi_TR.sum(axis=0)

                if mask.ndim == 2:
                    tmp_CC = (Lambda_RCC * Psi_R[:, None, None]).sum(axis=0)

                    tmp_s_VC = np.dot(tmp_CC, np.dot(mask_VV, Theta_VC).T).T
                    tmp_r_VC = np.dot(np.dot(mask_VV.T, Theta_VC), tmp_CC)
                else:
                    tmp_TCC = np.einsum('rcd,tr->tcd', Lambda_RCC, Psi_TR)
                    tmp_s_TCV = np.einsum('tij,jd->tid', mask_TVV, Theta_VC)
                    tmp_r_TCV = np.einsum('tij,ic->tcj', mask_TVV, Theta_VC)

                    tmp_s_VC = np.einsum('tcd,tid->ci', tmp_TCC, tmp_s_TCV).T
                    tmp_r_VC = np.einsum('tcd,tcj->dj', tmp_TCC, tmp_r_TCV).T
                tmp_VC = tmp_s_VC + tmp_r_VC

                post_shp_VC = alpha_V[:, None] + Y_s_VC + Y_r_VC
                post_rte_VC = beta + tmp_VC

                Theta_VC[:, :] = sample_gamma(post_shp_VC, 1. / post_rte_VC)

                if mask.ndim == 2:
                    tmp_CC = np.einsum('rcd,r->cd', Lambda_RCC, Psi_R)

                    tmp_s_VC = np.dot(tmp_CC, np.dot(mask_VV, Theta_VC).T).T
                    tmp_r_VC = np.dot(np.dot(mask_VV.T, Theta_VC), tmp_CC)
                else:
                    tmp_TCC = np.einsum('rcd,tr->tcd', Lambda_RCC, Psi_TR)
                    tmp_s_TCV = np.einsum('tij,jd->tid', mask_TVV, Theta_VC)
                    tmp_r_TCV = np.einsum('tij,ic->tcj', mask_TVV, Theta_VC)

                    tmp_s_VC = np.einsum('tcd,tid->ci', tmp_TCC, tmp_s_TCV).T
                    tmp_r_VC = np.einsum('tcd,tcj->dj', tmp_TCC, tmp_r_TCV).T
                tmp_VC = tmp_s_VC + tmp_r_VC

                H_V[:] = 0
                for (i, c) in np.ndindex((V, C)):
                    H_V[i] += crt(Y_s_VC[i, c] + Y_r_VC[i, c], alpha_V[i])
                post_shp_V = e + H_V
                post_rte_V = f + np.log1p(tmp_VC / beta).sum(axis=1)
                alpha_V[:] = sample_gamma(post_shp_V, 1. / post_rte_V)

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
                Y_RCC = Y_RKCC.sum(axis=1)

                if mask.ndim == 2:
                    tmp_RCC[:] = np.dot(Theta_VC.T, np.dot(mask_VV, Theta_VC))
                    tmp_RCC *= Psi_TR.sum(axis=0)[:, None, None]
                else:
                    tmp_TVC = np.einsum('tij,jd->tid', mask_TVV, Theta_VC)
                    tmp_TCC = np.einsum('tid,ic->tcd', tmp_TVC, Theta_VC)
                    tmp_RCC[:] = np.einsum('tcd,tr->rcd', tmp_TCC, Psi_TR)
                tmp_RCC[:] = np.log1p(tmp_RCC / delta)
                tmp_CC = w * (rho_R[:, None, None] * tmp_RCC).sum(axis=0)

                for c in xrange(C):
                    Y_2R[:R] = Y_RCC[:, c, c]
                    m_a = sumcrt(Y_2R[:R], rho_R * w * eta_d_C[c] * eta_a_C[c], num_threads=1)
                    tmp_a = eta_d_C[c] * tmp_CC[c, c]
                    eta_a_C[c] = sample_gamma(gam / (R + K + C + C) + m_a, 1. / (zeta + tmp_a))

                    m_d = sumcrt(Y_2R[:R], rho_R * w * eta_d_C[c] * eta_a_C[c], num_threads=1)
                    tmp_d = eta_a_C[c] * tmp_CC[c, c]
                    for c2 in xrange(C):
                        if c == c2:
                            continue
                        Y_2R[:R] = Y_RCC[:, c, c2]
                        Y_2R[R:] = Y_RCC[:, c2, c]
                        tmp_2R[:R] = rho_R * w * eta_d_C[c] * eta_d_C[c2]
                        tmp_2R[R:] = tmp_2R[:R]
                        m_d += sumcrt(Y_2R, tmp_2R, num_threads=1)
                        tmp_d += eta_d_C[c2] * (tmp_CC[c, c2] + tmp_CC[c2, c])
                    eta_d_C[c] = sample_gamma(gam / (R + K + C + C) + m_d, 1. / (zeta + tmp_d))

                end = time.time() - start
                if self.verbose:
                    print '%f: sampling W_C' % end

            if schedule['nu_K'] <= self.total_iter:
                start = time.time()
                shp_RCC = np.zeros((R, C, C))
                shp_RCC[:] = np.outer(eta_d_C, eta_d_C)
                shp_RCC[:, bool_diag_CC] = eta_a_C * eta_d_C
                shp_RCC *= rho_R[:, None, None]
                shp_ = shp_RCC.ravel()

                for k in xrange(K):
                    L_K[k] = sumcrt(Y_RKCC[:, k].ravel(), shp_ * nu_K[k], num_threads=1)

                if mask.ndim == 2:
                    tmp_RCC[:] = np.dot(Theta_VC.T, np.dot(mask_VV, Theta_VC))
                    tmp_RCC *= Psi_TR.sum(axis=0)[:, None, None]
                else:
                    tmp_TVC = np.einsum('tij,jd->tid', mask_TVV, Theta_VC)
                    tmp_TCC = np.einsum('tid,ic->tcd', tmp_TVC, Theta_VC)
                    tmp_RCC[:] = np.einsum('tcd,tr->rcd', tmp_TCC, Psi_TR)
                tmp = (shp_RCC * np.log1p(tmp_RCC / delta)).sum()

                post_shp_K = gam / (R + K + C + C) + L_K
                post_rte = zeta + tmp

                nu_K[:] = sample_gamma(post_shp_K, 1. / post_rte)

                end = time.time() - start
                if self.verbose:
                    print '%f: sampling nu_K' % end

            if schedule['rho_R'] <= self.total_iter:
                start = time.time()
                shp_KCC = np.zeros((K, C, C))
                shp_KCC[:] = np.outer(eta_d_C, eta_d_C)
                shp_KCC[:, bool_diag_CC] = eta_a_C * eta_d_C
                shp_KCC *= nu_K[:, None, None]
                shp_CC = shp_KCC.sum(axis=0)
                shp_ = shp_KCC.ravel()

                for r in xrange(R):
                    L_R[r] = sumcrt(Y_RKCC[r].ravel(), shp_ * rho_R[r], num_threads=1)

                if mask.ndim == 2:
                    tmp_RCC[:] = np.dot(Theta_VC.T, np.dot(mask_VV, Theta_VC))
                    tmp_RCC *= Psi_TR.sum(axis=0)[:, None, None]
                else:
                    tmp_TVC = np.einsum('tij,jd->tid', mask_TVV, Theta_VC)
                    tmp_TCC = np.einsum('tid,ic->tcd', tmp_TVC, Theta_VC)
                    tmp_RCC[:] = np.einsum('tcd,tr->rcd', tmp_TCC, Psi_TR)
                tmp_R = (shp_CC * np.log1p(tmp_RCC / delta)).sum(axis=(1, 2))

                post_shp_R = gam / (R + K + C + C) + L_R
                post_rte_R = zeta + tmp_R

                rho_R[:] = sample_gamma(post_shp_R, 1. / post_rte_R)

                end = time.time() - start
                if self.verbose:
                    print '%f: sampling rho_R' % end

            if schedule['zeta'] <= self.total_iter:
                start = time.time()
                post_shp = e + gam
                post_rte = f + rho_R.sum() + nu_K.sum() + eta_a_C.sum() + eta_d_C.sum()
                self.zeta = zeta = sample_gamma(post_shp, 1. / post_rte)
                end = time.time() - start
                if self.verbose:
                    print '%f: sampling zeta' % end

            if schedule['beta'] <= self.total_iter:
                start = time.time()
                post_shp = e + C * alpha_V.sum()
                post_rte = f + Theta_VC.sum()
                beta = self.beta = sample_gamma(post_shp, 1. / post_rte)
                end = time.time() - start
                if self.verbose:
                    print '%f: sampling b' % end

            if schedule['delta'] <= self.total_iter:
                start = time.time()
                shp_CC = np.outer(eta_d_C, eta_d_C)
                shp_CC[bool_diag_CC] = eta_a_C * eta_d_C
                post_shp = e + rho_R.sum() * nu_K.sum() * shp_CC.sum()
                post_rte = f + Lambda_RKCC.sum()
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
