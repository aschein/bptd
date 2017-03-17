"""
Main plot for exploring output of topic-partitioned dyadic events.

Multiple steps are involved:
-- Ordering actors
-- Ordering communities
-- Choosing which actors to display
-- Scale to display quantities of interest
-- Misc. annotations:
    -- Community lines
    -- Color-coding actor names
    -- Overlaying math notation of matrices
"""

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True

from path import path


ACTIONS = ['Make Statement',
           'Appeal',
           'Intend to Cooperate',
           'Consult',
           'Cooperate (Diplomatic)',
           'Cooperate (Material)',
           'Aid',
           'Yield',
           'Investigate',
           'Demand',
           'Disapprove',
           'Reject',
           'Threaten',
           'Protest',
           'Posture',
           'Reduce Relations',
           'Coerce',
           'Assault',
           'Fight',
           'Mass Violence']


def hard_cluster_actors(Theta_CN):
    assignments_N = Theta_CN.argmax(axis=0)
    confidences_N = (Theta_CN / Theta_CN.sum(axis=0)).max(axis=0)
    return assignments_N, confidences_N


def order_actors(assignments_N, confidences_N, order_C):
    order_N = []
    for c in order_C:
        actors = np.where(assignments_N == c)[0]
        order_N += sorted(actors, key=lambda x: confidences_N[x], reverse=True)
    return order_N


def plot_theta(Theta_NC, labels, scale_func=lambda x: np.log(x + 1), filename=None, figsize=None, dpi=None):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    sns.heatmap(scale_func(Theta_NC), ax=ax, vmin=0, cmap='Blues',
                xticklabels=range(1, Theta_NC.shape[1]+1), yticklabels=labels, cbar=False)
    plt.setp(ax.get_yticklabels(), fontsize=5, rotation=0, weight='bold')
    plt.setp(ax.get_xticklabels(), fontsize=5, rotation=0, weight='bold')
    if filename is not None:
        plt.savefig(filename, format='pdf', bbox_inches='tight')
    else:
        plt.show()


def plot_component(Y_NN, Lambda_CC, s_Theta_CN, r_Theta_CN, assignments_N,
                   scale_func=lambda x: np.log(x + 1), filename=None, figsize=None, dpi=None):
    plt.figure(figsize=figsize, dpi=dpi)

    fontsize = 8
    height_ratios = [4, 1]
    width_ratios = [1, 4]
    N = Y_NN.shape[0]

    gs = gridspec.GridSpec(2, 2, height_ratios=height_ratios, width_ratios=width_ratios)
    gs.update(wspace=0.025, hspace=0.025)
    ax1 = plt.subplot(gs[1, 0])  # Lambda
    ax2 = plt.subplot(gs[0, 0])  # s_Theta
    ax3 = plt.subplot(gs[1, 1])  # r_Theta
    ax4 = plt.subplot(gs[0, 1])  # Y

    sns.heatmap(scale_func(Lambda_CC), vmin=0, cmap='Reds', ax=ax1, cbar=False,
                xticklabels=range(1, C + 1), yticklabels=range(1, C + 1))
    plt.setp(ax1.get_yticklabels(), fontsize=fontsize, weight='bold')
    plt.setp(ax1.get_xticklabels(), fontsize=fontsize, weight='bold')

    sns.heatmap(scale_func(s_Theta_CN.T), ax=ax2, vmin=0, cmap='Blues',
                yticklabels=actors[order_N], cbar=False)
    plt.setp(ax2.get_yticklabels(), fontsize=fontsize, rotation=0, weight='bold')
    ax2.set_xticklabels([])

    sns.heatmap(scale_func(r_Theta_CN), ax=ax3, vmin=0, cmap='Blues',
                xticklabels=actors[order_N], cbar=False)
    plt.setp(ax3.get_xticklabels(), fontsize=fontsize, rotation=90, weight='bold')
    ax3.set_yticklabels([])

    sns.heatmap(scale_func(Y_NN), ax=ax4, vmin=0, cmap='Reds', cbar=False)

    N = assignments_N.size
    last_assignment = assignments_N[0]
    for i, assignment in enumerate(assignments_N):
        if assignment != last_assignment:
            ax4.axvline(i, c='g', lw=2.)
            ax4.axhline(N - i, c='g', lw=2.)
            ax2.axhline(N - i, c='g', lw=2.)
            ax3.axvline(i, c='g', lw=2.)
            last_assignment = assignment
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])

    if filename is not None:
        plt.savefig(filename, format='pdf', bbox_inches='tight')
    else:
        plt.show()

if __name__ == '__main__':
    data = np.load('../dat/1995-2000-M.dat')
    actors = data['actors']
    actors[218] = 'Curacao'
    actors[np.where(actors == 'United States')] = 'USA'
    actors[np.where(actors == 'United Arab Emirates')] = 'UAE'
    actors[np.where(actors == 'United Kingdom')] = 'UK'
    actors[np.where(actors == 'Czech Republic')] = 'Czech Rep.'
    actors[np.where(actors == 'Bosnia and Herzegovina')] = 'Bosnia'
    actors[np.where(actors == 'Bosnia and Herzegovina')] = 'Bosnia'
    actors[np.where(actors == 'Occupied Palestinian Territory')] = 'Palestine'
    actors[np.where(actors == 'Russian Federation')] = 'Russia'
    actors[np.where(actors == 'the former Yugoslav Republic of Macedonia')] = 'Macedonia'
    actors[np.where(actors == 'Democratic Republic of Congo')] = 'DR Congo'
    Y_NNAT = data['Y'].toarray()
    N, A, T = Y_NNAT.shape[1:]

    state = np.load('../dat/state_sample.npz')

    Theta_NC = state['Theta_NC']
    Phi_AK = state['Phi_AK']
    Psi_TR = state['Psi_TR']
    W_K = state['W_K']
    W_R = state['W_R']
    W_a_C = state['W_a_C']
    W_d_C = state['W_d_C']
    Lambda_RKCC = state['Lambda_RKCC']

    S, K, C = Lambda_RKCC.shape[:-1]
    N = Theta_NC.shape[0]
    mask_NN = np.abs(1 - np.identity(N).astype(int))

    Psi_R = Psi_TR.sum(axis=0)
    Phi_K = Phi_AK.sum(axis=0)
    Lambda_RKCC *= Psi_R[:, None, None, None] * Phi_K[None, :, None, None]
    sum_Theta_NC = np.dot(Theta_NC.T, mask_NN).T
    Theta_s_RKCN = Theta_NC.T * np.einsum('rkcd,id->rkci', Lambda_RKCC, sum_Theta_NC)
    Theta_r_RKNC = Theta_NC * np.einsum('rkcd,jc->rkjd', Lambda_RKCC, sum_Theta_NC)

    actor_indices = range(160)
    actors = actors[actor_indices]
    Y_NNAT = Y_NNAT[actor_indices][:, actor_indices]
    Y_NNAR = np.einsum('ijat,tr->ijar', Y_NNAT, Psi_TR / Psi_R)
    Y_NNKR = np.einsum('ijar,ak->ijkr', Y_NNAR, Phi_AK / Phi_K)
    Theta_s_RKCN = Theta_s_RKCN[:, :, :, actor_indices]
    Theta_r_RKNC = Theta_r_RKNC[:, :, actor_indices, :]

    out = path('../dat/figs')
    filename = out.joinpath('theta.pdf')
    out.makedirs_p()
    assignments_N, confidences_N = hard_cluster_actors(Theta_s_RKCN.sum(axis=(0, 1)))
    rel_W_a_C = W_a_C / (W_a_C + W_d_C)
    order_C = rel_W_a_C.argsort()
    global_order_N = order_actors(assignments_N, confidences_N, order_C)
    plot_theta(Theta_NC[global_order_N][:, order_C], labels=actors[global_order_N], scale_func=lambda x: np.log(x + 1), filename=filename, figsize=(2, 8), dpi=None)

    for rank_r, r in enumerate(W_R.argsort()[::-1]):
        for rank_k, k in enumerate(W_K.argsort()[::-1]):
            Theta_s_CN = Theta_s_RKCN[r, k]
            Theta_r_CN = Theta_r_RKNC[r, k].T

            assignments_N, confidences_N = hard_cluster_actors(Theta_s_RKCN.sum(axis=(0, 1)))

            rel_W_a_C = W_a_C / (W_a_C + W_d_C)
            order_C = rel_W_a_C.argsort()

            order_N = order_actors(assignments_N, confidences_N, order_C)
            Y_NN = Y_NNKR[:, :, k, r]
            Y_NN = Y_NN[order_N][:, order_N]
            Lambda_CC = Lambda_RKCC[r, k]
            Lambda_CC = Lambda_CC[order_C][:, order_C]
            Theta_s_CN = Theta_s_CN[order_C][:, order_N]
            Theta_r_CN = Theta_r_CN[order_C][:, order_N]
            assignments_N = assignments_N[order_N]

            filename = out.joinpath('component_s%d_k%d.pdf' % (rank_r+1, rank_k+1))
            plot_component(Y_NN, Lambda_CC, Theta_s_CN, Theta_r_CN, assignments_N,
                           scale_func=lambda x: np.log(x + 1), filename=filename, figsize=(20, 20), dpi=600)
            print filename
