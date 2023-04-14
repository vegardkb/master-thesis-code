import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy import signal, stats


EXP_NAME = "20220604_16_24_03_HuC_GCamp6s_GFAP_jRGECO_F3_PTZ"
DENOISED_DIR = f"Y:\\Vegard\\data\\{EXP_NAME}\\OpticTectum\\denoised"


GLIA_PATH = "results\\dff_regs.npy"
NEURON_PATH = "suite2p\\plane0\\F.npy"
NEURON_STAT_PATH = "suite2p\\plane0\\stat.npy"
NEUROPIL_PATH = "suite2p\\plane0\\Fneu.npy"
ISCELL_PATH = "suite2p\\plane0\\iscell.npy"

SUBTRACT_NEUROPIL = False
NEUROPIL_COEF = 0.7

FS = 5.15

LOW_PASS_FILTER = True
LOW_PASS_FILTER_C = 0.25
LOW_PASS_FILTER_ORDER = 4

NUM_TIMEBINS = 5

PIXEL_SIZE = 2.60417e-07


def low_pass_filter(dff, fs, f_c, order, axis=0):
    sos = signal.butter(order, f_c, btype="low", output="sos", fs=fs)
    dff_filt = signal.sosfiltfilt(sos, dff, axis=axis)
    return dff_filt


def leaky_integrator(a_init, i, tau, dt):
    num_frames = i.shape[0]
    a = np.zeros(num_frames)
    a[0] = a_init
    for t, t_prev in zip(range(1, num_frames), range(num_frames - 1)):
        a[t] = a[t_prev] + (i[t_prev] - a[t_prev]) * dt / tau

    return a


def grid_search(a, i, taus, dt):
    corrs = []
    for tau in taus:
        a_pred = leaky_integrator(a[0], i, tau, dt)
        corrs.append(stats.pearsonr(a_pred[int(tau * 2) :], a[int(tau * 2) :])[0])

    return np.array(corrs)


def calc_correlation_matrix(dff_n):
    corr_mat = np.corrcoef(dff_n, dff_n)
    return


def main():
    dff_regs = np.load(os.path.join(DENOISED_DIR, GLIA_PATH))
    f_neurons = np.load(os.path.join(DENOISED_DIR, NEURON_PATH), allow_pickle=True)
    stat_neurons = np.load(
        os.path.join(DENOISED_DIR, NEURON_STAT_PATH), allow_pickle=True
    )

    if SUBTRACT_NEUROPIL:
        f_neurons = f_neurons - np.load(
            os.path.join(DENOISED_DIR, NEUROPIL_PATH), allow_pickle=True
        )

    if LOW_PASS_FILTER:
        dff_regs = low_pass_filter(
            dff_regs, FS, LOW_PASS_FILTER_C, LOW_PASS_FILTER_ORDER
        )

    iscell = np.load(os.path.join(DENOISED_DIR, ISCELL_PATH), allow_pickle=True)

    f_neurons = f_neurons[iscell[:, 0] > 0.5]
    stat_neurons = stat_neurons[iscell[:, 0] > 0.5]
    pos_neurons = np.array(
        [stat_neurons[i]["med"] for i in range(stat_neurons.shape[0])]
    )
    f0 = np.percentile(f_neurons, 20, axis=1)
    dff_neurons = ((f_neurons.T - f0) / f0).T

    corr_mat = np.corrcoef(dff_neurons)
    plt.figure()
    plt.imshow(corr_mat, cmap=cm.coolwarm)
    plt.clim(-1, 1)
    plt.colorbar()
    plt.title("Entire experiment corr")

    num_neurons = dff_neurons.shape[0]
    corr = []
    dist = []

    for n1 in range(num_neurons):
        for n2 in range(n1):
            corr.append(corr_mat[n1, n2])
            dist.append(
                PIXEL_SIZE
                * np.sqrt(
                    (
                        np.power(pos_neurons[n1][0] - pos_neurons[n2][0], 2)
                        + np.power(pos_neurons[n1][1] - pos_neurons[n2][1], 2)
                    )
                )
            )

    corr = np.array(corr)
    dist = np.array(dist)

    pos_corr, pos_dist = corr[corr < 0], dist[corr < 0]
    neg_corr, neg_dist = corr[corr > 0], dist[corr > 0]

    for sign_corr, sign_dist in zip([pos_corr, neg_corr], [pos_dist, neg_dist]):

        min_dist, max_dist = np.amin(sign_dist), np.amax(sign_dist)

        num_bins = 100
        d_thresh = np.linspace(min_dist - 1e-6, max_dist + 1e-6, num_bins + 1)
        dist_bin = np.zeros(num_bins)
        corr_bin = np.zeros(num_bins)
        for i, t_l, t_h in zip(range(num_bins), d_thresh[:-1], d_thresh[1:]):
            dist_bin[i] = (t_l + t_h) / 2
            d_mask = np.logical_and(t_l <= sign_dist, t_h > sign_dist)
            if not np.any(d_mask):
                corr_bin[i] = np.nan
            else:
                corr_bin[i] = np.mean(sign_corr[d_mask])

        plt.figure()
        plt.plot(dist_bin * 1e6, corr_bin)
        plt.xlabel("Distance [micrometers]")

        plt.figure()
        plt.plot(np.sort(np.abs(sign_corr)), np.cumsum(sign_corr) / np.sum(sign_corr))
        plt.xlabel("Correlation")
        plt.ylabel("Ratio of cells")

    """ num_frames = dff_neurons.shape[1]
    t = np.arange(num_frames) / FS
    t_thresh = np.linspace(np.amin(t) - 1e-6, np.amax(t) + 1e-6, NUM_TIMEBINS + 1)
    t_masks = []
    for t_l, t_h in zip(t_thresh[:-1], t_thresh[1:]):
        t_mask = np.logical_and(t_l <= t, t_h > t)
        corr_mat = np.corrcoef(dff_neurons[:, t_mask])

        plt.figure()
        plt.imshow(corr_mat, cmap=cm.coolwarm)
        plt.clim(-1, 1)
        plt.colorbar()
        plt.title(f"corr t in [{t_l}, {t_h})") """

    plt.show()


if __name__ == "__main__":
    main()
