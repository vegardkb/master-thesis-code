import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy import signal, stats


EXP_NAME = "20220604_16_24_03_HuC_GCamp6s_GFAP_jRGECO_F3_PTZ"
DENOISED_DIR = f"Y:\\Vegard\\data\\{EXP_NAME}\\OpticTectum\\denoised"


GLIA_PATH = "results\\dff_regs.npy"
NEURON_PATH = "suite2p\\plane0\\F.npy"
NEUROPIL_PATH = "suite2p\\plane0\\Fneu.npy"
ISCELL_PATH = "suite2p\\plane0\\iscell.npy"

SUBTRACT_NEUROPIL = False
NEUROPIL_COEF = 0.7

FS = 5.15

LOW_PASS_FILTER = True
LOW_PASS_FILTER_C = 0.25
LOW_PASS_FILTER_ORDER = 8


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


def m1(a_init, dadt_init, n, alpha, k, d, dt):
    num_frames = n.shape[0]
    a = np.zeros(num_frames)
    dadt = np.zeros(num_frames)
    a[0] = a_init
    dadt[0] = dadt_init
    for t, t_prev in zip(range(1, num_frames), range(num_frames - 1)):
        dadt[t] = (
            dadt[t_prev] + (alpha * n[t_prev] - k * a[t_prev] - d * dadt[t_prev]) * dt
        )
        a[t] = a[t_prev] + dadt[t] * dt

    return a


def grid_search(a, i, taus, dt):
    corrs = []
    for tau in taus:
        a_pred = leaky_integrator(a[0], i, tau, dt)
        corrs.append(stats.pearsonr(a_pred[int(tau * 2) :], a[int(tau * 2) :])[0])

    return np.array(corrs)


def grid_search_m1(a, n, alphas, ks, ds, dt):
    a_init = a[0]
    dadt_init = (a[1] - a[0]) / dt

    losses = np.zeros((len(alphas), len(ks), len(ds)))
    alpha_ind = 0
    k_ind = 0
    d_ind = 0
    best_loss = np.inf
    for i, alpha in enumerate(alphas):
        for j, k in enumerate(ks):
            for l, d in enumerate(ds):
                a_pred = m1(a_init, dadt_init, n, alpha, k, d, dt)
                loss = np.mean(np.power(a_pred - a, 2))
                losses[i, j, l] = loss
                if loss < best_loss:
                    alpha_ind = i
                    k_ind = j
                    d_ind = l
                    best_loss = loss

    print(f"Best loss: {best_loss}")
    return losses, (alpha_ind, k_ind, d_ind)


def main():
    dff_regs = np.load(os.path.join(DENOISED_DIR, GLIA_PATH))
    f_neurons = np.load(os.path.join(DENOISED_DIR, NEURON_PATH), allow_pickle=True)

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
    f0 = np.percentile(f_neurons, 20, axis=1)
    dff_neurons = ((f_neurons.T - f0) / f0).T

    av_dff_n = np.mean(dff_neurons, axis=0)

    num_frames = dff_regs.shape[0]
    num_regions = dff_regs.shape[1]
    region_colors = cm.viridis_r(np.linspace(0, 1, num_regions))
    t = np.arange(num_frames) / FS

    reg_num = 3
    a = dff_regs[:, reg_num]
    n = av_dff_n

    alphas = [1 / np.power(2, i) for i in range(6)]
    ks = [1 / np.power(2, i) for i in range(6)]
    ds = [1 / np.power(2, i) for i in range(6)]

    corrs, best_inds = grid_search_m1(a, n, alphas, ks, ds, 1 / FS)
    alpha = alphas[best_inds[0]]
    k = ks[best_inds[1]]
    d = ds[best_inds[2]]

    print(f"alpha: {alpha}")
    print(f"k: {k}")
    print(f"d: {d}")

    a_m1 = m1(a[0], (a[1] - a[0]) * FS, n, alpha, k, d, 1 / FS)

    tau = 10
    a_leaky = leaky_integrator(a[0], n, tau, 1 / FS)

    plt.figure()
    plt.plot(t, n, color="black", alpha=0.9, label="neurons")
    plt.plot(t, a, color="blue", alpha=0.9, label="astroglia")
    plt.plot(t, a_leaky, color="orange", alpha=0.9, label="astroglia - leaky")
    plt.plot(t, a_m1, color="red", alpha=0.9, label="astroglia - m1")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
