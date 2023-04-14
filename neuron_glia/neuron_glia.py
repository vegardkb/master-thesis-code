import numpy as np
import os
from scipy import signal, stats
from fio import generate_exp_dir, load_cfg, generate_denoised_dir, load_custom_rois
from sigproc import get_pos, calc_c_pos, calc_pos_pca0
from rastermap import Rastermap
import matplotlib.pyplot as plt
import matplotlib.cm as cm


EXP_NAME = "20220604_13_23_11_HuC_GCamp6s_GFAP_jRGECO_F1_C"
# EXP_NAME = "20220604_16_24_03_HuC_GCamp6s_GFAP_jRGECO_F3_PTZ"
# EXP_NAME = "20220604_15_00_04_HuC_GCamp6s_GFAP_jRGECO_F2_PTZ"
CROP_ID = "OpticTectum"
EXP_DIR = f"Y:\\Vegard\\data\\{EXP_NAME}\\{CROP_ID}"
DENOISED_NEURON_DIR = os.path.join(EXP_DIR, "denoised")
DENOISED_GLIA_DIR = os.path.join(EXP_DIR, "denoised_chan2")

ROI_NUM = 0
GLIA_PATH = f"results\\ROI{ROI_NUM}\\dff\\rois\\dff_stat.npy"
NEURON_PATH = "suite2p\\plane0\\F.npy"
NEURON_STAT_PATH = "suite2p\\plane0\\stat.npy"
NEUROPIL_PATH = "suite2p\\plane0\\Fneu.npy"
ISCELL_PATH = "suite2p\\plane0\\iscell.npy"

SUBTRACT_NEUROPIL = False
NEUROPIL_COEF = 0.7

LOW_PASS_FILTER = True
LOW_PASS_FILTER_C = 0.25
LOW_PASS_FILTER_ORDER = 4

USE_DENOISED = True

ROIS_FNAME = "rois.npy"


MICROM_PER_M = 1000000
NUM_REGIONS = 6

GLIA_CM = cm.viridis_r
NEURON_CM = cm.rainbow


def get_reg_mask(pos_pca0, n_regions):
    """
    Divide into regions by position along proximal-distal axis, equal length of regions
    """

    thr = np.linspace(
        np.amin(pos_pca0) - 1e-10, np.amax(pos_pca0) + 1e-10, n_regions + 1
    )

    mask = []
    for i in range(n_regions):
        mask.append(np.logical_and(pos_pca0 > thr[i], pos_pca0 <= thr[i + 1]))

    return np.array(mask)


def get_reg_activity(reg_mask, x):
    """
    reg_mask: 2d boolean ndarray (n_regions, pix)
    x: 2d ndarray (t, pix)

    return:
     - y: 2d ndarray (t, n_regions)
    """
    num_frames = x.shape[0]
    num_reg = reg_mask.shape[0]
    y = np.zeros((num_frames, num_reg))
    for reg_num in range(reg_mask.shape[0]):
        x_reg = x[:, reg_mask[reg_num]]
        dyn_range = np.amax(x_reg, axis=0) - np.amin(x_reg, axis=0)
        max_ind = np.argmax(dyn_range)
        y[:, reg_num] = x_reg[:, max_ind]

    return y


def low_pass_filter(dff, fs, f_c, order, axis=0):
    sos = signal.butter(order, f_c, btype="low", output="sos", fs=fs)
    dff_filt = signal.sosfiltfilt(sos, dff, axis=axis)
    return dff_filt


def load_glia(cfg):
    dff = np.load(os.path.join(DENOISED_GLIA_DIR, GLIA_PATH))
    roi = np.load(os.path.join(DENOISED_GLIA_DIR, ROIS_FNAME))[ROI_NUM]

    fs = cfg.volume_rate
    pos = get_pos(cfg.Ly, cfg.Lx)
    c_pos = calc_c_pos(roi, pos, cfg.Ly)
    pos_pd = calc_pos_pca0(c_pos, cfg, MICROM_PER_M)

    region_mask = get_reg_mask(pos_pd, NUM_REGIONS)

    num_regions_cell = region_mask.shape[0]
    reg_mean_pos_2d = np.zeros((2, num_regions_cell))
    for reg_num in range(num_regions_cell):
        reg_pos = c_pos[:, region_mask[reg_num]]
        reg_mean_pos_2d[:, reg_num] = np.mean(reg_pos, axis=1)

    dff_reg = get_reg_activity(region_mask, dff)

    if LOW_PASS_FILTER:
        dff_reg = low_pass_filter(dff_reg, fs, LOW_PASS_FILTER_C, LOW_PASS_FILTER_ORDER)

    return dff_reg.T, reg_mean_pos_2d.T


def load_neurons(cfg):
    f_neurons = np.load(
        os.path.join(DENOISED_NEURON_DIR, NEURON_PATH), allow_pickle=True
    )
    stat_neurons = np.load(
        os.path.join(DENOISED_NEURON_DIR, NEURON_STAT_PATH), allow_pickle=True
    )

    if SUBTRACT_NEUROPIL:
        f_neurons = f_neurons - np.load(
            os.path.join(DENOISED_NEURON_DIR, NEUROPIL_PATH), allow_pickle=True
        )

    iscell = np.load(os.path.join(DENOISED_NEURON_DIR, ISCELL_PATH), allow_pickle=True)

    f_neurons = f_neurons[iscell[:, 0] > 0.5]
    stat_neurons = stat_neurons[iscell[:, 0] > 0.5]
    pos_neurons = np.array(
        [stat_neurons[i]["med"] for i in range(stat_neurons.shape[0])]
    )
    pos_neurons[:, 0] = cfg.Ly - pos_neurons[:, 0]
    f0 = np.percentile(f_neurons, 20, axis=1)
    dff_neurons = ((f_neurons.T - f0) / f0).T

    return dff_neurons, pos_neurons


def plot_pos(pos_g, pos_n):
    plt.figure()

    glia_colors = GLIA_CM(np.linspace(0, 1, pos_g.shape[0]))
    glia_dot_size = 40
    for i, g_pos in enumerate(pos_g):
        plt.scatter(g_pos[1], g_pos[0], s=glia_dot_size, color=glia_colors[i])

    neuron_colors = NEURON_CM(np.linspace(0, 1, pos_n.shape[0]))
    neuron_dot_size = 5
    for i, n_pos in enumerate(pos_n):
        plt.scatter(
            n_pos[1], n_pos[0], s=neuron_dot_size, color=neuron_colors[i], alpha=0.8
        )


def plot_act(
    dff_g,
    dff_n,
    fs,
):
    model = Rastermap(n_components=1, init="pca")
    _ = model.fit_transform(dff_n)

    t = np.arange(dff_g.shape[1]) / fs

    glia_colors = GLIA_CM(np.linspace(0, 1, dff_g.shape[0]))

    plt.figure()
    for i, g_dff in enumerate(dff_g):
        plt.plot(t, g_dff, color=glia_colors[i], alpha=0.9)

    plt.figure()
    plt.pcolormesh(t, np.arange(dff_n.shape[0]), dff_n[model.isort, :], cmap="inferno")
    plt.tick_params(left=False, right=False, labelleft=False)


def plot_corr(dff_g, dff_n, pos_g, pos_n, pixel_size):
    corr_regs = []
    dist_regs = []
    for reg_num in range(NUM_REGIONS):
        dff_reg = dff_g[reg_num]
        pos_reg = pos_g[reg_num]

        corr_regs.append(
            np.array(
                [stats.pearsonr(dff_reg, dff_n[i])[0] for i in range(dff_n.shape[0])]
            )
        )
        dist_regs.append(
            pixel_size * np.sqrt(np.sum(np.power(pos_n - pos_reg, 2), axis=1))
        )

    corr = np.array(corr_regs)
    dist = np.array(dist_regs)

    color_regs = GLIA_CM(np.linspace(0, 1, NUM_REGIONS))
    plt.figure()
    for reg_num in range(NUM_REGIONS):
        plt.plot(
            np.sort(np.abs(corr[reg_num])),
            np.cumsum(corr[reg_num]) / np.sum(corr[reg_num]),
            color=color_regs[reg_num],
            alpha=0.9,
        )

    plt.xlabel("Correlation")
    plt.ylabel("Ratio of cells")

    plt.figure()
    for reg_num in range(NUM_REGIONS):
        corr_reg = corr[reg_num]
        dist_reg = dist[reg_num]
        pos_mask = corr_reg > 0
        neg_mask = np.logical_not(pos_mask)

        pos_corr, pos_dist = corr_reg[pos_mask], dist_reg[pos_mask]
        neg_corr, neg_dist = corr_reg[neg_mask], dist_reg[neg_mask]

        num_bins = 20
        min_dist, max_dist = np.amin(dist_reg), np.amax(dist_reg)
        dist_bins = np.linspace(min_dist - 1e-12, max_dist + 1e-12, num_bins + 1)

        x = []
        y_pos = []
        y_neg = []
        for d_l, d_h in zip(dist_bins[:-1], dist_bins[1:]):
            x.append((d_l + d_h) / 2)

            mask = np.logical_and(pos_dist >= d_l, pos_dist < d_h)
            if not np.any(mask):
                y_pos.append(np.nan)
            else:
                y_pos.append(np.mean(pos_corr[mask]))

            mask = np.logical_and(neg_dist >= d_l, neg_dist < d_h)
            if not np.any(mask):
                y_neg.append(np.nan)
            else:
                y_neg.append(np.mean(neg_corr[mask]))

        plt.plot(x, y_pos, color=color_regs[reg_num])
        plt.plot(x, y_neg, color=color_regs[reg_num], alpha=0.8)

    plt.xlabel("Distance [micrometers]")


def process_exp():
    cfg = load_cfg(EXP_DIR)
    fs = cfg.volume_rate

    dff_g, pos_g = load_glia(cfg)
    dff_n, pos_n = load_neurons(cfg)

    print("Loaded data:")
    print(f"dff_g.shape: {dff_g.shape}")
    print(f"pos_g.shape: {pos_g.shape}")
    print(f"dff_n.shape: {dff_n.shape}")
    print(f"pos_n.shape: {pos_n.shape}")

    plot_pos(pos_g, pos_n)

    plot_act(dff_g, dff_n, fs)

    plot_corr(dff_g, dff_n, pos_g, pos_n, cfg.pixel_size)

    plt.show()


def main():
    process_exp()


if __name__ == "__main__":
    main()
