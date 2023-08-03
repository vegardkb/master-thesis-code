import numpy as np
import os
from scipy import signal, stats
from fio import generate_exp_dir, load_cfg, generate_denoised_dir, load_custom_rois
from sigproc import get_pos, calc_c_pos, calc_pos_pca0
from rastermap import Rastermap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


# EXP_NAME = "20220604_13_23_11_HuC_GCamp6s_GFAP_jRGECO_F1_C"
EXP_NAME = "20220604_16_24_03_HuC_GCamp6s_GFAP_jRGECO_F3_PTZ"
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

FIG_DIR = "Y:\\Vegard\\data\\figures\\neuron_glia"

SUBTRACT_NEUROPIL = False
NEUROPIL_COEF = 0.7

LOW_PASS_FILTER = True
LOW_PASS_FILTER_C = 0.3
LOW_PASS_FILTER_ORDER = 8

USE_DENOISED = True

ROIS_FNAME = "rois.npy"


MICROM_PER_M = 1000000
NUM_REGIONS = 6

GLIA_CM = cm.viridis_r
NEURON_CM = cm.rainbow

SMALL_SIZE = 8
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE, dpi=600)  # fontsize of the figure title
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["lines.linewidth"] = 0.75


def set_fig_size(scale, y_scale):
    a4_w = 8.3
    lmargin = 1
    rmargin = 1
    text_width = a4_w - lmargin - rmargin
    plt.rc(
        "figure", figsize=(scale * text_width, y_scale * scale * text_width), dpi=600
    )


def save_fig(fig_dir, fname):
    plt.savefig(
        os.path.join(fig_dir, fname + ".png"),
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(fig_dir, fname + ".svg"),
        bbox_inches="tight",
    )


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


def get_colors(colormap, num_regs):
    return colormap(np.linspace(0.2, 1, num_regs))


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


def plot_pos(pos_g, pos_n, cluster):
    plt.figure()

    num_regions = pos_g.shape[0]
    glia_colors = get_colors(GLIA_CM, num_regions)
    glia_dot_size = 40
    for i, g_pos in enumerate(pos_g):
        plt.scatter(g_pos[1], g_pos[0], s=glia_dot_size, color=glia_colors[i])

    num_clusters = np.amax(cluster) + 1
    neuron_colors = get_colors(NEURON_CM, num_clusters)
    neuron_dot_size = 5
    for cluster_id, color in zip(range(num_clusters), neuron_colors):
        mask = cluster == cluster_id
        plt.scatter(
            pos_n[mask, 1], pos_n[mask, 0], s=neuron_dot_size, color=color, alpha=0.8
        )


def second_derivative(x, dt):
    return (-x[1:-5] + 16 * x[2:-4] - 30 * x[3:-3] + 16 * x[4:-2] - x[5:-1]) / (
        12 * dt**2
    )


def detect_peaks(dff_g, reg, fs):
    peaks, _ = signal.find_peaks(dff_g[reg], height=1, prominence=1)

    return peaks


def get_trials(dff_g, dff_n, events, fs):
    """pre_peak_search_t = int(50 * fs)

    d2xdt2 = np.zeros(dff_g.shape[1])
    d2xdt2[3:-3] = second_derivative(dff_g[0], 1 / fs)

    onsets = []

    for event in events:
        rel_d2x = d2xdt2[event - pre_peak_search_t : event]

        t_onset = 0
        largest_event = 0

        this_event = 0
        this_start = 0
        for ind, d2x in enumerate(rel_d2x):
            if d2x <= 0:
                if this_event > largest_event:
                    largest_event = this_event
                    t_onset = int((this_start + ind - 1) / 2)

                this_event = 0
                continue

            if this_event == 0:
                this_start = ind

            this_event += d2x

        if this_event > largest_event:
            largest_event = this_event
            t_onset = int((this_start + ind - 1) / 2)

        onset = t_onset
        onsets.append(onset + event - pre_peak_search_t)"""

    pre_onset = int(120 * fs)
    post_onset = int(120 * fs)
    g = []
    n = []
    for event in events:
        if (event + post_onset >= dff_g.shape[1]) or (event - pre_onset < 0):
            continue

        g.append(
            (
                dff_g[:, event - pre_onset : event + post_onset].T
                - np.mean(dff_g[:, event - pre_onset : event], axis=1)
            ).T
        )
        n.append(
            (
                dff_n[:, event - pre_onset : event + post_onset].T
                - np.mean(dff_n[:, event - pre_onset : event], axis=1)
            ).T
        )

    return np.array(g), np.array(n), events


def get_amp_diff(dff, onset_distal, fs):
    """
    dff: (n_trials, n_regions, n_samples)
    onset_distal: time of distal onset
    fs: sample rate
    """

    dff_p_post = dff[:, :, int(onset_distal * fs) :]
    amp = np.amax(dff_p_post, axis=2)
    amp_diff = amp[:, -1] - amp[:, 0]
    return amp_diff


def plot_trace_stacked(
    x,
    fs,
    colormap,
    events=None,
    vlines=None,
    vline_colors=None,
):

    num_regs = x.shape[0]
    t = np.arange(x.shape[1]) / fs

    delta_y = np.amax(np.absolute(x))
    raw_scale_factor = 0.5
    delta_y = delta_y * raw_scale_factor

    colors = get_colors(colormap, num_regs)

    _, ax = plt.subplots()

    for reg_num in range(num_regs):
        dff_reg = x[reg_num]
        ax.plot(
            t,
            dff_reg - delta_y * reg_num,
            color=colors[reg_num],
            # label=f"ROI {reg_num+1}",
            alpha=0.8,
        )

    scalebar_width_pixels = 10
    inv = ax.transData.inverted()
    points = inv.transform([(0, 0), (scalebar_width_pixels, scalebar_width_pixels)])
    scale_x = points[0, 1] - points[0, 0]
    scale_y = points[1, 1] - points[1, 0]

    bar_size = 1  # min
    loc = "lower center"
    asb = AnchoredSizeBar(
        ax.transData,
        size=bar_size * fs * 60,
        size_vertical=scale_y,
        label=f"{str(bar_size)} min",
        loc=loc,
        pad=0.1,
        borderpad=-2,
        frameon=False,
        color="black",
    )
    ax.add_artist(asb)

    ylabel = "% $\Delta F / F_0$"

    bar_size = max(((delta_y / 5) // 0.5) * 0.5, 0.5)  # dff
    if second_derivative:
        bar_size = max(((delta_y / 5) // 0.05) * 0.05, 0.05)
    loc = "upper left"
    asb = AnchoredSizeBar(
        ax.transData,
        size=scale_x,
        size_vertical=bar_size / 1,
        label=f"{str(int(bar_size * 100))}{ylabel}",
        loc=loc,
        pad=0.1,
        borderpad=-2,
        frameon=False,
        color="black",
    )
    ax.add_artist(asb)

    ax.axis("off")

    if events is not None:
        events_t = t[events]
        plt.scatter(events_t, np.ones(events_t.shape), color="black", alpha=0.8, s=10)

    if vlines is not None:
        for vline, v_color in zip(vlines, vline_colors):
            ax.axvline(vline, 0.1, 0.9, color=v_color, linestyle="dashed")

    plt.tight_layout()


def plot_act(
    dff_g,
    dff_n,
    fs,
):
    set_fig_size(1, 1)

    plot_trace_stacked(dff_g, fs)
    save_fig(FIG_DIR, f"dff_g_{EXP_NAME}")

    model = Rastermap(n_components=1, init="pca")
    _ = model.fit_transform(dff_n)
    t = np.arange(dff_n.shape[1]) / fs

    plt.figure()
    plt.pcolormesh(t, np.arange(dff_n.shape[0]), dff_n[model.isort, :], cmap="inferno")
    plt.tick_params(left=False, right=False, labelleft=False)
    save_fig(FIG_DIR, f"dff_n_{EXP_NAME}")

    plt.close("all")


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


def get_clusters(x, n_clusters):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters)

    kmeans.fit(x)

    y = kmeans.cluster_centers_
    cluster_id = kmeans.predict(x)

    return y, cluster_id


def plot_manifold(dff_n, dff_g):
    pca = PCA(n_components=2)
    n_2d = pca.fit_transform(dff_n.T)
    av_g = np.mean(dff_g, axis=0)
    t = np.arange(n_2d.shape[0])

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(t, n_2d[:, 0], n_2d[:, 1], c=av_g, cmap=cm.inferno, s=2)
    ax.set_xlabel("time")
    ax.set_ylabel("PC1")
    ax.set_zlabel("PC2")


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
    # set_fig_size(1, 1)

    # Ugly

    # Save fig instead of showing
    # plot_act(dff_g, dff_n, fs)

    plot_trace_stacked(dff_g, fs, GLIA_CM)
    save_fig(FIG_DIR, f"dff_g_{EXP_NAME}")

    dff_n_c, cluster_id = get_clusters(dff_n, 8)
    plot_pos(pos_g, pos_n, cluster_id)
    save_fig(FIG_DIR, f"pos_{EXP_NAME}")

    plot_trace_stacked(dff_n_c, fs, NEURON_CM)
    save_fig(FIG_DIR, f"dff_n_c_{EXP_NAME}")

    events = detect_peaks(dff_g, 0, fs)

    plot_trace_stacked(dff_g, fs, GLIA_CM, events=events)
    save_fig(FIG_DIR, f"dff_g_w_event_{EXP_NAME}")

    plot_trace_stacked(dff_n_c, fs, NEURON_CM, events=events)
    save_fig(FIG_DIR, f"dff_n_c_w_event_{EXP_NAME}")

    dff_g_t, dff_n_t, _ = get_trials(dff_g, dff_n_c, events, fs)

    plot_trace_stacked(
        np.mean(dff_g_t, axis=0),
        fs,
        GLIA_CM,
        vlines=[120],
        vline_colors=[GLIA_CM(0.2)],
    )
    save_fig(FIG_DIR, f"av_dff_g_{EXP_NAME}")

    plot_trace_stacked(
        np.mean(dff_n_t, axis=0),
        fs,
        NEURON_CM,
        vlines=[120],
        vline_colors=[GLIA_CM(0.2)],
    )
    save_fig(FIG_DIR, f"av_dff_n_{EXP_NAME}")

    amp_diff = get_amp_diff(dff_g_t, 120, fs)

    median_amp = np.median(amp_diff)
    high_amp = amp_diff > median_amp
    low_amp = amp_diff <= median_amp

    for mask, group in zip([low_amp, high_amp], ["low_amp", "high_amp"]):
        plot_trace_stacked(
            np.mean(dff_g_t[mask], axis=0),
            fs,
            GLIA_CM,
            vlines=[120],
            vline_colors=[GLIA_CM(0.2)],
        )
        save_fig(FIG_DIR, f"av_dff_g_{group}_{EXP_NAME}")

        plot_trace_stacked(
            np.mean(dff_n_t[mask], axis=0),
            fs,
            NEURON_CM,
            vlines=[120],
            vline_colors=[GLIA_CM(0.2)],
        )
        save_fig(FIG_DIR, f"av_dff_n_{group}_{EXP_NAME}")

    plot_trace_stacked(
        np.mean(dff_g_t[high_amp], axis=0) - np.mean(dff_g_t[low_amp], axis=0),
        fs,
        GLIA_CM,
        vlines=[120],
        vline_colors=[GLIA_CM(0.2)],
    )
    save_fig(FIG_DIR, f"av_dff_g_diff_amp_{EXP_NAME}")

    plot_trace_stacked(
        np.mean(dff_n_t[high_amp], axis=0) - np.mean(dff_n_t[low_amp], axis=0),
        fs,
        NEURON_CM,
        vlines=[120],
        vline_colors=[GLIA_CM(0.2)],
    )
    save_fig(FIG_DIR, f"av_dff_n_diff_amp_{EXP_NAME}")

    """ trial_dir = os.path.join(FIG_DIR, "trialwise")
    num_trials = amp_proximal.shape[0]
    for trial_num in range(num_trials):
        plot_trace_stacked(
            dff_g_t[trial_num],
            fs,
            GLIA_CM,
            vlines=[120],
            vline_colors=[GLIA_CM(0.2)],
        )
        save_fig(trial_dir, f"trial_{trial_num}_dff_g_{EXP_NAME}")

        plot_trace_stacked(
            dff_n_t[trial_num],
            fs,
            NEURON_CM,
            vlines=[120],
            vline_colors=[GLIA_CM(0.2)],
        )
        save_fig(trial_dir, f"trial_{trial_num}_dff_n_{EXP_NAME}") """

    """ plot_manifold(dff_n, dff_g)
    save_fig(FIG_DIR, "neuron_manifold") """

    # Something is buggy
    # plot_corr(dff_g, dff_n, pos_g, pos_n, cfg.pixel_size)

    """
        Simple demo:
            - Detect calcium waves
            - Give spike triggered averages with the activity prior to spike

    """
    plt.show()


def main():
    process_exp()


if __name__ == "__main__":
    main()
