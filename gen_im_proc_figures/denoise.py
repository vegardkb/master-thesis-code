import numpy as np
import tifffile
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import os
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy import signal


EXP_DIR = "Y:\\Vegard\\data\\20220211_13_18_56_GFAP_GCamp6s_F2_C\\OpticTectum"

ALIGNED_GLIA_FNAME = "aligned.tif"
DENOISED_GLIA_FNAME = "denoised.tif"

VOLUME_RATE = 4.86
MICRON_PER_PIXEL = 0.455729

CLIM = (500, 3000)

LIGHTBLUE = (101 / 255, 221 / 255, 247 / 255)
ORANGE = (247 / 255, 151 / 255, 54 / 255)

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE, dpi=300)  # fontsize of the figure title
plt.rcParams["font.family"] = "serif"
plt.rcParams["lines.linewidth"] = 0.75


def set_fig_size(scale, y_scale):
    a4_w = 8.3
    lmargin = 1.5
    rmargin = 1
    text_width = a4_w - lmargin - rmargin
    plt.rc("figure", figsize=(scale * text_width, y_scale * scale * text_width))


def save_fig(fig_dir, fname):
    plt.savefig(
        os.path.join(fig_dir, fname + ".png"),
        bbox_inches="tight",
    )


def calc_snr(data):
    """
    data: np.ndarray (num_frames, num_pixels)
    """
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    snr = mu / std
    print(f"min snr: {np.amin(snr)}")
    print(f"max snr: {np.amax(snr)}")
    return snr


def plot_frame(im):
    fig, ax = plt.subplots()
    axim = ax.imshow(im, cmap="gray", origin="lower")
    ax.axis("off")
    axim.set_clim(CLIM[0], CLIM[1])

    bar_size = 10
    loc = "lower right"
    asb = AnchoredSizeBar(
        ax.transData,
        size=bar_size / MICRON_PER_PIXEL,
        size_vertical=1,
        label=f"{str(bar_size)} \u03bcm",
        loc=loc,
        pad=0.1,
        borderpad=2,
        frameon=False,
        color="white",
    )
    ax.add_artist(asb)

    plt.tight_layout()


def calc_dist(x, num_bins=100):
    bin_edges = np.linspace(np.amin(x) - 1e-6, np.amax(x) + 1e-6, num_bins + 1)
    dist = np.zeros(num_bins)
    bins = np.zeros(num_bins)
    for i, tl, th in zip(range(num_bins), bin_edges[:-1], bin_edges[1:]):
        dist[i] = np.sum(np.logical_and(x >= tl, x < th))
        bins[i] = (tl + th) / 2

    dist = dist / np.sum(dist)
    return bins, dist


def clip_longest(x1, x2):
    l1 = x1.shape[0]
    l2 = x2.shape[0]

    if l1 > l2:
        return x1[:l2], x2

    elif l2 > l1:
        return x1, x2[:l1]

    else:
        return x1, x2


def plot_traces(xs, colors, labels, fs, normalize=True):

    t = np.arange(0, xs[0].shape[0]) / fs / 60

    _, ax = plt.subplots()

    if normalize:
        for i, x in enumerate(xs):
            xs[i] = (x - np.mean(x)) / np.std(x)

    for x, color, label in zip(xs, colors, labels):
        ax.plot(t, x, color=color, label=label, alpha=0.6)

    ax.set_xlabel("Time [min]")
    if normalize:
        ax.set_ylabel("Fluorescence [a.u.]")

    ax.legend()


def plot_snr(xs, colors, labels, fs):

    _, ax = plt.subplots()

    for x, color, label in zip(xs, colors, labels):
        mu = np.reshape(np.mean(x, axis=0), (-1))

        sos = signal.butter(2, 0.5, btype="high", output="sos", fs=fs)
        x_hp = signal.sosfiltfilt(sos, x, axis=0)
        sigma = np.reshape(np.std(x_hp, axis=0), (-1))

        snr = np.sort(mu / sigma)

        ax.plot(snr, np.cumsum(snr) / np.sum(snr), label=label, color=color, alpha=0.8)

    ax.set_xlabel("SNR")
    ax.set_ylabel("CDF(SNR)")
    ax.legend()


def main():

    fig_dir = "Y:\\Vegard\\data\\figures\\preprocess"
    frame_num = 100
    frames_dropped_denoise = 30

    single_frames = []
    labels = ["aligned", "denoised"]
    fnames = [ALIGNED_GLIA_FNAME, DENOISED_GLIA_FNAME]

    f_denoised = None
    f_aligned = None

    for fname, label in zip(fnames, labels):
        print(f"Loading {fname}")
        img = tifffile.imread(os.path.join(EXP_DIR, fname))

        if label == "denoised":
            single_frames.append(img[frame_num - frames_dropped_denoise])
            f_denoised = img
        else:
            single_frames.append(img[frame_num])
            f_aligned = img[frames_dropped_denoise:]

    f_denoised, f_aligned = clip_longest(f_denoised, f_aligned)

    f_av_aligned, f_av_denoised = np.mean(f_aligned, axis=(1, 2)), np.mean(
        f_denoised, axis=(1, 2)
    )

    set_fig_size(0.48, 1)
    for im, label in zip(single_frames, labels):
        plot_frame(im)
        save_fig(fig_dir, f"{label}_frame")

    set_fig_size(0.48, 1)
    plot_traces(
        [f_av_aligned, f_av_denoised],
        [ORANGE, LIGHTBLUE],
        ["aligned", "denoised"],
        VOLUME_RATE,
        normalize=True,
    )
    save_fig(fig_dir, "traces_av_aligned_denoised")

    set_fig_size(0.48, 1)
    plot_snr(
        [f_aligned, f_denoised],
        [ORANGE, LIGHTBLUE],
        ["aligned", "denoised"],
        VOLUME_RATE,
    )
    save_fig(fig_dir, "snr_aligned_denoised")
    plt.show()


if __name__ == "__main__":
    main()
