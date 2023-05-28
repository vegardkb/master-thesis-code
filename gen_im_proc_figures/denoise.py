import numpy as np
import tifffile
from tqdm import tqdm
import skimage as ski
from rastermap import Rastermap
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import os


EXP_DIR = (
    "Y:\\Vegard\\data\\20220604_13_23_11_HuC_GCamp6s_GFAP_jRGECO_F1_C\\OpticTectum"
)

ALIGNED_GLIA_FNAME = "aligned_chan2.tif"
DENOISED_GLIA_FNAME = "denoised_chan2.tif"

VOLUME_RATE = 4.86

THRESHOLD_ABS = 700
BIN_SIZE = 1  # size of square for spatial bin
EDGE_ARTIFACT = 0  # number of edge artifact frames after lowpass filter

SHOW_THRESH_IMPACT = False
SAVE_FILT = False

CLIM = (300, 1500)

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
    plt.figure()
    plt.imshow(im, cmap="gray", origin="lower")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.tight_layout()
    plt.clim(CLIM[0], CLIM[1])


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
        ax.plot(t, x, color=color, label=label, alpha=0.8)

    ax.set_xlabel("Time [min]")
    if normalize:
        ax.set_ylabel("Fluorescence [a.u.]")

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
            f_denoised = np.mean(img, axis=(1, 2))
        else:
            single_frames.append(img[frame_num])
            f_aligned = np.mean(img[frames_dropped_denoise:], axis=(1, 2))

    f_denoised, f_aligned = clip_longest(f_denoised, f_aligned)

    set_fig_size(0.48, 1)
    for im, label in zip(single_frames, labels):
        plot_frame(im)
        save_fig(fig_dir, f"{label}_frame")

    set_fig_size(0.6, 1)
    plot_traces(
        [f_aligned, f_denoised],
        [ORANGE, LIGHTBLUE],
        ["aligned", "denoised"],
        VOLUME_RATE,
        normalize=True,
    )
    save_fig(fig_dir, "traces_av_aligned_denoised")
    plt.show()


if __name__ == "__main__":
    main()
