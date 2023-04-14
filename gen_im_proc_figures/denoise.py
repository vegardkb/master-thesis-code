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

THRESHOLD_ABS = 700
BIN_SIZE = 1  # size of square for spatial bin
EDGE_ARTIFACT = 0  # number of edge artifact frames after lowpass filter

SHOW_THRESH_IMPACT = False
SAVE_FILT = False

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


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


def calc_dist(x, num_bins=100):
    bin_edges = np.linspace(np.amin(x) - 1e-6, np.amax(x) + 1e-6, num_bins + 1)
    dist = np.zeros(num_bins)
    bins = np.zeros(num_bins)
    for i, tl, th in zip(range(num_bins), bin_edges[:-1], bin_edges[1:]):
        dist[i] = np.sum(np.logical_and(x >= tl, x < th))
        bins[i] = (tl + th) / 2

    dist = dist / np.sum(dist)
    return bins, dist


def main():

    frame_num = 100
    frames_dropped_denoise = 30

    single_frames = []
    snrs = []
    labels = ["aligned", "denoised"]
    fnames = [ALIGNED_GLIA_FNAME, DENOISED_GLIA_FNAME]

    for fname, label in zip(fnames, labels):
        print(f"Loading {fname}")
        img = tifffile.imread(os.path.join(EXP_DIR, fname))

        if label == "denoised":
            single_frames.append(img[frame_num - frames_dropped_denoise])
        else:
            single_frames.append(img[frame_num])

        img_flat = np.reshape(img, (img.shape[0], -1))

        print(f"Calculating SNR")
        snrs.append(calc_snr(img_flat))

    for im, _ in zip(single_frames, labels):
        plot_frame(im)

    plt.figure()
    for snr, label in zip(snrs, labels):
        print(f"Calculating SNR distribution")
        bins, snr_dist = calc_dist(snr)
        plt.plot(bins, snr_dist, label=label)

    plt.xlabel("SNR")
    plt.ylabel("Ratio of pixels")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
