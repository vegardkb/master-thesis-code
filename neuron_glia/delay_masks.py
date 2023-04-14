import numpy as np
import tifffile
from tqdm import tqdm
import skimage as ski
from rastermap import Rastermap
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from scipy import signal
import os

EXP_NAMES = [
    "20220604_13_23_11_HuC_GCamp6s_GFAP_jRGECO_F1_C",
    "20220604_15_00_04_HuC_GCamp6s_GFAP_jRGECO_F2_PTZ",
    "20220604_16_24_03_HuC_GCamp6s_GFAP_jRGECO_F3_PTZ",
]

THRESHOLD_ABS = 700
BIN_SIZE = 1  # size of square for spatial bin
EDGE_ARTIFACT = 0  # number of edge artifact frames after lowpass filter

SHOW_THRESH_IMPACT = False
SAVE_FILT = False

VOLUME_RATE = 5.15
MAX_LAG = 15

HIGHPASS_FILTER = False
HIGHPASS_CUTOFF = 1 / 120
FILTER_ORDER = 2

CROP_Y_START, CROP_Y_END = 100, 250
CROP_X_START, CROP_X_END = 0, 300

TIMEBINS_XCORR = 10
REG_CM = cm.viridis_r


NUM_REGIONS = 4


def matplotlib_params():
    plt.rcParams["figure.figsize"] = (12, 12)


def crop_img(img):
    img = img[:, CROP_Y_START:CROP_Y_END]
    img = img[:, :, CROP_X_START:CROP_X_END]
    return img


def high_pass_filter(im, fs, f_c, order, axis=0):
    sos = signal.butter(order, f_c, btype="high", output="sos", fs=fs)
    im_filt = signal.sosfiltfilt(sos, im, axis=axis)
    return im_filt


def space_bin(img, bin_size=BIN_SIZE):
    img_dim = img.shape
    nb_frames = img_dim[0]
    y_dim = img_dim[1]
    x_dim = img_dim[2]

    x_bin = x_dim // bin_size
    y_bin = y_dim // bin_size
    x_crop = x_bin * bin_size
    y_crop = y_bin * bin_size

    img = img[:, :y_crop, :x_crop]
    img_bin = np.empty((nb_frames, y_bin, x_bin))

    print("Bin image in space ({} by {} groups)".format(bin_size, bin_size))
    for frame in tqdm(range(nb_frames)):
        img_bin[frame] = ski.measure.block_reduce(img[frame], bin_size, func=np.mean)
    return img_bin


def threshold(
    img,
    thresh=THRESHOLD_ABS,
    show_thresh_impact=SHOW_THRESH_IMPACT,
):
    print("Thresholding (t={}) data".format(thresh))

    img_dim = img.shape
    nb_frames = img_dim[0]
    y_dim = img_dim[1]
    x_dim = img_dim[2]

    img_flat = np.reshape(img, (nb_frames, x_dim * y_dim))

    if show_thresh_impact:
        thresh_impact(3000, img_flat, step=100)

    low_columns = np.all(img_flat < thresh, axis=0)
    filtered_img_flat = img_flat[:, ~low_columns]

    img_flat[:, low_columns] = 0
    # img_filt = img_flat.reshape(img_dim)

    print(
        "{}/{} pixels removed".format(
            x_dim * y_dim - filtered_img_flat.shape[1], x_dim * y_dim
        )
    )
    return (
        filtered_img_flat,
        img_flat,
    )  # img_flat is same as filtered_img_flat but with original size, removed pixels are marked 0


def thresh_impact(thresh_max, flat_img, step=0.05):
    test_y = []
    print("Running thresh_impact")
    for thresh_test in tqdm(np.arange(0.0, thresh_max, step)):
        low_columns_test = np.all(flat_img < thresh_test, axis=0)
        filtered_flat_img = flat_img[:, ~low_columns_test]
        test_y.append(filtered_flat_img.shape[1])
    plt.figure()
    plt.title("Impact of Threshold")
    plt.minorticks_on()
    plt.grid(which="major", color="#DDDDDD", linewidth=0.8)
    plt.grid(which="minor", color="#DDDDDD", linestyle=":", linewidth=0.5)
    plt.plot(np.arange(0.0, thresh_max, step), test_y)
    plt.xlabel("Threshold")
    plt.ylabel("Pixels Retained")
    plt.legend()
    plt.show()


def dff_sig(img, bin_size=BIN_SIZE, thresh=THRESHOLD_ABS, edge_artifact=EDGE_ARTIFACT):

    if edge_artifact > 0:
        img = img[edge_artifact:-edge_artifact, :, :]

    if bin_size > 1:
        img = space_bin(img)

    if thresh > 0:
        filtered_img_flat, img_flat_full = threshold(img)

    dff_signals = (
        filtered_img_flat.astype(np.float64)
        - np.mean(filtered_img_flat.astype(np.float64), axis=0)
    ) / np.mean(filtered_img_flat.astype(np.float64), axis=0)
    return dff_signals.transpose(), img_flat_full


def post_proc_img_dim(img, bin_size=BIN_SIZE):
    img_dim = img.shape
    y_dim = img_dim[1]
    x_dim = img_dim[2]
    if bin_size > 1:
        x_dim = x_dim // bin_size
        y_dim = y_dim // bin_size
    return (y_dim, x_dim)


def calc_lag_corr(x, x_ref, fs, max_lag):
    norm_x = np.sqrt(np.sum(np.power(x, 2)))
    norm_ref = np.sqrt(np.sum(np.power(x_ref, 2)))
    corr = signal.correlate(x - np.mean(x), x_ref - np.mean(x_ref), mode="same")
    norm = norm_x * norm_ref

    corr = corr / norm
    lags = np.arange(x.shape[0]) / fs
    lags = lags - np.amax(lags) / 2

    lag_mask = np.absolute(lags) < max_lag
    lags = lags[lag_mask]
    corr = corr[lag_mask]

    lag_ind = np.argmax(corr)

    return lags[lag_ind], corr[lag_ind]


def main():
    for exp_name in EXP_NAMES:
        INPUT_TIF = f"Y:\\Vegard\\data\\{exp_name}\\OpticTectum\\denoised_chan2.tif"
        RESULTS_DIR = f"Y:\\Vegard\\data\\{exp_name}\\OpticTectum\\denoised\\results"
        FIGURES_DIR = f"Y:\\Vegard\\data\\{exp_name}\\OpticTectum\\denoised\\figures"

        if not os.path.isdir(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

        if not os.path.isdir(FIGURES_DIR):
            os.makedirs(FIGURES_DIR)

        print(f"Reading {INPUT_TIF}")
        img = tifffile.imread(INPUT_TIF)
        img = crop_img(img)

        dff_signals, img_flat_full = dff_sig(img)
        print(f"dff_signals.shape: {dff_signals.shape}")

        if HIGHPASS_FILTER:
            print("Performing high-pass filtering")
            dff_signals = high_pass_filter(
                dff_signals, VOLUME_RATE, HIGHPASS_CUTOFF, FILTER_ORDER, axis=1
            )

        nonzero_mask = np.any(img_flat_full > 0, axis=0)
        non_zero_pos = np.nonzero(nonzero_mask)[0]

        non_zero_pos = np.reshape(non_zero_pos, (np.size(non_zero_pos), 1))

        num_pixels = dff_signals.shape[0]
        dff_mean = np.mean(dff_signals, axis=0)
        corrs, lags, weights = (
            np.zeros((num_pixels, TIMEBINS_XCORR)),
            np.zeros((num_pixels, TIMEBINS_XCORR)),
            np.zeros((num_pixels, TIMEBINS_XCORR)),
        )

        print("Starting to calculate xcorrs")
        dff_signals_bin = np.reshape(
            dff_signals, (dff_signals.shape[0], TIMEBINS_XCORR, -1)
        )
        dff_mean_bin = np.mean(dff_signals_bin, axis=0)
        for time_ind in tqdm(range(TIMEBINS_XCORR), desc="timebin"):
            dff_bin = dff_signals_bin[:, time_ind]
            dff_ref = dff_mean_bin[time_ind]
            for pixel_ind in tqdm(range(num_pixels), desc="pixel", leave=False):
                dff = dff_bin[pixel_ind]
                lags[pixel_ind, time_ind], corrs[pixel_ind, time_ind] = calc_lag_corr(
                    dff, dff_ref, VOLUME_RATE, MAX_LAG
                )
                weights[pixel_ind, time_ind] = np.std(dff)

        corrs = np.average(corrs, axis=1, weights=weights)
        lags = np.average(lags, axis=1, weights=weights)

        plt.figure()
        plt.hist(corrs, 300)
        plt.ylabel("Pixel count")
        plt.xlabel("Correlation")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "corrs"))
        plt.close()

        plt.figure()
        plt.hist(lags, 300, range=(-MAX_LAG, MAX_LAG))
        plt.ylabel("Pixel count")
        plt.xlabel("Lags [sec]")
        plt.savefig(os.path.join(FIGURES_DIR, "lags"))
        plt.close()

        plt.figure()
        plt.plot(np.arange(dff_mean.shape[0]) / VOLUME_RATE, dff_mean)
        plt.xlabel("Time")
        plt.ylabel(r"$\Delta F/F_0$")
        plt.savefig(os.path.join(FIGURES_DIR, "mean_dff"))
        plt.close()

        corr_img = np.zeros(img_flat_full.shape[1])
        lag_img = np.zeros(img_flat_full.shape[1])
        for idx, non_zero_idx in enumerate(non_zero_pos):
            corr_img[non_zero_idx] = corrs[idx]
            lag_img[non_zero_idx] = lags[idx]

        corr_img = corr_img.reshape(post_proc_img_dim(img))
        corr_img[corr_img < 0] = 0
        corr_img[corr_img > 0] = 1
        lag_img = lag_img.reshape(post_proc_img_dim(img))

        plt.figure(frameon=False, figsize=(10, 10))
        plt.imshow(
            np.zeros(img_flat_full.shape[1]).reshape(post_proc_img_dim(img)),
            cmap="gray",
        )
        plt.imshow(lag_img, cmap="viridis_r", alpha=corr_img)
        plt.title("Delay map of glia")
        cbar = plt.colorbar()
        cbar.set_label("Lag [sec]")
        plt.clim(-MAX_LAG, MAX_LAG)
        plt.tight_layout()
        plt.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
        plt.savefig(os.path.join(FIGURES_DIR, "delay_map"))
        plt.close()

        min_lag, max_lag = -15, 10
        min_corr = 0.4
        lag_ts = np.linspace(min_lag, max_lag, NUM_REGIONS + 1)

        lag_masks = [
            np.logical_and(lags >= tl, lags < th)
            for tl, th in zip(lag_ts[:-1], lag_ts[1:])
        ]
        corr_mask = corrs > min_corr

        num_frames = dff_signals.shape[1]
        t = np.arange(num_frames) / VOLUME_RATE
        t_minutes = t / 60
        dff_regs = np.zeros((num_frames, NUM_REGIONS))
        for reg_num, lag_mask in enumerate(lag_masks):
            dff_regs[:, reg_num] = np.mean(dff_signals[lag_mask], axis=0)

        np.save(os.path.join(RESULTS_DIR, "dff_regs.npy"), dff_regs)

        reg_colors = REG_CM(np.linspace(0, 1, NUM_REGIONS))

        plt.figure(figsize=(20, 12))
        for reg_num in range(NUM_REGIONS):
            plt.plot(t_minutes, dff_regs[:, reg_num], color=reg_colors[reg_num])

        plt.xlabel("Time [min]")
        plt.ylabel(r"$\Delta F/F_0$")
        plt.savefig(os.path.join(FIGURES_DIR, "traces_regions"))
        plt.close()

    """
        Rastermap stuff below
    """

    """ model = Rastermap(n_components=1, init="pca")
    embedding = model.fit_transform(dff_signals)
    # note: embedding and non_zero_pos have the same size

    plt.figure()
    plt.imshow(dff_signals[model.isort, :], cmap="inferno")
    plt.title("Activity Trace")
    plt.tick_params(left=False, right=False, labelleft=False)
    plt.clim(0, 0.9)

    plt.figure()
    plt.plot(embedding[model.isort])
    plt.title("Sorted Embedding")
    plt.minorticks_on()
    plt.grid(which="major", color="#DDDDDD", linewidth=0.8)
    plt.grid(which="minor", color="#DDDDDD", linestyle=":", linewidth=0.5)
    plt.xlabel("Pixel Number")
    plt.ylabel("Cluster Identity")

    embedding_img = np.zeros(img_flat_full.shape[1])
    for embedding_idx, non_zero_idx in enumerate(non_zero_pos):
        embedding_img[non_zero_idx] = embedding[embedding_idx]
    embedding_img = embedding_img.reshape(post_proc_img_dim(img))

    plt.figure()
    plt.imshow(
        np.zeros(img_flat_full.shape[1]).reshape(post_proc_img_dim(img)), cmap="gray"
    )
    plt.imshow(embedding_img, cmap="rainbow", alpha=corr_img)
    plt.title("Rastermap of glia")
    plt.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )
    plt.colorbar() """


if __name__ == "__main__":
    main()
