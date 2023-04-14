from tqdm import tqdm
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy import signal, ndimage
import os


from fio import (
    load_cfg,
    load_custom_rois,
    generate_exp_dir,
    generate_denoised_dir,
    generate_roi_dff_dir,
    generate_figures_dir,
    gen_npy_fname,
    gen_image_fname,
)
from sigproc import (
    get_pos,
    calc_c_pos,
    calc_pos_pca0,
)
from data_analysis_events import EventType


EXP_NAMES = [
    "20211112_13_23_43_GFAP_GCamp6s_F2_PTZ",
    "20211112_18_30_27_GFAP_GCamp6s_F5_c2",
    "20211112_19_48_54_GFAP_GCamp6s_F6_c3",
    # "20211117_14_17_58_GFAP_GCamp6s_F2_C",
    # "20211117_17_33_00_GFAP_GCamp6s_F4_PTZ",
    "20211117_21_31_08_GFAP_GCamp6s_F6_PTZ",
    "20211119_16_36_20_GFAP_GCamp6s_F4_PTZ",
    # "20211119_18_15_06_GFAP_GCamp6s_F5_C",
    # "20211119_21_52_35_GFAP_GCamp6s_F7_C",
    "20220211_13_18_56_GFAP_GCamp6s_F2_C",
    # "20220211_15_02_16_GFAP_GCamp6s_F3_PTZ",
    "20220211_16_51_15_GFAP_GCamp6s_F4_PTZ",
    # "20220412_10_43_04_GFAP_GCamp6s_F1_PTZ",
    "20220412_12_32_27_GFAP_GCamp6s_F2_PTZ",
    "20220412_13_59_55_GFAP_GCamp6s_F3_C",
    # "20220412_16_06_54_GFAP_GCamp6s_F4_PTZ",
]
CROP_IDS = ["OpticTectum"]
ROIS_FNAME = "rois_smooth"

EVENT_TYPE = EventType.LIGHT_ONSET
ISIS = [300]
NUM_EVTS = 5

BUFFER_T = 5
PRE_EVENT_T = 5
POST_EVENT_T = 120

MICROM_PER_M = 1000000

MAX_ONSET_LAG = 15
AXIAL_BIN_SIZE = 0.5
CELL_LENGTH = 55

SPATIAL_FILTER_ORDER = 4
SPATIAL_FILTER_CUTOFF_LOW = 1 / 40  # 1/micrometers
SPATIAL_FILTER_CUTOFF_HIGH = 1 / 2.5  # 1/micrometers

AMP_CM = cm.inferno
POS_CM = cm.viridis_r

USE_DENOISED = True
USE_CHAN2 = False

plt.rcParams["font.family"] = "Times New Roman"

""" def find_local_extremes(x):
    peaks_ind, prop = signal.find_peaks(x, prominence=PEAK_PROMINENCE)
    return (
        peaks_ind,
        prop["prominences"],
        prop["left_bases"].astype(int),
        prop["right_bases"].astype(int),
    ) """


def bandpass_filt(x, fl, fh, fs, order, padlen=100):
    y = np.zeros(x.shape[0] + 2 * padlen)
    y[padlen:-padlen] = x

    sos = signal.butter(order, [fl, fh], btype="bandpass", output="sos", fs=fs)
    y = signal.sosfiltfilt(sos, y)

    return y[padlen:-padlen]


def axial_bins(pos_pca0, bin_size, value_range):
    num_bins = int(np.floor((value_range[1] - value_range[0]) / bin_size))
    num_bin_edges = num_bins + 1
    bin_edges = np.linspace(value_range[0], value_range[1], num_bin_edges)
    bin_mask = np.zeros((num_bins, pos_pca0.shape[0]), dtype=bool)
    for i in range(num_bin_edges - 1):
        bin_mask[i] = np.logical_and(
            pos_pca0 >= bin_edges[i], pos_pca0 < bin_edges[i + 1]
        )

    return bin_mask, bin_edges


def plot_roi_axial_bin(c_pos, bin_mask, cmap):
    plt.figure(figsize=(12, 12))
    num_bins = bin_mask.shape[0]

    color_bin = cmap(np.linspace(0, 1, num_bins))

    for bin_num in range(num_bins):
        pos_bin = c_pos[:, bin_mask[bin_num]]
        plt.scatter(
            pos_bin[1, :], pos_bin[0, :], s=12, color=color_bin[bin_num], marker="s"
        )

    plt.tight_layout()


def plot_axial_bins_vs_time(dff_evt, x, t, event_num, cmap):
    extent = np.min(x), np.max(x), np.min(t), np.max(t)

    plt.figure(frameon=False)
    plt.imshow(
        dff_evt.T,
        cmap=cmap,
        extent=extent,
        origin="lower",
        interpolation="none",
        aspect="equal",
    )
    """ plt.plot([np.amin(x), np.amax(x)], [0, 0], color="orange", alpha=0.7)
    plt.plot([np.amin(x), np.amax(x)], [10, 10], color="darkgray", alpha=0.7) """
    plt.title(f"Event {event_num}")
    plt.xlabel(r"Distal <-> Proximal [ \mu m]")
    plt.ylabel("Time [s]")
    plt.tight_layout()


def plot_spatial_pseudo_dist(dff_evt, x, event_num):
    dff_evt_resp = dff_evt[:, : int(dff_evt.shape[1] / 2)]
    av_amp = np.mean(dff_evt_resp, axis=1)
    max_amp = np.amax(dff_evt_resp, axis=1)
    std_amp = np.std(dff_evt_resp, axis=1)

    plt.figure()
    plt.plot(x, av_amp, label="average amp", color="blue")
    plt.plot(x, max_amp, label="max amp", color="orange")
    plt.plot(x, std_amp, label="std amp", color="green")
    plt.legend()
    plt.title(f"Event {event_num+1}")
    plt.xlabel(r"Distal <-> Proximal [\mu m]")

    return [av_amp, max_amp, std_amp]


def plot_spatial_autocorrelation(amp_dists, x, event_num):
    plt.figure()
    colors = ["blue", "orange", "green"]
    labels = ["average", "max", "std"]

    for amp_dist, color, label in zip(amp_dists, colors, labels):
        mu = np.mean(amp_dist)
        sigma_sq = np.var(amp_dist)

        norm_dist = amp_dist - mu
        num_bins = norm_dist.shape[0]
        acorr = np.correlate(norm_dist, norm_dist, "full")[num_bins - 1 :]
        acorr = acorr / (sigma_sq * num_bins)

        plt.plot(x, acorr, label=label + " autocorrelation", color=color)
        plt.xlabel(r"Distance [\mu m]")
        plt.legend()
        plt.title(f"Event {event_num+1}")


def calc_autocorr(amp_dist):
    mu = np.mean(amp_dist)
    sigma_sq = np.var(amp_dist)

    norm_dist = amp_dist - mu

    num_bins = norm_dist.shape[0]
    acorr = np.correlate(norm_dist, norm_dist, "full")[num_bins - 1 :]
    acorr = acorr / (sigma_sq * num_bins)
    return acorr


def calc_spatial_dist(dff_evt, func=np.amax):
    dff_evt_resp = dff_evt[:, : int(dff_evt.shape[1] / 2)]
    amp_dist = func(dff_evt_resp, axis=1)
    return amp_dist


def calc_fft_freqs(x, d, remove_dc=False):
    y = np.fft.fft(x)
    freq = np.fft.fftfreq(y.shape[0], d)

    pos_freq = freq >= 0
    y = y[pos_freq]
    freq = freq[pos_freq]

    if remove_dc:
        y = y[1:]
        freq = freq[1:]

    return y, freq


def calc_spatial_dist_autocorr(dff_evt, func=np.amax):
    amp_dist = calc_spatial_dist(dff_evt, func)

    acorr = calc_autocorr(amp_dist)

    return amp_dist, acorr


def plot_spatial_dist_autocorr(spatial_dist, autocorr, x, event_num):
    plt.figure()
    plt.plot(x, spatial_dist, label="max amp")
    plt.plot(x, autocorr, label="max amp autocorrelation")
    plt.xlabel(r"Distance [\mu m]")
    plt.legend()
    plt.title(f"Event {event_num+1}")


def max_min_filt(z, max_filt_len=2, min_filt_len=2):
    z_max = ndimage.maximum_filter1d(z, size=max_filt_len)
    z_max_r = np.flip(ndimage.maximum_filter1d(np.flip(z), size=max_filt_len))
    z_max = np.amin(np.stack([z_max, z_max_r], axis=1), axis=1)
    # std_max = ndimage.gaussian_filter1d(std_max, sigma=20)
    # std_max_base = ndimage.gaussian_filter1d(std_max, sigma=20)

    z_max_min = ndimage.minimum_filter1d(z_max, size=min_filt_len)
    z_max_min_r = np.flip(ndimage.minimum_filter1d(np.flip(z_max), size=min_filt_len))
    z_max_base = np.amin(np.stack([z_max_min, z_max_min_r], axis=1), axis=1)
    z_max_base = ndimage.gaussian_filter1d(z_max_base, sigma=1)

    z_filt = (z - z_max_base) / z_max_base
    return z_filt


def process_roi(exp_dir, roi_num, cfg, rois, pos, fig_dir):
    roi_num_str = str(roi_num)

    fig_roi_dir = os.path.join(fig_dir, f"ROI{roi_num}")
    if not os.path.isdir(fig_roi_dir):
        os.makedirs(fig_roi_dir)

    roi = rois[roi_num]
    c_pos = calc_c_pos(roi, pos, cfg.Ly)
    pos_pca0 = calc_pos_pca0(c_pos, cfg, MICROM_PER_M)
    value_range = (np.amin(pos_pca0), np.amax(pos_pca0))
    bin_mask, _ = axial_bins(pos_pca0, AXIAL_BIN_SIZE, value_range)

    plot_roi_axial_bin(c_pos, bin_mask, POS_CM)
    plt.tight_layout()
    plt.savefig(gen_image_fname(fig_roi_dir, f"cell_axial_binned"))
    plt.close()

    dff_dir = generate_roi_dff_dir(exp_dir, roi_num_str, ROIS_FNAME)

    evt_type = EVENT_TYPE

    amp_dists = []
    amp_ffts = []
    dff_evts = []
    freqs = []

    for isi in ISIS:
        for evt_num in tqdm(range(NUM_EVTS), desc="Event", leave=False):

            try:
                dff_evt = np.load(
                    gen_npy_fname(
                        dff_dir,
                        f"dff_axial_{CELL_LENGTH}_evt_{evt_type}_ISI_{isi}_num_{evt_num}",
                    )
                )
            except FileNotFoundError:
                continue

            num_bins = dff_evt.shape[0]
            num_frames = dff_evt.shape[1]

            t = np.linspace(PRE_EVENT_T, POST_EVENT_T, num_frames)
            x = np.linspace(0, AXIAL_BIN_SIZE * (num_bins - 1), num_bins)

            plot_axial_bins_vs_time(
                dff_evt,
                x,
                t,
                evt_num,
                AMP_CM,
            )
            plt.savefig(gen_image_fname(fig_roi_dir, f"2d_light_response_evt{evt_num}"))
            plt.close()

            amp_dist = calc_spatial_dist(dff_evt, np.amax)
            amp_fft, freq = calc_fft_freqs(amp_dist, AXIAL_BIN_SIZE, remove_dc=True)

            amp_dists.append(amp_dist)
            amp_ffts.append(amp_fft)
            dff_evts.append(dff_evt)
            freqs.append(freq)

            plt.figure()
            plt.plot(x, amp_dist, label="max amp")
            plt.xlabel(r"Distance from distal end [\mu m]")
            plt.legend()
            plt.title(f"Event {evt_num+1}")
            plt.tight_layout()
            plt.savefig(gen_image_fname(fig_roi_dir, f"amp_dist_evt{evt_num}"))
            plt.close()

            plt.figure()
            plt.plot(
                freq, 10 * np.log10(np.power(np.absolute(amp_fft), 2)), label="power"
            )
            plt.plot(freq, np.angle(amp_fft), label="phase")
            plt.xlabel("MegaSpaceHz [1 / \u03bc m]")
            plt.legend()
            plt.title(f"Event {evt_num+1}")
            plt.tight_layout()
            plt.savefig(gen_image_fname(fig_roi_dir, f"amp_dist_fft_evt{evt_num}"))
            plt.close()

            # norm_amp_dist = max_min_filt(amp_dist)
            """ filt_amp_dist = bandpass_filt(
                amp_dist,
                SPATIAL_FILTER_CUTOFF_LOW,
                SPATIAL_FILTER_CUTOFF_HIGH,
                1 / AXIAL_BIN_SIZE,
                SPATIAL_FILTER_ORDER,
            )

            filt_auto_corr = calc_autocorr(filt_amp_dist)
            plot_spatial_dist_autocorr(filt_amp_dist, filt_auto_corr, x, evt_num) """

    if len(amp_dists) == 0:
        return None, None, None, None, None, None

    amp_dists = np.array(amp_dists)
    amp_ffts = np.array(amp_ffts)
    dff_evts = np.array(dff_evts)

    av_dff_evt = np.mean(dff_evts, axis=0)

    plot_axial_bins_vs_time(
        av_dff_evt,
        x,
        t,
        "Average",
        AMP_CM,
    )
    plt.tight_layout()
    plt.savefig(gen_image_fname(fig_roi_dir, "2d_light_response_average"))
    plt.close()

    plt.figure()
    for i in range(amp_dists.shape[0]):
        amp_dist = amp_dists[i]
        if i == 0:
            plt.plot(x, amp_dist, label="single events", alpha=0.5, color="darkgray")
        else:
            plt.plot(x, amp_dist, alpha=0.5, color="darkgray")

    plt.plot(x, np.mean(amp_dists, axis=0), label="average", color="black", alpha=0.8)
    plt.legend()
    plt.xlabel("Distance from distal end [\u03bcm]")
    plt.ylabel(r"Max amplitude axial bin [$\Delta$F/F0]")
    plt.tight_layout()
    plt.savefig(gen_image_fname(fig_roi_dir, f"amp_dist_average"))
    plt.close()

    plt.figure()
    for i in range(amp_ffts.shape[0]):
        amp_fft = amp_ffts[i]
        if i == 0:
            plt.plot(
                freq,
                10 * np.log10(np.power(np.absolute(amp_fft), 2)),
                label="single events",
                alpha=0.5,
                color="darkgray",
            )
        else:
            plt.plot(
                freq,
                10 * np.log10(np.power(np.absolute(amp_fft), 2)),
                alpha=0.5,
                color="darkgray",
            )

    plt.plot(
        freq,
        np.mean(10 * np.log10(np.power(np.absolute(amp_ffts), 2)), axis=0),
        label="average",
        color="black",
        alpha=0.8,
    )
    plt.legend()
    plt.xlabel("MegaSpaceHz [1 / \u03bcm]")
    plt.ylabel("Power")
    plt.tight_layout()
    plt.savefig(gen_image_fname(fig_roi_dir, f"amp_dist_fft_average"))
    plt.close()

    # plt.show()

    return t, av_dff_evt, x, np.mean(amp_dists, axis=0), freq, np.mean(amp_ffts, axis=0)


def main():
    fig_dir = os.path.join(generate_figures_dir(), "amplitude_spatial")

    ts = []
    dffs = []
    xs = []
    amp_dists = []
    freqs = []
    amp_ffts = []

    for exp_name in tqdm(EXP_NAMES, desc="Experiment"):
        for crop_id in CROP_IDS:
            exp_dir = generate_exp_dir(exp_name, crop_id)
            cfg = load_cfg(exp_dir)

            if USE_DENOISED:
                exp_dir = generate_denoised_dir(exp_dir, USE_CHAN2)

            rois = load_custom_rois(exp_dir, ROIS_FNAME)
            n_rois = len(rois)
            pos = get_pos(cfg.Ly, cfg.Lx)

            fig_exp_dir = os.path.join(fig_dir, exp_name)
            if not os.path.isdir(fig_exp_dir):
                os.makedirs(fig_exp_dir)

            for roi_num in tqdm(range(n_rois), desc="Roi", leave=False):
                t_r, dff_r, x_r, amp_dist_r, freq_r, amp_fft_r = process_roi(
                    exp_dir, roi_num, cfg, rois, pos, fig_exp_dir
                )

                if t_r is not None:
                    ts.append(t_r)
                    dffs.append(dff_r)
                    xs.append(x_r)
                    amp_dists.append(amp_dist_r)
                    freqs.append(freq_r)
                    amp_ffts.append(amp_fft_r)

    ts, dffs, xs, amp_dists, freqs, amp_ffts = (
        np.array(ts),
        np.array(dffs),
        np.array(xs),
        np.array(amp_dists),
        np.array(freqs),
        np.array(amp_ffts),
    )
    print(dffs.shape)

    plot_axial_bins_vs_time(
        np.mean(dffs, axis=0),
        np.mean(xs, axis=0),
        ts[0],
        "Grand average",
        AMP_CM,
    )
    plt.tight_layout()
    plt.savefig(gen_image_fname(fig_dir, f"2d_light_response_average"))

    plt.figure()
    for i in range(amp_dists.shape[0]):
        amp_dist = amp_dists[i]
        x = xs[i]
        if i == 0:
            plt.plot(x, amp_dist, label="single rois", alpha=0.5, color="darkgray")
        else:
            plt.plot(x, amp_dist, alpha=0.5, color="darkgray")

    plt.plot(x, np.mean(amp_dists, axis=0), label="average", color="black", alpha=0.8)
    plt.legend()
    plt.xlabel("Distance from distal end [\u03bcm]")
    plt.ylabel(r"Max amplitude axial bin [$\Delta$F/F0]")
    plt.tight_layout()
    plt.savefig(gen_image_fname(fig_dir, f"amp_dist_average"))

    plt.figure()
    for i in range(amp_ffts.shape[0]):
        amp_fft = amp_ffts[i]
        freq = freqs[i]
        if i == 0:
            plt.plot(
                freq,
                10 * np.log10(np.power(np.absolute(amp_fft), 2)),
                label="single rois",
                alpha=0.5,
                color="darkgray",
            )
        else:
            plt.plot(
                freq,
                10 * np.log10(np.power(np.absolute(amp_fft), 2)),
                alpha=0.5,
                color="darkgray",
            )

    plt.plot(
        freq,
        np.mean(10 * np.log10(np.power(np.absolute(amp_ffts), 2)), axis=0),
        label="average",
        color="black",
        alpha=0.8,
    )
    plt.legend()
    plt.xlabel("MegaSpaceHz [1 / \u03bcm]")
    plt.ylabel("Power")
    plt.tight_layout()
    plt.savefig(gen_image_fname(fig_dir, f"amp_dist_fft_average"))

    plt.show()


if __name__ == "__main__":
    main()
