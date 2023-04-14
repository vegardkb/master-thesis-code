from tqdm import tqdm
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy import signal, ndimage, stats
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


def plot_axial_bins_vs_time(
    dff_evt, x, t, event_num, cmap, title="", equal_clim=False, interpolation="none"
):
    extent = np.min(x), np.max(x), np.min(t), np.max(t)

    plt.figure(frameon=False)
    plt.imshow(
        dff_evt.T,
        cmap=cmap,
        extent=extent,
        origin="lower",
        interpolation=interpolation,
        aspect="equal",
    )
    plt.title(title + f"Event {event_num}")
    plt.xlabel(r"Distal <-> Proximal [ \mu m]")
    plt.ylabel("Time [s]")
    plt.colorbar()
    if equal_clim:
        max_c = np.amax(np.absolute(dff_evt))
        plt.clim(-max_c, max_c)


def calc_lag_corr(x, x_ref, fs, max_lag=5):
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

    for isi in ISIS:
        for evt_num in tqdm(range(NUM_EVTS), desc="Event", leave=False):

            try:
                dff_evt = np.load(
                    gen_npy_fname(
                        dff_dir,
                        f"dff_axial_{CELL_LENGTH}_evt_{evt_type}_ISI_{isi}_num_{evt_num}",
                    )
                )
                ddffdt = np.load(
                    gen_npy_fname(
                        dff_dir,
                        f"ddffdt_axial_{CELL_LENGTH}_evt_{evt_type}_ISI_{isi}_num_{evt_num}",
                    )
                )
                d2dffdt2 = np.load(
                    gen_npy_fname(
                        dff_dir,
                        f"d2dffdt2_axial_{CELL_LENGTH}_evt_{evt_type}_ISI_{isi}_num_{evt_num}",
                    )
                )
            except FileNotFoundError:
                continue

            num_bins = dff_evt.shape[0]
            num_frames = dff_evt.shape[1]

            t = np.linspace(PRE_EVENT_T, POST_EVENT_T, num_frames)
            x = np.linspace(0, AXIAL_BIN_SIZE * (num_bins - 1), num_bins)

            plot_axial_bins_vs_time(dff_evt, x, t, evt_num, cm.inferno, title="dff ")
            """ plot_axial_bins_vs_time(
                ddffdt, x, t, evt_num, cm.coolwarm, title="ddffdt ", equal_clim=True
            ) """

            d2dffdt2_pos = np.zeros(d2dffdt2.shape)
            d2dffdt2_neg = np.zeros(d2dffdt2.shape)

            d2dffdt2_pos[d2dffdt2 > 0] = d2dffdt2[d2dffdt2 > 0]
            d2dffdt2_neg[d2dffdt2 < 0] = np.absolute(d2dffdt2[d2dffdt2 < 0])

            plot_axial_bins_vs_time(
                d2dffdt2, x, t, evt_num, cm.coolwarm, title="d2dffdt2 ", equal_clim=True
            )
            plot_axial_bins_vs_time(
                d2dffdt2_pos,
                x,
                t,
                evt_num,
                cm.inferno,
                title="d2dffdt2 pos ",
                interpolation="bessel",
            )
            """ plot_axial_bins_vs_time(
                d2dffdt2_neg, x, t, evt_num, cm.inferno, title="d2dffdt2 neg "
            ) """

            """ fs = cfg.volume_rate
            lag = np.zeros(d2dffdt2.shape[0])
            corr = np.zeros(d2dffdt2.shape[0])

            d = 2
            for bin_num in range(num_bins):
                if bin_num < d:
                    lag[bin_num], corr[bin_num] = calc_lag_corr(
                        d2dffdt2[bin_num], d2dffdt2[bin_num + d], fs
                    )

                elif bin_num > num_bins - 1 - d:
                    lag[bin_num], corr[bin_num] = calc_lag_corr(
                        d2dffdt2[bin_num], d2dffdt2[bin_num - d], fs
                    )

                else:
                    lag_l, corr_l = calc_lag_corr(
                        d2dffdt2[bin_num], d2dffdt2[bin_num - d], fs
                    )
                    lag_r, corr_r = calc_lag_corr(
                        d2dffdt2[bin_num], d2dffdt2[bin_num + d], fs
                    )
                    lag[bin_num] = (lag_l + lag_r) / 2
                    corr[bin_num] = (corr_l + corr_r) / 2 """

            """ bad_mask = corr < 0.5
            lag[bad_mask] = np.nan
            corr[bad_mask] = np.nan """

            """ amp_range = np.amax(dff_evt, axis=1) - np.amin(dff_evt, axis=1)
            amp_range = amp_range / np.amax(amp_range)

            plt.figure()
            plt.plot(x, lag, label="lag")
            plt.plot(x, amp_range, label="amp")
            plt.xlabel("Distal <-> Proximal")
            plt.ylabel("Lag neighbour [sec]")
            pearson_result = stats.pearsonr(lag, amp_range)
            plt.title(
                f"Lags d={d} pearsonr={np.round(pearson_result[0], 3)} p={np.round(pearson_result[1], 3)}"
            )
            plt.legend()

            plt.figure()
            plt.plot(x, corr)
            plt.xlabel("Distal <-> Proximal")
            plt.ylabel("Correlation")
            plt.title(f"Corr d={d}") """

            plt.show()


def main():
    fig_dir = os.path.join(generate_figures_dir(), "amplitude_spatial")

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
                process_roi(exp_dir, roi_num, cfg, rois, pos, fig_exp_dir)


if __name__ == "__main__":
    main()
