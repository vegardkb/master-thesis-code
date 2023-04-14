from tqdm import tqdm
import numpy as np
import threading
import multiprocessing
import os
from scipy import signal


from fio import (
    load_cfg,
    load_custom_rois,
    generate_exp_dir,
    generate_denoised_dir,
    generate_roi_dff_dir,
    generate_figures_dir,
    gen_npy_fname,
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
CROP_ID = "OpticTectum"
ROIS_FNAME = "rois_smooth"

EVENT_TYPE = EventType.LIGHT_ONSET
ISIS = [300]

BUFFER_T = 5
PRE_EVENT_T = 5

MICROM_PER_M = 1000000

CELL_LENGTH = 55
AXIAL_BIN_SIZE = 0.5

USE_DENOISED = True
USE_CHAN2 = False


SPATIAL_FILTER_CUTOFF = 1 / 2  # 1/micrometers
SPATIAL_FILTER_ORDER = 4

TEMPORAL_FILTER_CUTOFF = 0.25
TEMPORAL_FILTER_ORDER = 8

PEAK_PROMINENCE = 0.05


def low_pass_filter(dff_raw, fs, f_c, order):
    sos = signal.butter(order, f_c, btype="low", output="sos", fs=fs)
    dff_reg = signal.sosfiltfilt(sos, dff_raw, axis=0)
    return dff_reg


def first_derivative(x, dt):
    return (x[1:-5] - 8 * x[2:-4] + 8 * x[4:-2] - x[5:-1]) / (12 * dt)


def second_derivative(x, dt):
    return (-x[1:-5] + 16 * x[2:-4] - 30 * x[3:-3] + 16 * x[4:-2] - x[5:-1]) / (
        12 * dt**2
    )


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


def process_exp(exp_name):
    exp_dir = generate_exp_dir(exp_name, CROP_ID)
    cfg = load_cfg(exp_dir)

    if USE_DENOISED:
        exp_dir = generate_denoised_dir(exp_dir, USE_CHAN2)

    rois = load_custom_rois(exp_dir, ROIS_FNAME)
    n_rois = len(rois)
    pos = get_pos(cfg.Ly, cfg.Lx)

    buffer_samples = int(BUFFER_T * cfg.volume_rate)
    pre_event_samples = int(PRE_EVENT_T * cfg.volume_rate)

    for roi_num in tqdm(range(n_rois), desc="Roi", leave=False):
        roi_num_str = str(roi_num)
        roi = rois[roi_num]
        c_pos = calc_c_pos(roi, pos, cfg.Ly)
        pos_pca0 = calc_pos_pca0(c_pos, cfg, MICROM_PER_M)

        dff_dir = generate_roi_dff_dir(exp_dir, roi_num_str, ROIS_FNAME)

        max_pos_pca0 = np.amax(pos_pca0)
        min_range = max_pos_pca0 - CELL_LENGTH
        if min_range < np.amin(pos_pca0):
            continue

        value_range = (min_range, max_pos_pca0)

        bin_mask, _ = axial_bins(pos_pca0, AXIAL_BIN_SIZE, value_range)
        num_bins = bin_mask.shape[0]
        dff_dir = generate_roi_dff_dir(exp_dir, roi_num_str, ROIS_FNAME)

        evt_type = EVENT_TYPE

        for isi in ISIS:
            dff_evts = np.load(gen_npy_fname(dff_dir, f"dff_evt_{evt_type}_ISI_{isi}"))

            dff_evts = dff_evts[:, buffer_samples:-buffer_samples]
            bl = np.mean(dff_evts[:, :pre_event_samples], axis=1)
            dff_evts = np.swapaxes(np.swapaxes(dff_evts, 0, 1) - bl, 0, 1)

            num_evts = dff_evts.shape[0]

            for evt_num in range(num_evts):
                dff_evt = dff_evts[evt_num]

                dff_evt = low_pass_filter(
                    dff_evt,
                    cfg.volume_rate,
                    TEMPORAL_FILTER_CUTOFF,
                    TEMPORAL_FILTER_ORDER,
                )
                ddffdt = np.zeros(dff_evt.shape)
                ddffdt[3:-3] = first_derivative(dff_evt, 1 / cfg.volume_rate)

                d2dffdt2 = np.zeros(dff_evt.shape)
                d2dffdt2[3:-3] = second_derivative(dff_evt, 1 / cfg.volume_rate)

                num_frames = dff_evt.shape[0]

                """ binned_dff = np.zeros((num_bins, num_frames))
                for frame_num in range(num_frames):
                    for bin_num in range(num_bins):
                        binned_dff[bin_num, frame_num] = np.percentile(
                            dff_evt[frame_num, bin_mask[bin_num]], 90
                        ) """

                binned_dff = np.zeros((num_bins, num_frames))
                binned_ddffdt = np.zeros((num_bins, num_frames))
                binned_d2dffdt2 = np.zeros((num_bins, num_frames))
                for bin_num in range(num_bins):
                    dff_bin = dff_evt[:, bin_mask[bin_num]]
                    ddffdt_bin = ddffdt[:, bin_mask[bin_num]]
                    d2dffdt2_bin = d2dffdt2[:, bin_mask[bin_num]]

                    range_dff = np.amax(dff_bin, axis=0) - np.amin(dff_bin, axis=0)
                    max_range_ind = np.argmax(range_dff)

                    binned_dff[bin_num] = dff_bin[:, max_range_ind]
                    binned_ddffdt[bin_num] = ddffdt_bin[:, max_range_ind]
                    binned_d2dffdt2[bin_num] = d2dffdt2_bin[:, max_range_ind]

                np.save(
                    gen_npy_fname(
                        dff_dir,
                        f"dff_axial_{CELL_LENGTH}_evt_{evt_type}_ISI_{isi}_num_{evt_num}",
                    ),
                    binned_dff,
                )
                np.save(
                    gen_npy_fname(
                        dff_dir,
                        f"ddffdt_axial_{CELL_LENGTH}_evt_{evt_type}_ISI_{isi}_num_{evt_num}",
                    ),
                    binned_ddffdt,
                )
                np.save(
                    gen_npy_fname(
                        dff_dir,
                        f"d2dffdt2_axial_{CELL_LENGTH}_evt_{evt_type}_ISI_{isi}_num_{evt_num}",
                    ),
                    binned_d2dffdt2,
                )


def main():
    """threads = []

    for exp_name in EXP_NAMES:
        t = threading.Thread(target=process_exp, args=(exp_name,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()"""

    processes = []

    for exp_name in EXP_NAMES:
        p = multiprocessing.Process(target=process_exp, args=(exp_name,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
