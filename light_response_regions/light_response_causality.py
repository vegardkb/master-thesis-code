from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pickle
import copy
from scipy import signal
from scipy.stats import linregress

from fio import (
    load_cfg,
    load_custom_rois,
    generate_global_results_dir,
    generate_exp_dir,
    generate_denoised_dir,
    generate_roi_dff_dir,
    gen_npy_fname,
    gen_pickle_fname,
)
from sigproc import (
    get_pos,
    calc_c_pos,
    calc_pos_pca0,
    get_reg_mask,
    get_reg_activity,
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
ROIS_FNAME = "rois"

DFF_REGS_FNAME = "dff_light_response"
CURVATURE_REGS_FNAME = "curvature_light_response"
STATS_FNAME = "stats_light_response"

MAX_CELL_LENGTH_MICROMETERS = 111.6  # Ugly, fix?
MICROM_PER_M = 1000000
N_REGIONS = 5
DISTAL_REG = 0
PROXIMAL_REG = N_REGIONS - 1
BUFFER_T = 5
PRE_EVENT_T = 5
POST_EVENT_T = 120
POST_EVENT_T_PEAK = 40
POST_STIM_TIME = 10
STIM_DURATION = 10

DECAY_THRESHOLD = 0.95
RATIO_INCLUDE = 0.2
MIN_PIX_INCLUDE = 10
CURVATURE_THRESHOLD = 1e-6

EVENT_TYPE = EventType.LIGHT_ONSET

USE_DENOISED = True
USE_CHAN2 = False

# ISIS = [300, 120, 60, 30, 15]
ISIS = [300]

CELL_LENGTH_THRESHOLDS = [40, 55]

FREQ_CUT_OFF = 0.25
FILTER_ORDER = 8

MAX_LAG_Y = 10
MIN_LAG_X = 1
MAX_LAG_X = 10


def granger_causality(x, y, fs):

    y_shift = y[MAX_LAG_Y:]
    t = np.linspace(0, y_shift.shape[0] - 1, y_shift.shape[0]) / fs

    p0 = 0.05

    X_medial = []
    p_val_medial = []
    r_val_medial = []

    x_shift = x[:-lag]
    reg_result = linregress(x, y)
    p_val_medial.append(reg_result.pvalue)
    r_val_medial.append(reg_result.rvalue)
    X_medial.append(x)

    r_val_medial = np.array(r_val_medial)
    p_val_medial = np.array(p_val_medial)
    X_medial = np.array(X_medial)
    X_medial = X_medial[p_val_medial < p0].T
    reg_medial = LinearRegression().fit(X_medial, y)
    y_pred_medial = reg_medial.predict(X_medial)

    X_lateral = []
    p_val_lateral = []
    r_val_lateral = []
    for lag in lags:
        x = dff_lateral[max_lag - lag : -lag]
        reg_result = stats.linregress(x, y)
        p_val_lateral.append(reg_result.pvalue)
        r_val_lateral.append(reg_result.rvalue)
        X_lateral.append(x)

    r_val_lateral = np.array(r_val_lateral)
    p_val_lateral = np.array(p_val_lateral)
    X_lateral = np.array(X_lateral)
    X_lateral = X_lateral[p_val_lateral < p0].T
    reg_lateral = LinearRegression().fit(X_lateral, y)
    y_pred_lat = reg_lateral.predict(X_lateral)

    plt.figure()
    t_lag = np.array(lags) / cfg.volume_rate
    plt.plot(t_lag, p_val_lateral, label="Lateral")
    plt.plot(t_lag, p_val_medial, label="Medial")
    plt.legend()
    plt.xlabel("Time lag")
    plt.ylabel("p-value")
    plt.show()

    X = np.concatenate([X_medial, X_lateral], axis=1)
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)

    rss_med = np.sum(np.power(y - y_pred_medial, 2))
    rss_tot = np.sum(np.power(y - y_pred, 2))
    k_med = np.sum(p_val_medial > p0)
    k_tot = k_med + np.sum(p_val_lateral > p0)
    n = y.shape[0]

    f_stat = (rss_med - rss_tot) * (n - k_tot) / ((k_tot - k_med) * rss_tot)

    plt.figure(figsize=(10, 10))
    plt.plot(t, y, color="blue", label="True activity")
    plt.plot(t, y_pred_medial, color="darkgray", label="Predicted activity")
    plt.plot(t, y - y_pred_medial, color="brown", label="Prediction error")
    plt.title(f"Medial (Var(y-y_pred) = {np.round(np.var(y-y_pred_medial), 6)})")
    plt.legend()

    plt.figure(figsize=(10, 10))
    plt.plot(t, y, color="blue", label="True activity")
    plt.plot(t, y_pred, color="darkgray", label="Predicted activity")
    plt.plot(t, y - y_pred, color="brown", label="Prediction error")
    plt.title(f"Medial and lateral (F = {np.round(f_stat, 3)})")
    plt.legend()

    plt.show()


def get_responding_mask(amp, ratio_include):
    num_regions = len(amp)

    responding_mask = []

    for reg_num in range(num_regions):
        num_pix = amp[reg_num].shape[0]
        responding = np.zeros(num_pix, dtype=bool)
        amp_ind_sort = np.argsort(amp[reg_num])
        num_include = max(int(num_pix * ratio_include), MIN_PIX_INCLUDE)
        include_ind = amp_ind_sort[num_include:]
        for pix in include_ind:
            responding[pix] = True

        responding_mask.append(responding)

    return responding_mask


def average_responding_dff(dff_reg, responding_mask):
    num_regions = len(dff_reg)
    num_frames = dff_reg[0].shape[0]
    av_dff = np.zeros((num_frames, num_regions))
    for reg_num in range(num_regions):
        av_dff[:, reg_num] = np.mean(
            dff_reg[reg_num][:, responding_mask[reg_num]], axis=1
        )

    return av_dff


def main():
    for exp_name in tqdm(EXP_NAMES, desc="Experiment"):
        for crop_id in tqdm(CROP_IDS, desc="Crop id", leave=False):
            exp_dir = generate_exp_dir(exp_name, crop_id)
            cfg = load_cfg(exp_dir)

            if USE_DENOISED:
                exp_dir = generate_denoised_dir(exp_dir, USE_CHAN2)

            rois = load_custom_rois(exp_dir, ROIS_FNAME)
            n_rois = len(rois)
            pos = get_pos(cfg.Ly, cfg.Lx)

            for roi_num in tqdm(range(n_rois), desc="Roi", leave=False):
                roi_num_str = str(roi_num)
                roi = rois[roi_num]
                c_pos = calc_c_pos(roi, pos, cfg.Ly)
                pos_pca0 = calc_pos_pca0(c_pos, cfg, MICROM_PER_M)
                c_length = np.amax(pos_pca0) - np.amin(pos_pca0)
                cell_length = 0
                for c_length_t in CELL_LENGTH_THRESHOLDS:
                    if c_length > c_length_t:
                        mask = np.amax(pos_pca0) - pos_pca0 < c_length_t
                        cell_length = c_length_t

                if cell_length == 0:
                    continue

                pos_pd = pos_pca0[mask]
                region_mask = get_reg_mask(pos_pd, N_REGIONS)

                num_regions_cell = region_mask.shape[0]
                reg_mean_pos = np.zeros(num_regions_cell)
                for reg_num in range(num_regions_cell):
                    reg_pos = pos_pd[region_mask[reg_num]]
                    reg_mean_pos[reg_num] = np.mean(reg_pos)

                dff_dir = generate_roi_dff_dir(exp_dir, roi_num_str, ROIS_FNAME)

                evt_type = EVENT_TYPE

                for isi in tqdm(ISIS, desc="ISI", leave=False):
                    dff_evts = np.load(
                        gen_npy_fname(dff_dir, f"dff_evt_{evt_type}_ISI_{isi}")
                    )

                    dff_evts = dff_evts[:, :, mask]

                    num_evts = dff_evts.shape[0]

                    for evt_num in tqdm(range(num_evts), desc="Event", leave=False):
                        dff_evt = dff_evts[evt_num]
                        dff_reg = get_reg_activity(region_mask, dff_evt)

                        dff_distal = np.mean(dff_reg[DISTAL_REG], axis=1)
                        dff_proximal = np.mean(dff_reg[PROXIMAL_REG], axis=1)


if __name__ == "__main__":
    main()
