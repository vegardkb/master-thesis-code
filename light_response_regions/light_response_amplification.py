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
ROIS_FNAME = "rois_smooth"

DFF_REGS_FNAME = "dff_light_response"
CURVATURE_REGS_FNAME = "curvature_light_response"
STATS_FNAME = "stats_light_response"

MAX_CELL_LENGTH_MICROMETERS = 111.6  # Ugly, fix?
MICROM_PER_M = 1000000
N_REGIONS = 8
DISTAL_REG = 0
PROXIMAL_REG = N_REGIONS - 1
BUFFER_T = 5
PRE_EVENT_T = 5
POST_EVENT_T = 120
POST_EVENT_T_PEAK = 40
POST_STIM_TIME = 10
STIM_DURATION = 10

DECAY_THRESHOLD = 0.5
RATIO_INCLUDE = 0.2
MIN_PIX_INCLUDE = 10
CURVATURE_THRESHOLD = 1e-6

EVENT_TYPE = EventType.LIGHT_ONSET

USE_DENOISED = True
USE_CHAN2 = False

# ISIS = [300, 120, 60, 30, 15]
ISIS = [300]

CELL_LENGTH_THRESHOLDS = [40, 60]
CELL_LENGTH = 60

FREQ_CUT_OFF = 0.25
FILTER_ORDER = 8


def first_derivative(x, dt):
    return (x[1:-5] - 8 * x[2:-4] + 8 * x[4:-2] - x[5:-1]) / (12 * dt)


def second_derivative(x, dt):
    return (-x[1:-5] + 16 * x[2:-4] - 30 * x[3:-3] + 16 * x[4:-2] - x[5:-1]) / (
        12 * dt**2
    )


def third_derivative(x, dt):
    return (
        x[:-6] - 8 * x[1:-5] + 13 * x[2:-4] - 13 * x[4:-2] + 8 * x[5:-1] - x[6:]
    ) / (8 * dt**3)


def get_responding_mask_old(amp, ratio_include):
    num_regions = amp.shape[0]
    num_pix = amp.shape[1]

    responding_mask = np.zeros((num_regions, num_pix), dtype=bool)

    for reg_num in range(num_regions):
        amp_ind_sort = np.argsort(amp[reg_num])
        include_ind = amp_ind_sort[int((1 - ratio_include) * num_pix) :]
        for pix in include_ind:
            responding_mask[reg_num, pix] = True

    return responding_mask


def average_responding_dff_old(dff_reg_pix, responding_mask):
    num_frames = dff_reg_pix.shape[0]
    num_regions = dff_reg_pix.shape[1]
    av_dff = np.zeros((num_frames, num_regions))
    for reg_num in range(num_regions):
        av_dff[:, reg_num] = np.mean(
            dff_reg_pix[:, reg_num, responding_mask[reg_num]], axis=1
        )

    return av_dff


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


def average_responding_stats_old(stats, responding_mask):
    num_regions = responding_mask.shape[0]
    av_per_region_stats = []
    for stat in stats:
        av_stat = np.zeros(num_regions)
        for reg_num in range(num_regions):
            av_stat[reg_num] = np.mean(stat[reg_num, responding_mask[reg_num]])

        av_per_region_stats.append(av_stat)

    return av_per_region_stats


def average_responding_stats(stats, responding_mask):
    num_regions = len(responding_mask)
    av_per_region_stats = []
    for stat in stats:
        av_stat = np.zeros(num_regions)
        for reg_num in range(num_regions):
            av_stat[reg_num] = np.mean(stat[reg_num][responding_mask[reg_num]])

        av_per_region_stats.append(av_stat)

    return av_per_region_stats


def calc_peak_bl_amp_old(dff_reg, fs):
    peak = np.amax(dff_reg[int(PRE_EVENT_T * fs) : int(POST_EVENT_T_PEAK * fs)], axis=0)
    bl = np.mean(dff_reg[: int(PRE_EVENT_T * fs)], axis=0)
    amp = peak - bl
    return peak, bl, amp


def calc_peak_bl_amp(dff_reg, fs):
    n_regions = len(dff_reg)
    peak, bl, amp = [], [], []
    for reg_num in range(n_regions):
        peak.append(
            np.amax(
                dff_reg[reg_num][int(PRE_EVENT_T * fs) : int(POST_EVENT_T_PEAK * fs)],
                axis=0,
            )
        )
        bl.append(np.mean(dff_reg[reg_num][: int(PRE_EVENT_T * fs)], axis=0))
        amp.append(peak[-1] - bl[-1])
    return peak, bl, amp


def calc_t_onset(dff_reg, k, fs):
    num_regions = dff_reg.shape[1]
    t_onset = np.zeros(num_regions)
    for reg_num in range(num_regions):
        k_onset = k[: int((PRE_EVENT_T + STIM_DURATION + POST_STIM_TIME) * fs), reg_num]
        if not np.any(k_onset > CURVATURE_THRESHOLD):
            t_onset[reg_num] = np.nan
            continue

        ind_start = np.argmax(k_onset)
        t_onset[reg_num] = ind_start / fs - PRE_EVENT_T

    return t_onset


def low_pass_filter(dff_raw, fs, f_c, order):
    sos = signal.butter(order, f_c, btype="low", output="sos", fs=fs)
    dff_reg = signal.sosfiltfilt(sos, dff_raw, axis=0)
    return dff_reg


def remove_baseline(dff_reg_filt, fs, pre_event_t):
    return dff_reg_filt - np.mean(dff_reg_filt[: int(pre_event_t * fs)], axis=0)


def curvature(x, dt):
    """
    The first and last 3 samples are discarded by calculating derivatives.
    """
    dxdt = first_derivative(x, dt)
    d2xdt2 = second_derivative(x, dt)
    k = np.zeros(x.shape)

    pos_slope_mask = np.zeros(x.shape, dtype=bool)
    pos_slope_mask[3:-3] = dxdt > 0

    dxdt = dxdt[pos_slope_mask[3:-3]]
    d2xdt2 = d2xdt2[pos_slope_mask[3:-3]]
    k_temp = d2xdt2 * np.power((1 + np.power(dxdt, 2)), -3 / 2)

    k[pos_slope_mask] = k_temp
    k[k < 0] = 0
    return k


def propagation_dist(amp_reg):
    prop_dists = []
    prop_origins = []
    prop_dist = 0
    prop_origin = 0
    amp = amp_reg[0]
    for reg_num in range(1, N_REGIONS):
        if amp_reg[reg_num] > amp:
            prop_dist += 1

        else:
            prop_dists.append(prop_dist)
            prop_origins.append(prop_origin)
            prop_dist = 0
            prop_origin = reg_num

        amp = amp_reg[reg_num]

    prop_dists.append(prop_dist)
    prop_origins.append(prop_origin)

    max_dist = -np.inf
    argmax = 0
    ind = 0

    while ind < len(prop_dists):
        dist = prop_dists[ind]
        if dist > max_dist:
            max_dist = dist
            argmax = ind
            max_dist = dist

        ind += 1

    return prop_dists[argmax], prop_origins[argmax]


def realign(dff_av_reg, k, fs):
    buffer_samples = int(BUFFER_T * fs)
    pre_event_samples = int(PRE_EVENT_T * fs)
    post_event_samples = int(POST_EVENT_T * fs)
    k_onset_dist = k[
        buffer_samples : 2 * buffer_samples + pre_event_samples, DISTAL_REG
    ]
    if not np.any(k_onset_dist > CURVATURE_THRESHOLD + 10):
        ind_response_start = pre_event_samples

    else:
        ind_response_start = np.argmax(k_onset_dist > CURVATURE_THRESHOLD)

    ind_response_start += buffer_samples
    start_ind = ind_response_start - buffer_samples
    end_ind = ind_response_start + post_event_samples

    """ print(f"num_frames w buffer: {dff_av_reg.shape[0]}")
    print(f"(start_frame, end_frame): ({start_ind}, {end_ind})")
    print(f"num_frames wo buffer: {end_ind-start_ind}") """

    return dff_av_reg[start_ind:end_ind], k[start_ind:end_ind]


def calc_stats(dff_reg, reg_mean_pos, cfg):
    fs = cfg.volume_rate
    peak, bl, amp = calc_peak_bl_amp(dff_reg, fs)

    stats = [peak, bl, amp]

    responding_mask = get_responding_mask(amp, RATIO_INCLUDE)
    dff_av_reg = average_responding_dff(dff_reg, responding_mask)
    av_stats = average_responding_stats(stats, responding_mask)
    peak, bl, amp = av_stats[0], av_stats[1], av_stats[2]

    dff_reg_filt = low_pass_filter(
        dff_av_reg,
        cfg.volume_rate,
        FREQ_CUT_OFF,
        FILTER_ORDER,
    )
    dff_av_reg = remove_baseline(dff_av_reg, fs, PRE_EVENT_T)
    dff_reg_filt = remove_baseline(dff_reg_filt, fs, PRE_EVENT_T)
    k = curvature(dff_reg_filt, 1 / fs)

    dff_av_reg, k = realign(dff_av_reg, k, fs)

    t_onset = calc_t_onset(dff_reg_filt, k, fs)

    d_amp = amp[1:] - amp[:-1]
    d_t_onset = (t_onset[1:] - t_onset[:-1]) / np.absolute(
        reg_mean_pos[1:] - reg_mean_pos[:-1]
    )

    return peak, bl, amp, d_amp, d_t_onset


def separate_list(mask_list, var_list):
    pos_list = []
    neg_list = []
    for b, var in zip(mask_list, var_list):
        if b:
            pos_list.append(var)

        else:
            neg_list.append(var)

    return np.array(pos_list), np.array(neg_list)


def main():

    ptz_exps = []
    d_amps = []
    d_t_onsets = []
    pre_amps = []
    post_amps = []
    pre_peaks = []
    post_peaks = []
    pre_bl = []
    post_bl = []
    pre_region = []
    post_region = []

    for exp_name in tqdm(EXP_NAMES, desc="Experiment"):
        ptz_exp = "ptz" in exp_name.lower()

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

                if cell_length != CELL_LENGTH:
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

                        peak, bl, amp, d_amp, d_t_onset = calc_stats(
                            dff_reg,
                            reg_mean_pos,
                            cfg,
                        )

                        for conn_num in range(N_REGIONS - 1):
                            ptz_exps.append(ptz_exp)
                            d_amps.append(d_amp[conn_num])
                            d_t_onsets.append(d_t_onset[conn_num])
                            pre_amps.append(amp[conn_num])
                            post_amps.append(amp[conn_num + 1])
                            pre_peaks.append(peak[conn_num])
                            post_peaks.append(peak[conn_num + 1])
                            pre_bl.append(bl[conn_num])
                            post_bl.append(bl[conn_num + 1])
                            pre_region.append(conn_num)
                            post_region.append(conn_num + 1)

    d_amp_ptz, d_amp_ctrl = separate_list(ptz_exps, d_amps)
    d_t_onset_ptz, d_t_onset_ctrl = separate_list(ptz_exps, d_t_onsets)
    post_amp_ptz, post_amp_ctrl = separate_list(ptz_exps, post_amps)
    pre_peak_ptz, pre_peak_ctrl = separate_list(ptz_exps, pre_peaks)
    pre_amp_ptz, pre_amp_ctrl = separate_list(ptz_exps, pre_amps)
    pre_bl_ptz, pre_bl_ctrl = separate_list(ptz_exps, pre_bl)
    pre_region_ptz, pre_region_ctrl = separate_list(ptz_exps, pre_region)

    min_post_amp = min(np.amin(post_amp_ctrl), np.amin(post_amp_ptz))
    max_post_amp = max(np.amax(post_amp_ctrl), np.amax(post_amp_ptz))

    s_max = 20
    s_min = 1
    dot_size_ptz = ((post_amp_ptz - min_post_amp) / (max_post_amp - min_post_amp)) * (
        s_max - s_min
    ) + s_min
    dot_size_ctrl = ((post_amp_ctrl - min_post_amp) / (max_post_amp - min_post_amp)) * (
        s_max - s_min
    ) + s_min

    plt.figure()
    plt.scatter(d_t_onset_ctrl, d_amp_ctrl, s=dot_size_ctrl, color="darkgray")
    plt.scatter(d_t_onset_ptz, d_amp_ptz, s=dot_size_ptz, color="brown")
    plt.xlabel("d_t_onset per micrometer")
    plt.ylabel("amplitude difference")

    plt.figure()
    plt.scatter(pre_bl_ctrl, d_amp_ctrl, s=dot_size_ctrl, color="darkgray")
    plt.scatter(pre_bl_ptz, d_amp_ptz, s=dot_size_ptz, color="brown")
    plt.xlabel("bl distal-most region")
    plt.ylabel("amplitude difference")

    plt.figure()
    plt.hist2d(pre_peak_ctrl, d_amp_ctrl, 50)
    plt.xlabel("amplitude distal-most region")
    plt.ylabel("amplitude difference")
    plt.title("ctrl")

    plt.figure()
    plt.hist2d(pre_amp_ptz, d_amp_ptz, 50)
    plt.xlabel("amplitude distal-most region")
    plt.ylabel("amplitude difference")
    plt.title("ptz")

    plt.figure()
    plt.scatter(pre_region_ctrl, d_amp_ctrl, s=dot_size_ctrl, color="darkgray")
    plt.scatter(pre_region_ptz, d_amp_ptz, s=dot_size_ptz, color="brown")
    plt.grid(axis="y")
    plt.xlabel("region number distal-most")
    plt.ylabel("amplitude difference")

    plt.show()


if __name__ == "__main__":
    main()
