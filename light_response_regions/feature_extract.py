from tqdm import tqdm
import numpy as np
import copy
from scipy import signal
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import pandas as pd

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
from sigproc import get_pos, calc_c_pos, calc_pos_pca0, get_reg_mask

EXP_NAMES = [
    "20211112_13_23_43_GFAP_GCamp6s_F2_PTZ",
    "20211112_18_30_27_GFAP_GCamp6s_F5_c2",
    "20211112_19_48_54_GFAP_GCamp6s_F6_c3",
    "20211119_16_36_20_GFAP_GCamp6s_F4_PTZ",
    "20220211_13_18_56_GFAP_GCamp6s_F2_C",
    "20220211_16_51_15_GFAP_GCamp6s_F4_PTZ",
    "20220412_12_32_27_GFAP_GCamp6s_F2_PTZ",
    "20220412_13_59_55_GFAP_GCamp6s_F3_C",
]
CROP_IDS = ["OpticTectum"]
ROIS_FNAME = "rois"
ACTIVITY_FNAME = "dff_stat"

DFF_REGS_FNAME = "dff_light_response"
SECOND_DERIVATIVE_FNAME = "d2xdt2_light_response"
STATS_FNAME = "stats_light_response"
T_ONSET_FNAME = "t_onsets_light_response"
POS_REGS_FNAME = "pos_regs"

DISTAL_REG = 0
PRE_EVENT_T = 5
POST_EVENT_T_PEAK = 40
POST_STIM_T_LAG = 10
POST_STIM_T_EVENT = 5
STIM_DURATION = 10

DECAY_THRESHOLD = 0.8

USE_DENOISED = True
USE_CHAN2 = False

ISIS = [300]

MICROM_PER_M = 1000000

LOW_PASS_FILTER = True
FREQ_CUT_OFF = 0.375
FILTER_ORDER = 8

MERGE_PEAKS = True

N_REGIONS_LIST = [3, 6]

MAX_ROI_SIZE = 3500


def arp_model_fit(x, p):
    n = len(x)
    X = np.zeros((n - p, p))
    y = np.zeros(n - p)

    for t in range(n - p):
        X[t] = np.flip(x[t : t + p])
        y[t] = x[t + p]

    model = LinearRegression(fit_intercept=False).fit(X, y)
    phis = model.coef_

    return phis


def arp_model_pred(x, par):
    p = len(par)
    n = len(x)
    y = np.zeros(n)
    y[:p] = x[:p]
    for t in range(p, n):
        for i in range(p):
            y[t] += par[i] * x[t - 1 - i]

    return y


def arp_model_res(x, par):
    y = arp_model_pred(x, par)
    z = x - y
    return z


def create_column_names(num_regions):
    cols = [
        "exp_name",
        "crop_id",
        "roi_number",
        "isi",
        "ptz",
        "evt_num",
        "t_onset_slope",
        "t_onset_rsq",
    ]
    for reg_num in range(num_regions):
        cols = cols + [
            "peak" + f"_r{reg_num}",
            "peak5s" + f"_r{reg_num}",
            "peak15s" + f"_r{reg_num}",
            "bl" + f"_r{reg_num}",
            "amp" + f"_r{reg_num}",
            "amp5s" + f"_r{reg_num}",
            "amp15s" + f"_r{reg_num}",
            "t_peak" + f"_r{reg_num}",
            "t_decay" + f"_r{reg_num}",
            "t_constant_decay" + f"_r{reg_num}",
            "t_lag_dff" + f"_r{reg_num}",
            "corr_dff" + f"_r{reg_num}",
            "limit_dff" + f"_r{reg_num}",
            "t_lag_res" + f"_r{reg_num}",
            "corr_res" + f"_r{reg_num}",
            "limit_res" + f"_r{reg_num}",
            "t_onset" + f"_r{reg_num}",
            "diff_dxdt" + f"_r{reg_num}",
        ]
    return cols


def create_empty_results_dict(cols):
    results_dict = {}
    for col in cols:
        results_dict[col] = []

    return results_dict


def create_empty_t_onsets_dict():
    t_onsets_dict = {
        "exp_name": [],
        "crop_id": [],
        "roi_number": [],
        "isi": [],
        "ptz": [],
        "evt_num": [],
        "time_com": [],
        "time_start": [],
        "time_end": [],
        "diff_dxdt": [],
        "pre_dff": [],
        "pre_bl_sub_dff": [],
        "region": [],
    }
    return t_onsets_dict


def first_derivative(x, dt):
    return (x[1:-5] - 8 * x[2:-4] + 8 * x[4:-2] - x[5:-1]) / (12 * dt)


def second_derivative(x, dt):
    return (-x[1:-5] + 16 * x[2:-4] - 30 * x[3:-3] + 16 * x[4:-2] - x[5:-1]) / (
        12 * dt**2
    )


def low_pass_filter(dff_raw, fs, f_c, order):
    sos = signal.butter(order, f_c, btype="low", output="sos", fs=fs)
    dff_reg = signal.sosfiltfilt(sos, dff_raw, axis=0)
    return dff_reg


def time_constant(x, dt):
    y = copy.deepcopy(x)
    dxdt = first_derivative(y, dt)
    y = y[3:-3]

    dxdt[dxdt >= 0] = -1e-10
    y[y <= 0.2] = np.inf
    tau = -y / dxdt

    time_const = np.amin(tau, axis=0)

    return time_const


def calc_peak_bl_amp(dff_reg, fs):
    dff_bl = dff_reg[: int(PRE_EVENT_T * fs)]
    dff_peak = dff_reg[
        int(PRE_EVENT_T * fs) : int((PRE_EVENT_T + POST_EVENT_T_PEAK) * fs)
    ]

    bl = np.mean(dff_bl, axis=0)
    peak = np.amax(dff_peak, axis=0)
    dff5s = dff_reg[int((PRE_EVENT_T + 5) * fs)]
    dff15s = dff_reg[int((PRE_EVENT_T + 15) * fs)]
    neg = peak < bl

    if np.any(neg):
        peak[neg] = np.amin(dff_peak[:, neg], axis=0)
    amp = peak - bl
    amp5s = dff5s - bl
    amp15s = dff15s - bl

    stats, names = [], []
    for reg_num in range(dff_reg.shape[1]):
        stats.append(bl[reg_num]), names.append(f"bl_r{reg_num}")
        stats.append(peak[reg_num]), names.append(f"peak_r{reg_num}")
        stats.append(dff5s[reg_num]), names.append(f"peak5s_r{reg_num}")
        stats.append(dff15s[reg_num]), names.append(f"peak15s_r{reg_num}")
        stats.append(amp[reg_num]), names.append(f"amp_r{reg_num}")
        stats.append(amp5s[reg_num]), names.append(f"amp5s_r{reg_num}")
        stats.append(amp15s[reg_num]), names.append(f"amp15s_r{reg_num}")

    return stats, names


def subtract_baseline(dff_reg, fs):
    dff_bl = dff_reg[: int(PRE_EVENT_T * fs)]
    bl_sub_dff = dff_reg - np.mean(dff_bl, axis=0)
    return bl_sub_dff


def get_region_pos(roi, cfg, num_reg):
    pos = get_pos(cfg.Ly, cfg.Lx)
    c_pos = calc_c_pos(roi, pos, cfg.Ly)
    pos_pca0 = calc_pos_pca0(c_pos, cfg, MICROM_PER_M)
    region_mask = get_reg_mask(pos_pca0, num_reg)
    region_pos = (
        np.zeros((pos.shape[0], region_mask.shape[0], region_mask.shape[1])) - 1
    )
    region_pos_z = np.zeros((pos.shape[0], region_mask.shape[0], MAX_ROI_SIZE)) - 1
    for reg_num in range(region_mask.shape[0]):
        region_pos[:, reg_num, region_mask[reg_num]] = c_pos[:, region_mask[reg_num]]

    region_pos_z[:, :, : region_mask.shape[1]] = region_pos
    return region_pos_z


def calc_ts(dff_reg, fs):
    num_regions = dff_reg.shape[1]
    t_stim = np.ones(num_regions) * PRE_EVENT_T
    ind_peak = np.argmax(
        dff_reg[int(PRE_EVENT_T * fs) : int((PRE_EVENT_T + POST_EVENT_T_PEAK) * fs)],
        axis=0,
    )
    t_peak = (ind_peak + int(PRE_EVENT_T * fs)) / fs - t_stim
    t_constant_decay = time_constant(dff_reg, 1 / fs)

    t_decay = np.zeros(num_regions)
    for reg_num in range(num_regions):
        peak = np.amax(
            dff_reg[
                int(PRE_EVENT_T * fs) : int((PRE_EVENT_T + POST_EVENT_T_PEAK) * fs),
                reg_num,
            ],
            axis=0,
        )
        ind_decay = ind_peak[reg_num] + int(PRE_EVENT_T * fs)
        while (
            dff_reg[ind_decay, reg_num] > DECAY_THRESHOLD * peak
            and ind_decay < dff_reg.shape[0] - 1
        ):
            ind_decay += 1

        t_decay[reg_num] = (ind_decay - ind_peak[reg_num]) / fs

    stats, names = [], []
    for reg_num in range(dff_reg.shape[1]):
        stats.append(t_peak[reg_num]), names.append(f"t_peak_r{reg_num}")
        stats.append(t_decay[reg_num]), names.append(f"t_decay_r{reg_num}")
        stats.append(t_constant_decay[reg_num]), names.append(
            f"t_constant_decay_r{reg_num}"
        )

    return stats, names


def calc_lag_corr(x, x_ref, fs, max_lag=10):
    x_norm = (x - np.mean(x)) / np.std(x)
    x_ref_norm = (x_ref - np.mean(x_ref)) / np.std(x_ref)
    n = x.shape[0]
    corr = signal.correlate(x_norm, x_ref_norm, mode="same") / n

    lags = np.arange(n) / fs
    lags = lags - np.amax(lags) / 2

    lag_mask = np.absolute(lags) < max_lag
    lags = lags[lag_mask]
    corr = corr[lag_mask]

    lag_ind = np.argmax(corr)

    return lags[lag_ind], corr[lag_ind], 2.33 / np.sqrt(n)  # 1.96 - 5%, 2.33 - 1%


def calc_t_lag(dff_reg, fs):
    num_regions = dff_reg.shape[1]

    dff_distal_whole = dff_reg[:, DISTAL_REG]
    p = 10
    phis = arp_model_fit(dff_distal_whole, p)

    dff_onset = dff_reg[: int((PRE_EVENT_T + STIM_DURATION + POST_STIM_T_LAG) * fs)]

    dff_onset_res = np.array(
        [arp_model_res(dff_onset[:, reg_num], phis) for reg_num in range(num_regions)]
    ).T

    dff_distal_res = dff_onset_res[p:, DISTAL_REG]

    dff_distal = dff_onset[:, DISTAL_REG]

    corr_dff, t_lag_dff, limit_dff = [], [], []
    corr_res, t_lag_res, limit_res = [], [], []

    for reg_num in range(num_regions):
        dff = dff_onset[:, reg_num]
        dff_res = dff_onset_res[:, reg_num]

        lag, corr, lim = calc_lag_corr(dff, dff_distal, fs)
        t_lag_dff.append(lag), corr_dff.append(corr), limit_dff.append(lim)
        lag, corr, lim = calc_lag_corr(dff_res, dff_distal_res, fs)
        t_lag_res.append(lag), corr_res.append(corr), limit_res.append(lim)

    stats, names = [], []
    for reg_num in range(num_regions):
        stats.append(corr_dff[reg_num]), names.append(f"corr_dff_r{reg_num}")
        stats.append(t_lag_dff[reg_num]), names.append(f"t_lag_dff_r{reg_num}")
        stats.append(limit_dff[reg_num]), names.append(f"limit_dff_r{reg_num}")

        stats.append(corr_res[reg_num]), names.append(f"corr_res_r{reg_num}")
        stats.append(t_lag_res[reg_num]), names.append(f"t_lag_res_r{reg_num}")
        stats.append(limit_res[reg_num]), names.append(f"limit_res_r{reg_num}")

    return stats, names


def merge_peaks(peaks, d2xdt2_onset, merge):
    midpoint_list = []
    if len(peaks) > 1 and merge:
        peaks_filt = peaks.tolist()
        merge_list = []

        for ind1, ind2 in zip(peaks[:-1], peaks[1:]):
            x_between = d2xdt2_onset[ind1 : ind2 + 1]
            midpoint = np.argmin(x_between)
            midpoint_list.append(midpoint + ind1)

            if np.any(x_between < 0):
                merge_list.append(False)
            else:
                merge_list.append(True)

        i = 0
        while i < len(merge_list):
            ind1 = peaks_filt[i]
            ind2 = peaks_filt[i + 1]
            merge = merge_list[i]

            if merge:
                merge_list.pop(i)
                midpoint_list.pop(i)
                peaks_filt.pop(i)

                if d2xdt2_onset[ind1] >= d2xdt2_onset[ind2]:
                    peaks_filt[i] = ind1
                else:
                    peaks_filt[i] = ind2

                i = 0
            else:
                i += 1

        peaks = np.array(peaks_filt)

    return peaks, midpoint_list


def calc_t_onsets(d2xdt2, dff_reg, reg_mean_pos, fs):
    t_onset_dict = {
        "time_com": [],
        "time_start": [],
        "time_end": [],
        "diff_dxdt": [],
        "pre_dff": [],
        "pre_bl_sub_dff": [],
        "region": [],
    }

    stats, names = [], []
    num_regions = d2xdt2.shape[1]

    dff_bl = dff_reg[: int(PRE_EVENT_T * fs)]

    bl_reg = np.mean(dff_bl, axis=0)

    t_onsets = np.zeros(num_regions)
    for reg_num in range(num_regions):
        d2xdt2_onset = d2xdt2[
            int(PRE_EVENT_T * fs / 2) : int(
                (PRE_EVENT_T + STIM_DURATION + POST_STIM_T_EVENT) * fs
            ),
            reg_num,
        ]
        dff_onset = dff_reg[
            int(PRE_EVENT_T * fs / 2) : int(
                (PRE_EVENT_T + STIM_DURATION + POST_STIM_T_EVENT) * fs
            ),
            reg_num,
        ]
        bl = bl_reg[reg_num]

        peaks, _ = signal.find_peaks(d2xdt2_onset, height=0.0001)
        peaks, midpoint_list = merge_peaks(peaks, d2xdt2_onset, MERGE_PEAKS)

        t_com = np.nan
        diff_dxdt = -np.inf

        for i, peak in enumerate(peaks):
            pre_mid = 0
            post_mid = d2xdt2_onset.shape[0] - 1

            if len(midpoint_list):
                if i == 0:
                    post_mid = midpoint_list[i]
                elif i == len(midpoint_list):
                    pre_mid = midpoint_list[i - 1]
                else:
                    pre_mid = midpoint_list[i - 1]
                    post_mid = midpoint_list[i]

            start_ind, end_ind = peak, peak
            while start_ind > pre_mid and d2xdt2_onset[start_ind - 1] > 0:
                start_ind = start_ind - 1

            while end_ind < post_mid and d2xdt2_onset[end_ind + 1] > 0:
                end_ind = end_ind + 1

            t_int = np.arange(start_ind, end_ind + 1) / fs - PRE_EVENT_T / 2
            x_int = d2xdt2_onset[start_ind : end_ind + 1]

            int_d2xdt2 = np.sum(x_int) / fs
            try:
                t_cmass = np.average(t_int, weights=x_int)
            except ZeroDivisionError:
                print(f"\n\n\n\n\nx_int: {x_int}")
                raise ZeroDivisionError

            t_onset_dict["time_com"].append(t_cmass)
            t_onset_dict["time_start"].append(np.amin(t_int))
            t_onset_dict["time_end"].append(np.amax(t_int))
            t_onset_dict["diff_dxdt"].append(int_d2xdt2)
            t_onset_dict["pre_dff"].append(dff_onset[start_ind])
            t_onset_dict["pre_bl_sub_dff"].append(dff_onset[start_ind] - bl)
            t_onset_dict["region"].append(reg_num)

            if int_d2xdt2 > diff_dxdt:
                diff_dxdt = int_d2xdt2
                t_com = t_cmass

        t_onsets[reg_num] = t_com
        stats.append(t_com), names.append(f"t_onset_r{reg_num}")
        stats.append(diff_dxdt), names.append(f"diff_dxdt_r{reg_num}")

    good = np.logical_not(np.logical_or(np.isnan(t_onsets), np.isinf(t_onsets)))

    linreg = linregress(reg_mean_pos[good], t_onsets[good])
    t_onset_slope = linreg.slope
    t_onset_rsq = np.power(linreg.rvalue, 2)

    stats.append(t_onset_slope), names.append("t_onset_slope")
    stats.append(t_onset_rsq), names.append("t_onset_rsq")

    return t_onset_dict, stats, names


def calc_stats(dff_reg_filt, fs):
    stats = []
    names = []

    stat_list, name_list = calc_peak_bl_amp(dff_reg_filt, fs)
    stats += stat_list
    names += name_list

    stat_list, name_list = calc_ts(dff_reg_filt, fs)
    stats += stat_list
    names += name_list

    stat_list, name_list = calc_t_lag(dff_reg_filt, fs)
    stats += stat_list
    names += name_list

    return stats, names


def dump_meta_stats(result_dict, exp_name, crop_id, roi_num, isi, ptz_exp, evt_num):
    key = list(result_dict.keys())[0]
    num_elements = len(result_dict[key])
    result_dict["exp_name"] = [exp_name for _ in range(num_elements)]
    result_dict["crop_id"] = [crop_id for _ in range(num_elements)]
    result_dict["roi_number"] = [roi_num for _ in range(num_elements)]
    result_dict["isi"] = [isi for _ in range(num_elements)]
    result_dict["ptz"] = [ptz_exp for _ in range(num_elements)]
    result_dict["evt_num"] = [evt_num for _ in range(num_elements)]

    return result_dict


def dump_stats(stats, names, exp_name, crop_id, roi_num, isi, ptz, evt_num):
    result_dict = {}
    for stat, name in zip(stats, names):
        result_dict[name] = [stat]

    result_dict = dump_meta_stats(
        result_dict, exp_name, crop_id, roi_num, isi, ptz, evt_num
    )
    return result_dict


def concat_dicts(dict1, dict2):
    ret_dict = {}
    for key in dict1.keys():
        try:
            ret_dict[key] = dict1[key] + dict2[key]
        except KeyError:
            print(f"key not used: {key}")

    return ret_dict


def process_roi(
    exp_name,
    crop_id,
    roi_num,
    isi,
    ptz,
    dff_dir,
    results_dict,
    dff_regs,
    d2xdt2s,
    t_onsets_dict,
    pos_regs,
    num_regions,
    fs,
    region_pos,
):

    try:
        dff_evts = np.load(
            gen_npy_fname(
                dff_dir,
                f"{ACTIVITY_FNAME}_{num_regions}_regions_light_response_ISI_{isi}",
            )
        )
        reg_mean_pos = np.load(
            gen_npy_fname(
                dff_dir,
                f"regpos_{num_regions}_regions",
            )
        )
    except FileNotFoundError:
        path = gen_npy_fname(
            dff_dir,
            f"{ACTIVITY_FNAME}_{num_regions}_regions_light_response_ISI_{isi}",
        )
        print(f"Warning: file not found {path}")
        return results_dict, dff_regs, d2xdt2s, t_onsets_dict, pos_regs

    num_evts = dff_evts.shape[0]

    for evt_num in tqdm(range(num_evts), desc="Event", leave=False):
        dff_reg = dff_evts[evt_num]

        if LOW_PASS_FILTER:
            dff_reg_filt = low_pass_filter(
                dff_reg,
                fs,
                FREQ_CUT_OFF,
                FILTER_ORDER,
            )
        else:
            dff_reg_filt = dff_reg
        d2xdt2 = np.zeros(dff_reg_filt.shape)
        d2xdt2[3:-3] = second_derivative(dff_reg_filt, 1 / fs)
        dxdt = np.zeros(dff_reg_filt.shape)
        dxdt[3:-3] = first_derivative(dff_reg_filt, 1 / fs)
        dff_regs.append(subtract_baseline(dff_reg_filt, fs))
        d2xdt2s.append(d2xdt2)

        t_onset_dict, stats_t_onset, names_t_onset = calc_t_onsets(
            d2xdt2, dff_reg, reg_mean_pos, fs
        )
        t_onset_dict = dump_meta_stats(
            t_onset_dict, exp_name, crop_id, roi_num, isi, ptz, evt_num
        )

        stats, names = calc_stats(dff_reg_filt, fs)
        stats += stats_t_onset
        names += names_t_onset

        result_dict = dump_stats(
            stats, names, exp_name, crop_id, roi_num, isi, ptz, evt_num
        )

        results_dict = concat_dicts(results_dict, result_dict)
        t_onsets_dict = concat_dicts(t_onsets_dict, t_onset_dict)

        pos_regs.append(region_pos)

    column_lengths = []
    for key in results_dict.keys():
        col_len = len(results_dict[key])
        column_lengths.append(col_len)
        if not col_len:
            print(f"Empty column: {key}")

    column_lengths = np.array(column_lengths)
    if not np.all(column_lengths == column_lengths[0]):
        print("Not all columns are same length")

    return results_dict, dff_regs, d2xdt2s, t_onsets_dict, pos_regs


def main():

    results_dir = generate_global_results_dir()
    exp_crop = []
    for exp_name in EXP_NAMES:
        for crop_id in CROP_IDS:
            exp_crop.append((exp_name, crop_id))

    for num_regions in tqdm(N_REGIONS_LIST, desc="num_regions"):
        cols = create_column_names(num_regions)
        results_dict = create_empty_results_dict(cols)
        dff_regs = []
        d2xdt2s = []
        t_onsets_dict = create_empty_t_onsets_dict()
        pos_regs = []
        for exp_name, crop_id in tqdm(exp_crop, desc="exp_crop", leave=False):
            ptz_exp = "ptz" in exp_name.lower()
            exp_dir = generate_exp_dir(exp_name, crop_id)
            cfg = load_cfg(exp_dir)
            fs = cfg.volume_rate

            if USE_DENOISED:
                exp_dir = generate_denoised_dir(exp_dir, USE_CHAN2)

            rois = load_custom_rois(exp_dir, ROIS_FNAME)
            n_rois = len(rois)

            for roi_num in tqdm(range(n_rois), desc="Roi", leave=False):
                region_pos = get_region_pos(rois[roi_num], cfg, num_regions)
                roi_num_str = str(roi_num)
                dff_dir = generate_roi_dff_dir(exp_dir, roi_num_str, ROIS_FNAME)

                for isi in ISIS:
                    (
                        results_dict,
                        dff_regs,
                        d2xdt2s,
                        t_onsets_dict,
                        pos_regs,
                    ) = process_roi(
                        exp_name,
                        crop_id,
                        roi_num,
                        isi,
                        ptz_exp,
                        dff_dir,
                        results_dict,
                        dff_regs,
                        d2xdt2s,
                        t_onsets_dict,
                        pos_regs,
                        num_regions,
                        fs,
                        region_pos,
                    )

        df_results = pd.DataFrame(results_dict)
        print(df_results.head())
        df_t_onset = pd.DataFrame(t_onsets_dict)
        dff_regs = np.array(dff_regs)
        d2xdt2s = np.array(d2xdt2s)
        pos_regs = np.array(pos_regs)

        df_results.to_pickle(
            gen_pickle_fname(results_dir, STATS_FNAME + f"_{num_regions}_regions")
        )
        df_t_onset.to_pickle(
            gen_pickle_fname(results_dir, T_ONSET_FNAME + f"_{num_regions}_regions")
        )
        np.save(
            gen_npy_fname(results_dir, DFF_REGS_FNAME + f"_{num_regions}_regions"),
            dff_regs,
        )
        np.save(
            gen_npy_fname(
                results_dir, SECOND_DERIVATIVE_FNAME + f"_{num_regions}_regions"
            ),
            d2xdt2s,
        )
        np.save(
            gen_npy_fname(results_dir, POS_REGS_FNAME + f"_{num_regions}_regions"),
            pos_regs,
        )


if __name__ == "__main__":
    main()
