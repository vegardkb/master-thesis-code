from tqdm import tqdm
import numpy as np
import copy
from scipy import signal
from scipy.stats import linregress, pearsonr
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
ROIS_FNAME = "rois_smooth"
ACTIVITY_FNAME = "dff_stat"

DFF_REGS_FNAME = "dff_light_response"
SECOND_DERIVATIVE_FNAME = "d2xdt2_light_response"
STATS_FNAME = "stats_light_response"
T_ONSET_FNAME = "t_onsets_light_response"
POS_REGS_FNAME = "pos_regs"

DISTAL_REG = 0
BUFFER_T = 5
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
CELL_LENGTH = 50

FREQ_CUT_OFF = 0.25
FILTER_ORDER = 8

N_REGIONS_LIST = [110]

MAX_ROI_SIZE = 3500


def create_column_names(num_regions):
    cols = [
        "exp_name",
        "crop_id",
        "roi_number",
        "isi",
        "ptz",
        "evt_num",
    ]
    for reg_num in range(num_regions):
        cols = cols + [
            "peak" + f"_r{reg_num}",
            "bl" + f"_r{reg_num}",
            "amp" + f"_r{reg_num}",
            "amp5s" + f"_r{reg_num}",
            "amprnd" + f"_r{reg_num}",
            "filt_bl" + f"_r{reg_num}",
            "filt_amp5s" + f"_r{reg_num}",
            "filt_amprnd" + f"_r{reg_num}",
            "t_peak" + f"_r{reg_num}",
        ]
    return cols


def create_empty_results_dict(cols):
    results_dict = {}
    for col in cols:
        results_dict[col] = []

    return results_dict


def first_derivative(x, dt):
    return (x[1:-5] - 8 * x[2:-4] + 8 * x[4:-2] - x[5:-1]) / (12 * dt)


def second_derivative(x, dt):
    return (-x[1:-5] + 16 * x[2:-4] - 30 * x[3:-3] + 16 * x[4:-2] - x[5:-1]) / (
        12 * dt**2
    )


def high_pass_filter(x, fs, f_c, order):
    sos = signal.butter(order, f_c, btype="high", output="sos", fs=fs)
    dff_reg = signal.sosfiltfilt(
        sos,
        x,
    )
    return dff_reg


def calc_peak_bl_amp(dff_reg, fs):
    dff_bl = dff_reg[: int(PRE_EVENT_T * fs)]
    dff_peak = dff_reg[
        int(PRE_EVENT_T * fs) : int((PRE_EVENT_T + POST_EVENT_T_PEAK) * fs)
    ]

    bl = np.mean(dff_bl, axis=0)
    peak = np.amax(dff_peak, axis=0)
    dff5s = dff_reg[int((PRE_EVENT_T + 5) * fs)]
    dffrnd = dff_reg[int((PRE_EVENT_T + 110) * fs)]
    neg = peak < bl

    if np.any(neg):
        peak[neg] = np.amin(dff_peak[:, neg], axis=0)
    amp = peak - bl
    amp5s = dff5s - bl
    amprnd = dffrnd - bl

    fs_space = 1 / (CELL_LENGTH / dff_reg.shape[1])  # Regions per micrometer
    fc_space = 1 / 10  # Cut off at 10 micrometer period
    f_order = 2

    filt_bl = high_pass_filter(bl, fs_space, fc_space, f_order)
    filt_amp5s = high_pass_filter(amp5s, fs_space, fc_space, f_order)
    filt_amprnd = high_pass_filter(amprnd, fs_space, fc_space, f_order)

    stats, names = [], []
    for reg_num in range(dff_reg.shape[1]):
        stats.append(peak[reg_num]), names.append(f"peak_r{reg_num}")
        stats.append(amp[reg_num]), names.append(f"amp_r{reg_num}")
        stats.append(bl[reg_num]), names.append(f"bl_r{reg_num}")
        stats.append(amp5s[reg_num]), names.append(f"amp5s_r{reg_num}")
        stats.append(amprnd[reg_num]), names.append(f"amprnd_r{reg_num}")
        stats.append(filt_bl[reg_num]), names.append(f"filt_bl_r{reg_num}")
        stats.append(filt_amp5s[reg_num]), names.append(f"filt_amp5s_r{reg_num}")
        stats.append(filt_amprnd[reg_num]), names.append(f"filt_amprnd_r{reg_num}")

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

    stats, names = [], []
    for reg_num in range(dff_reg.shape[1]):
        stats.append(t_peak[reg_num]), names.append(f"t_peak_r{reg_num}")

    return stats, names


def low_pass_filter(dff_raw, fs, f_c, order):
    sos = signal.butter(order, f_c, btype="low", output="sos", fs=fs)
    dff_reg = signal.sosfiltfilt(sos, dff_raw, axis=0)
    return dff_reg


def calc_stats(dff_reg_filt, fs):
    stats = []
    names = []

    stat_list, name_list = calc_peak_bl_amp(dff_reg_filt, fs)
    stats += stat_list
    names += name_list

    stat_list, name_list = calc_ts(dff_reg_filt, fs)
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
        return results_dict, dff_regs, d2xdt2s, pos_regs

    num_evts = dff_evts.shape[0]

    for evt_num in tqdm(range(num_evts), desc="Event", leave=False):
        dff_reg = dff_evts[evt_num]

        dff_reg_filt = low_pass_filter(
            dff_reg,
            fs,
            FREQ_CUT_OFF,
            FILTER_ORDER,
        )
        d2xdt2 = np.zeros(dff_reg_filt.shape)
        d2xdt2[3:-3] = second_derivative(dff_reg_filt, 1 / fs)
        dxdt = np.zeros(dff_reg_filt.shape)
        dxdt[3:-3] = first_derivative(dff_reg_filt, 1 / fs)
        dff_regs.append(subtract_baseline(dff_reg, fs))
        d2xdt2s.append(d2xdt2)

        stats, names = calc_stats(dff_reg_filt, fs)

        result_dict = dump_stats(
            stats, names, exp_name, crop_id, roi_num, isi, ptz, evt_num
        )

        results_dict = concat_dicts(results_dict, result_dict)

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

    return results_dict, dff_regs, d2xdt2s, pos_regs


def main():

    results_dir = generate_global_results_dir()
    exp_crop = []
    for exp_name in EXP_NAMES:
        for crop_id in CROP_IDS:
            exp_crop.append((exp_name, crop_id))
            print(exp_name)

    for num_regions in tqdm(N_REGIONS_LIST, desc="num_regions"):
        cols = create_column_names(num_regions)
        results_dict = create_empty_results_dict(cols)
        dff_regs = []
        d2xdt2s = []
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
                    (results_dict, dff_regs, d2xdt2s, pos_regs) = process_roi(
                        exp_name,
                        crop_id,
                        roi_num,
                        isi,
                        ptz_exp,
                        dff_dir,
                        results_dict,
                        dff_regs,
                        d2xdt2s,
                        pos_regs,
                        num_regions,
                        fs,
                        region_pos,
                    )

        df_results = pd.DataFrame(results_dict)
        print(df_results)
        dff_regs = np.array(dff_regs)
        d2xdt2s = np.array(d2xdt2s)
        pos_regs = np.array(pos_regs)

        df_results.to_pickle(
            gen_pickle_fname(results_dir, STATS_FNAME + f"_{num_regions}_regions")
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
