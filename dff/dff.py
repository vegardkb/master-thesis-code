import numpy as np
from scipy import ndimage
from tqdm import tqdm
import threading
from enum import Enum

from fio import (
    load_cfg,
    load_s2p_df_rois,
    generate_exp_dir,
    generate_denoised_dir,
    generate_results_dir,
    generate_roi_dff_dir,
    gen_npy_fname,
    load_protocol,
)
from sigproc import (
    get_pos,
    calc_c_pos,
    calc_pos_pca0,
)

EXP_NAMES = [
    "20211112_13_23_43_GFAP_GCamp6s_F2_PTZ",
    "20211112_18_30_27_GFAP_GCamp6s_F5_c2",
    "20211112_19_48_54_GFAP_GCamp6s_F6_c3",
    "20211117_21_31_08_GFAP_GCamp6s_F6_PTZ",
    "20211119_16_36_20_GFAP_GCamp6s_F4_PTZ",
    "20220211_13_18_56_GFAP_GCamp6s_F2_C",
    "20220211_16_51_15_GFAP_GCamp6s_F4_PTZ",
    "20220412_12_32_27_GFAP_GCamp6s_F2_PTZ",
    "20220412_13_59_55_GFAP_GCamp6s_F3_C",
]

""" EXP_NAMES = [
    # "20220604_13_23_11_HuC_GCamp6s_GFAP_jRGECO_F1_C",
    "20220604_15_00_04_HuC_GCamp6s_GFAP_jRGECO_F2_PTZ",
    # "20220604_16_24_03_HuC_GCamp6s_GFAP_jRGECO_F3_PTZ",
] """
CROP_ID = "OpticTectum"
ROIS_FNAME = "rois"
PROTOCOL_NAME = "GFAP;Gcamp6s2021_11_12_data"
ACTIVITY_FNAME = "dff_stat"

USE_DENOISED = True
USE_CHAN2 = False
USE_LIGHT_PROTOCOL = True


FRAMES_PRE_POST_DENOISE = 30
FRAMES_DROPPED_DENOISE = 1 + 2 * FRAMES_PRE_POST_DENOISE
PRE_EVENT_T = 5
N_T_START_CROP = 2000
# N_T_START_CROP = 0
N_T_START_DISCARD = 0

CORRECTION_VOLUME_RATE = 30 / 5

FILTER_DYN = ndimage.uniform_filter1d
FILTER_STAT = np.percentile
FILTER_PARAMETER = 20
N_T_FILT_DYN = 1000
N_T_FILT_STAT = 860 - FRAMES_PRE_POST_DENOISE

MICROM_PER_M = 1000000
CROP_CELL = False
CELL_LENGTH = 55
AXIAL_BIN_SIZE = 0.5

N_REGIONS_LIST = [3, 6, 9]


class EventType(Enum):
    ACTIVITY_PEAK = 0
    LIGHT_ONSET = 1


def crop_cell(f_c, c_pos, pos_pca0):
    max_range = np.amax(pos_pca0)
    min_range = max_range - CELL_LENGTH
    if min_range < np.amin(pos_pca0):
        return False, None, None, None

    mask = np.logical_and(min_range <= pos_pca0, pos_pca0 <= max_range)
    f_c = f_c[:, mask]
    c_pos = c_pos[:, mask]
    pos_pca0 = pos_pca0[mask]
    return True, f_c, c_pos, pos_pca0


def extract_events_light_stimulus_events(
    protocol,
    fs,
    n_frames,
    n_t_start_discard,
    n_t_start_crop,
    pre_event_samples,
    correction_volume_rate,
):
    stim_type = "t_onset_light"
    evt_start = []
    evt_end = []
    evt_type = []
    t_first_stim = protocol[stim_type][0] * correction_volume_rate / fs
    isis_num = [(300, 5), (120, 5), (60, 5), (30, 5), (15, 4)]
    bl_t = 5

    t_stim = [t_first_stim]

    for isi, num in isis_num:
        for _ in range(num):
            t_stim.append(t_stim[-1] + isi * 0.999)

    post_event_t = [120, 120, 60, 30, 15]
    post_event_frames = [
        int((post_event_t[i] - bl_t) * fs) for i in range(len(post_event_t))
    ]

    t_stim = np.array(t_stim)
    frame_stim = np.round(t_stim * fs) - n_t_start_discard - n_t_start_crop
    for i, frame in enumerate(frame_stim):
        if frame < 0:
            continue
        evt_start.append(int(frame))
        evt_end.append(int(frame) + post_event_frames[i // 5])
        evt_type.append(1)

    evt_start = np.array(evt_start)
    evt_end = np.array(evt_end)
    evt_type = np.array(evt_type)

    good = np.logical_and(
        evt_start > pre_event_samples,
        evt_end < n_frames,
    )

    return evt_start[good], evt_end[good], evt_type[good]


def light_response(dff_stat, fs, pre_event_samples):
    protocol = load_protocol(PROTOCOL_NAME)
    (event_indices, event_ends, event_types,) = extract_events_light_stimulus_events(
        protocol,
        fs,
        dff_stat.shape[0],
        N_T_START_DISCARD,
        N_T_START_CROP,
        pre_event_samples,
        CORRECTION_VOLUME_RATE,
    )

    evt_type = EventType.LIGHT_ONSET

    evt_ind = event_indices[event_types == evt_type.value]
    evt_end = event_ends[event_types == evt_type.value]
    dff_evt_dict = calc_dff_evt_stim(
        dff_stat,
        evt_ind,
        evt_end,
        pre_event_samples,
        dff_stat.shape[1],
    )
    return dff_evt_dict


def calc_dff_stat(f, n_t, filter_func, parameter=None):
    if parameter is not None:
        f0 = filter_func(f[:n_t], parameter, axis=0)
    else:
        f0 = filter_func(f[:n_t], axis=0)
    f_filt = np.ones(f.shape) * f0
    dff = (f - f0) / f0

    return dff, f_filt


def calc_dff_evt_stim(dff, event_indices, event_ends, pre_event_samples, n_pix_cell):
    dff = dff[:, :n_pix_cell]

    isi_num = [(300, 5), (120, 5), (60, 5), (30, 5), (15, 5)]
    dff_evts = {}

    isi_num_index = 0
    event_counter = 0
    for event_ind, event_end in zip(event_indices, event_ends):
        if event_counter == 0:
            dff_evt = np.zeros(
                (
                    isi_num[isi_num_index][1],
                    pre_event_samples + (event_end - event_ind),
                    n_pix_cell,
                )
            )

        evt_start = event_ind - pre_event_samples

        dff_evt[event_counter] = dff[evt_start:event_end]

        event_counter = event_counter + 1
        if event_counter == isi_num[isi_num_index][1]:
            dff_evts[str(isi_num[isi_num_index][0])] = dff_evt
            event_counter = 0
            isi_num_index = isi_num_index + 1

    return dff_evts


def get_reg_mask(pos_pca0, n_regions):
    """
    Divide into regions by position along proximal-distal axis, equal length of regions
    """

    thr = np.linspace(
        np.amin(pos_pca0) - 1e-10, np.amax(pos_pca0) + 1e-10, n_regions + 1
    )

    mask = []
    for i in range(n_regions):
        mask.append(np.logical_and(pos_pca0 > thr[i], pos_pca0 <= thr[i + 1]))

    return np.array(mask)


def get_reg_activity(reg_mask, x, max_dyn_range=False):
    """
    reg_mask: 2d boolean ndarray (n_regions, pix)
    x: 3d ndarray (evt_num, t, pix)

    return:
     - y: 3d ndarray (evt_num, t, n_regions)
    """
    num_evt = x.shape[0]
    num_frames = x.shape[1]
    num_reg = reg_mask.shape[0]
    y = np.zeros((num_evt, num_frames, num_reg))
    for reg_num in range(reg_mask.shape[0]):
        x_reg = x[:, :, reg_mask[reg_num]]
        if max_dyn_range:
            dyn_range = np.amax(x_reg, axis=1) - np.amin(x_reg, axis=1)
            for evt_num in range(num_evt):
                max_ind = np.argmax(dyn_range[evt_num])
                y[evt_num, :, reg_num] = x_reg[evt_num, :, max_ind]

        else:
            y[:, :, reg_num] = np.mean(x_reg, axis=2)

    return y


def light_response_regions(dff_evt_dict, output_dir, pos_pd, c_pos):
    for isi, dff_evts in dff_evt_dict.items():
        for n_reg in N_REGIONS_LIST:
            region_mask = get_reg_mask(pos_pd, n_reg)

            num_regions_cell = region_mask.shape[0]
            reg_mean_pos_2d = np.zeros((2, num_regions_cell))
            for reg_num in range(num_regions_cell):
                reg_pos = c_pos[:, region_mask[reg_num]]
                reg_mean_pos_2d[:, reg_num] = np.mean(reg_pos, axis=1)

            reg_mean_pos = np.zeros(num_regions_cell)
            for reg_num in range(num_regions_cell):
                reg_pos = pos_pd[region_mask[reg_num]]
                reg_mean_pos[reg_num] = np.mean(reg_pos)

            dff_reg = get_reg_activity(region_mask, dff_evts)

            np.save(
                gen_npy_fname(
                    output_dir,
                    f"{ACTIVITY_FNAME}_{n_reg}_regions_light_response_ISI_{isi}",
                ),
                dff_reg,
            )
            np.save(
                gen_npy_fname(
                    output_dir,
                    f"regpos_{n_reg}_regions",
                ),
                reg_mean_pos,
            )
            np.save(
                gen_npy_fname(
                    output_dir,
                    f"regpos2d_{n_reg}_regions",
                ),
                reg_mean_pos_2d,
            )


def process_exp(exp_name):
    exp_dir = generate_exp_dir(exp_name, CROP_ID)
    cfg = load_cfg(exp_dir)

    if USE_DENOISED:
        exp_dir = generate_denoised_dir(exp_dir, USE_CHAN2)

    s2p_df, rois = load_s2p_df_rois(exp_dir, cfg, ROIS_FNAME)

    n_rois = len(rois)
    pos = get_pos(cfg.Ly, cfg.Lx)

    f = s2p_df["x"]
    n_frames = f.shape[0]

    output_dir = generate_results_dir(exp_dir)

    av_dff_all = np.zeros((n_frames - N_T_START_DISCARD, n_rois))
    fs = cfg.volume_rate

    pre_event_samples = int(PRE_EVENT_T * fs)

    for roi_num in tqdm(range(n_rois), desc="Roi"):
        roi_num_str = str(roi_num)
        roi = rois[roi_num]
        f_c = f[N_T_START_DISCARD:, roi]
        c_pos = calc_c_pos(roi, pos, cfg.Ly)
        pos_pd = calc_pos_pca0(c_pos, cfg, MICROM_PER_M)

        if CROP_CELL:
            success, f_c, c_pos, pos_pd = crop_cell(f_c, c_pos, pos_pd)
            if not success:
                continue

        dff_stat, _ = calc_dff_stat(f_c, N_T_FILT_STAT, FILTER_STAT, FILTER_PARAMETER)

        output_dir = generate_roi_dff_dir(exp_dir, roi_num_str, ROIS_FNAME)

        if USE_LIGHT_PROTOCOL:
            dff_evt_dict = light_response(dff_stat, fs, pre_event_samples)
            light_response_regions(dff_evt_dict, output_dir, pos_pd, c_pos)

        np.save(gen_npy_fname(output_dir, ACTIVITY_FNAME), dff_stat)
        np.save(gen_npy_fname(output_dir, "f"), f_c)

        av_dff_all[:, roi_num] = np.mean(dff_stat, axis=1)

    output_dir = generate_results_dir(exp_dir)


def main():

    threads = []

    for exp_name in EXP_NAMES:
        t = threading.Thread(target=process_exp, args=(exp_name,))
        t.start()
        threads.append(t)

    for t in tqdm(threads, desc="experiments"):
        t.join()


if __name__ == "__main__":
    main()
