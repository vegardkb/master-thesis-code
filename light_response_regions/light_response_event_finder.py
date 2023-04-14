from tqdm import tqdm
import numpy as np
import matplotlib.cm as cm
from enum import Enum
import threading

from fio import (
    load_cfg,
    load_custom_rois,
    load_protocol,
    generate_exp_dir,
    generate_denoised_dir,
    generate_roi_dff_dir,
    gen_npy_fname,
    get_t,
)
from sigproc import (
    extract_events_light_stimulus_events,
    calc_dff_evt_stim,
)

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
ACTIVITY_FNAME = "dff_stat"

PROTOCOL_NAME = "GFAP;Gcamp6s2021_11_12_data"

CORRECTION_VOLUME_RATE = 30 / 5

MICROM_PER_M = 1000000

BUFFER_T = 5
PRE_EVENT_T = 5

N_T_START_CROP = 2000
N_T_START_DISCARD = 0

N_REGIONS = 6
DISTAL_REG = 0
PROXIMAL_REG = N_REGIONS - 1


class EventType(Enum):
    ACTIVITY_PEAK = 0
    LIGHT_ONSET = 1


EVENT_TYPES = [EventType.LIGHT_ONSET]
EVENT_COLORS = ["orange"]

POS_CM = cm.viridis_r

USE_DENOISED = True
USE_CHAN2 = False
FRAMES_PRE_POST_DENOISE = 30
FRAMES_DROPPED_DENOISE = 1 + 2 * FRAMES_PRE_POST_DENOISE


def extract_z_evts_roi(
    exp_dir,
    roi_num_str,
    event_indices,
    event_ends,
    event_types,
    pre_event_samples,
    buffer_samples,
):
    dff_dir = generate_roi_dff_dir(exp_dir, roi_num_str, ROIS_FNAME)
    dff = np.load(gen_npy_fname(dff_dir, ACTIVITY_FNAME))

    for e in EVENT_TYPES:
        evt_ind = event_indices[event_types == e.value]
        evt_end = event_ends[event_types == e.value]
        z_evts = calc_dff_evt_stim(
            dff,
            evt_ind,
            evt_end,
            pre_event_samples,
            buffer_samples,
            dff.shape[1],
        )

        for isi, z_evt in z_evts.items():
            np.save(
                gen_npy_fname(dff_dir, f"{ACTIVITY_FNAME}_evt_{e}_ISI_{isi}"), z_evt
            )


def main():

    for exp_name in tqdm(EXP_NAMES, desc="Experiment"):
        exp_dir = generate_exp_dir(exp_name, CROP_ID)
        cfg = load_cfg(exp_dir)

        real_volume_rate = cfg.volume_rate

        if USE_DENOISED:
            exp_dir = generate_denoised_dir(exp_dir, USE_CHAN2)

        rois = load_custom_rois(exp_dir, ROIS_FNAME)
        n_rois = len(rois)

        protocol = load_protocol(PROTOCOL_NAME)

        PRE_EVENT_SAMPLES = int(PRE_EVENT_T * real_volume_rate)
        BUFFER_SAMPLES = int(BUFFER_T * real_volume_rate)
        t = get_t(real_volume_rate, cfg.n_frames, N_T_START_DISCARD, N_T_START_CROP)
        """ if USE_DENOISED:
            t = t[:-FRAMES_DROPPED_DENOISE] """

        (
            event_indices,
            event_ends,
            event_types,
        ) = extract_events_light_stimulus_events(
            protocol,
            real_volume_rate,
            t.shape[0],
            N_T_START_DISCARD,
            N_T_START_CROP,
            PRE_EVENT_SAMPLES,
            BUFFER_SAMPLES,
            CORRECTION_VOLUME_RATE,
        )

        threads = []

        for roi_num in range(n_rois):
            roi_num_str = str(roi_num)
            t = threading.Thread(
                target=extract_z_evts_roi,
                args=(
                    exp_dir,
                    roi_num_str,
                    event_indices,
                    event_ends,
                    event_types,
                    PRE_EVENT_SAMPLES,
                    BUFFER_SAMPLES,
                ),
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()


if __name__ == "__main__":
    main()
