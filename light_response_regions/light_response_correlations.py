from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from scipy.stats import pearsonr

from fio import (
    load_cfg,
    load_custom_rois,
    generate_exp_dir,
    generate_denoised_dir,
    generate_roi_dff_dir,
    gen_npy_fname,
)
from sigproc import (
    get_pos,
    calc_c_pos,
    calc_pos_pca0,
    get_reg_mask_pd_old,
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
N_REGIONS = 6
DISTAL_REG = 0
PROXIMAL_REG = N_REGIONS - 1
PRE_EVENT_T = 5
POST_EVENT_T = 20
POST_STIM_TIME = 10
STIM_DURATION = 10 + POST_STIM_TIME

DECAY_THRESHOLD = 0.5
RATIO_INCLUDE = 0.2
CURVATURE_THRESHOLD = 1e-6

EVENT_TYPE = EventType.LIGHT_ONSET

USE_DENOISED = True
USE_CHAN2 = False

# ISIS = [300, 120, 60, 30, 15]
ISIS = [300]

REG_CM = cm.viridis_r


def calc_inter_trial_correlation(dff_reg_evts):
    num_events = dff_reg_evts.shape[0]
    num_regions = dff_reg_evts.shape[2]

    corr_reg = np.zeros(num_regions)
    for region_num in range(num_regions):
        dff_evts = dff_reg_evts[:, :, region_num]

        n = 0
        s = 0
        for event_num1 in range(num_events):
            for event_num2 in range(event_num1):
                statistic, _ = pearsonr(dff_evts[event_num1], dff_evts[event_num2])
                s += statistic
                n += 1

        corr_reg[region_num] = s / n

    return corr_reg


def main():

    ptz_exps = []
    it_corr_regs = []

    for exp_name in tqdm(EXP_NAMES, desc="Experiment"):
        ptz_exp = "ptz" in exp_name.lower()

        for crop_id in tqdm(CROP_IDS, desc="Crop id", leave=False):
            exp_dir = generate_exp_dir(exp_name, crop_id)
            cfg = load_cfg(exp_dir)
            fs = cfg.volume_rate

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

                pos_pd = pos_pca0
                region_mask = get_reg_mask_pd_old(pos_pd, N_REGIONS)

                dff_dir = generate_roi_dff_dir(exp_dir, roi_num_str, ROIS_FNAME)

                evt_type = EVENT_TYPE
                isi = ISIS[0]

                dff_evts = np.load(
                    gen_npy_fname(dff_dir, f"dff_evt_{evt_type}_ISI_{isi}")
                )
                dff_evts = dff_evts[:, : int(fs * (PRE_EVENT_T + POST_EVENT_T))]

                num_evts = dff_evts.shape[0]
                dff_reg_evts = []

                for evt_num in range(num_evts):
                    dff_reg_evts.append(
                        np.mean(
                            get_reg_activity(region_mask, dff_evts[evt_num]), axis=2
                        )
                    )

                dff_reg_evts = np.array(dff_reg_evts)

                it_corr_reg = calc_inter_trial_correlation(dff_reg_evts)

                ptz_exps.append(ptz_exp)
                it_corr_regs.append(it_corr_reg)

    ptz_mask = np.array(ptz_exps)
    ctrl_mask = np.logical_not(ptz_mask)
    it_corr = np.array(it_corr_regs)

    reg_colors = REG_CM(np.linspace(0, 1, N_REGIONS))
    plt.figure()
    dist = 1
    width = 0.5
    for reg_num in range(it_corr.shape[1]):

        y = it_corr[ptz_mask, reg_num]

        num_points = len(y)
        x = np.linspace(reg_num * dist, reg_num * dist + width, num_points)
        plt.scatter(x, y, color=reg_colors[reg_num], s=2)

    plt.xlabel("Region")
    plt.ylabel("Intertrial correlation")
    plt.title("Reliability by region ptz")

    plt.figure()
    for reg_num in range(it_corr.shape[1]):

        y = it_corr[ctrl_mask, reg_num]

        num_points = len(y)
        x = np.linspace(reg_num * dist, reg_num * dist + width, num_points)
        plt.scatter(x, y, color=reg_colors[reg_num], s=2)

    plt.xlabel("Region")
    plt.ylabel("Intertrial correlation")
    plt.title("Reliability by region ctrl")

    plt.figure()
    dist = 1
    width = 0.25
    for group_num, mask, group, color in zip(
        [2, 1], [ptz_mask, ctrl_mask], ["ptz", "ctrl"], ["brown", "darkgray"]
    ):
        y = np.mean(it_corr[mask], axis=1)
        num_points = len(y)
        x = np.linspace(group_num * dist, group_num * dist + width, num_points)

        plt.scatter(x, y, s=5, color=color, label=group)

    plt.title("Response reliability per cell")
    plt.ylabel("Correlation")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
