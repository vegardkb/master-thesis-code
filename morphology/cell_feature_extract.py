import numpy as np
from tqdm import tqdm
import pandas as pd

from fio import (
    load_cfg,
    load_custom_rois,
    generate_exp_dir,
    generate_denoised_dir,
    generate_global_results_dir,
    gen_pickle_fname,
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
    "20211119_16_36_20_GFAP_GCamp6s_F4_PTZ",
    "20220211_13_18_56_GFAP_GCamp6s_F2_C",
    "20220211_16_51_15_GFAP_GCamp6s_F4_PTZ",
    "20220412_12_32_27_GFAP_GCamp6s_F2_PTZ",
    "20220412_13_59_55_GFAP_GCamp6s_F3_C",
]
CROP_ID = "OpticTectum"
ROIS_FNAME = "rois"
STATS_FNAME = "cell_stats"

USE_DENOISED = True
USE_CHAN2 = False

MICRON_PER_METER = 1000000


def create_empty_cell_stat_dict():
    results_dict = {
        "exp_name": [],
        "roi_number": [],
        "length": [],
        "area": [],
        "max_distance_to_midline": [],
        "min_distance_to_midline": [],
    }
    return results_dict


def main():

    cell_stat_dict = create_empty_cell_stat_dict()

    for exp_name in tqdm(EXP_NAMES, desc="Experiment"):
        exp_dir = generate_exp_dir(exp_name, CROP_ID)
        cfg = load_cfg(exp_dir)

        if USE_DENOISED:
            exp_dir = generate_denoised_dir(exp_dir, USE_CHAN2)

        rois = load_custom_rois(exp_dir, ROIS_FNAME)
        n_rois = len(rois)
        pos = get_pos(cfg.Ly, cfg.Lx)

        for roi_num in tqdm(range(n_rois), desc="Roi"):
            roi = rois[roi_num]
            c_pos = calc_c_pos(roi, pos, cfg.Ly)

            pos_dp = calc_pos_pca0(c_pos, cfg, MICRON_PER_METER)

            length = np.amax(pos_dp) - np.amin(pos_dp)
            area = pos_dp.shape[0] * cfg.pixel_size**2
            y_pos = c_pos[0]
            midline = cfg.Ly / 2
            min_distance_to_midline = np.amin(y_pos - midline) * cfg.pixel_size
            max_distance_to_midline = np.amax(y_pos - midline) * cfg.pixel_size

            cell_stat_dict["exp_name"].append(exp_name)
            cell_stat_dict["roi_number"].append(roi_num)
            cell_stat_dict["length"].append(length)
            cell_stat_dict["area"].append(area)
            cell_stat_dict["max_distance_to_midline"].append(max_distance_to_midline)
            cell_stat_dict["min_distance_to_midline"].append(min_distance_to_midline)

    results_dir = generate_global_results_dir()
    df_cell_stat = pd.DataFrame(cell_stat_dict)
    print(df_cell_stat.head())

    df_cell_stat.to_pickle(gen_pickle_fname(results_dir, STATS_FNAME))


if __name__ == "__main__":
    main()
