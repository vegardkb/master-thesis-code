import numpy as np
import matplotlib.pyplot as plt

from fio import (
    load_cfg,
    load_s2p_df_rois,
    get_t,
    generate_exp_dir,
    generate_denoised_dir,
    save_rois,
)

from plotters import plot_frame, plot_rois

from sigproc import (
    get_mean_img_rgb,
    get_pos,
    spatial_distance,
    activity_distance,
)

from uio import merge_suggest_gui


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
EXP_NAMES = [
    "20220604_15_00_04_HuC_GCamp6s_GFAP_jRGECO_F2_PTZ",
    # "20220604_16_24_03_HuC_GCamp6s_GFAP_jRGECO_F3_PTZ",
]
CROP_IDS = [
    "OpticTectum",
]

USE_DENOISED = True
USE_CHAN2 = True

NAME_ROIS = "rois"


def remove_roi(roi, d, rois_merged):
    rois_merged.append(roi)
    d[roi, :] = np.inf
    d[:, roi] = np.inf
    return d, rois_merged


def merge_rois(k, l, rois):
    roi1 = rois[k]
    roi2 = rois[l]
    rois.append(np.logical_or(roi1, roi2))
    return rois


def semi_auto_merge(
    f, rois, pos, spat_distance_metric, act_distance_metric, t, pixel_size
):
    """
    The algorithm.

    Bad:
        - This is a very long function
    """
    rois_orig = rois
    mean_im = get_mean_img_rgb(f)
    n_rois = len(rois)
    n_iteration = 0
    user_input = "n"
    while user_input != "q":
        print(f"Iteration {n_iteration} starting with {n_rois} rois")
        d = np.zeros((n_rois, n_rois))
        for i in range(n_rois):
            roi1 = rois[i]
            for j in range(i):
                roi2 = rois[j]
                d[i, j] = spat_distance_metric(roi1, roi2, pos)
                d[j, i] = np.inf

            d[i, i] = np.inf

        d_0 = d
        ind_best = np.argmin(d_0, axis=1)
        d = np.ones(d_0.shape) * np.inf
        for i, ind in enumerate(ind_best):
            if i != ind:
                roi1 = rois[i]
                roi2 = rois[ind]
                d[i, ind] = act_distance_metric(roi1, roi2, f, pos, t, pixel_size)

        d_amin = np.amin(d)
        rois_merged = []
        skip = False
        while d_amin != np.inf and user_input != "q" and skip != True:
            n_comb = np.sum(np.logical_not(np.isinf(d)))
            print(f"Number of roi combinations left: {n_comb}")
            d_argmin = np.argmin(d)
            k = d_argmin // n_rois
            l = d_argmin % n_rois

            user_input = merge_suggest_gui(mean_im, f, rois, k, l, t)

            if user_input == "y":
                rois = merge_rois(k, l, rois)

                d, rois_merged = remove_roi(k, d, rois_merged)
                d, rois_merged = remove_roi(l, d, rois_merged)

            elif user_input == "remove_1":
                d, rois_merged = remove_roi(k, d, rois_merged)

            elif user_input == "remove_2":
                d, rois_merged = remove_roi(l, d, rois_merged)

            elif user_input == "skip":
                skip = True

            d[l, k] = np.inf
            d[k, l] = np.inf

            d_amin = np.amin(d)

        """
            Remove parent rois
        """
        new_rois = []
        for i, roi in enumerate(rois):
            if i not in rois_merged:
                new_rois.append(roi)

        print(f"rois_merged(or removed): {rois_merged}")

        rois = new_rois
        n_rois = len(rois)
        n_iteration = n_iteration + 1
        print(f"Iteration {n_iteration} ended with {n_rois} rois")

    return rois, rois_orig


def main():
    for exp_name in EXP_NAMES:
        for crop_id in CROP_IDS:
            exp_dir = generate_exp_dir(exp_name, crop_id)
            cfg = load_cfg(exp_dir)

            if USE_DENOISED:
                exp_dir = generate_denoised_dir(exp_dir, USE_CHAN2)

            s2p_df, rois = load_s2p_df_rois(exp_dir, cfg)

            """
                Debug
            """
            # Debug code snippet is broken..
            """ plot_frame(s2p_df["x"])

            plot_rois(rois)
            plt.show() """
            """
                End debug
            """

            print(f"(Ly, Lx) = ({cfg.Ly}, {cfg.Lx})")
            t = get_t(cfg.volume_rate, s2p_df["x"].shape[0])
            pos = get_pos(cfg.Ly, cfg.Lx)

            new_rois, old_rois = semi_auto_merge(
                s2p_df["x"],
                rois,
                pos,
                spatial_distance,
                activity_distance,
                t,
                cfg.pixel_size,
            )

            save_rois(new_rois, exp_dir, NAME_ROIS)

            """ plot_rois(old_rois, title=f"Original ROIs, n={len(old_rois)}")
            plot_rois(new_rois, title=f"New ROIs, n={len(new_rois)}")


            plt.show() """


if __name__ == "__main__":
    main()
