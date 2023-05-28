from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import os

from fio import (
    generate_global_results_dir,
    generate_figures_dir,
    gen_pickle_fname,
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
CROP_IDS = ["OpticTectum"]
ROIS_FNAME = "rois"
STATS_FNAME = "cell_stats"

MICROM_PER_M = 1000000
PRE_EVENT_T = 5
POST_EVENT_T = 60

USE_DENOISED = True
USE_CHAN2 = False

REG_CM = cm.viridis
N_REGIONS = 6

MAX_DIST_SOMA = 0.5
CELL_LENGTH_T = 50

SMALL_SIZE = 8
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE, dpi=600)  # fontsize of the figure title
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["lines.linewidth"] = 0.75


def set_fig_size(scale, y_scale):
    a4_w = 8.3
    lmargin = 1
    rmargin = 1
    text_width = a4_w - lmargin - rmargin
    plt.rc(
        "figure", figsize=(scale * text_width, y_scale * scale * text_width), dpi=600
    )


def save_fig(fig_dir, fname):
    plt.savefig(
        os.path.join(fig_dir, fname + ".png"),
        bbox_inches="tight",
    )


def create_empty_cell_stat_dict():
    results_dict = {
        "exp_name": [],
        "roi_number": [],
        "area": [],
        "max_distance_to_midline": [],
        "min_distance_to_midline": [],
    }
    return results_dict


def load_pickle(folder, fname):
    return pd.read_pickle(gen_pickle_fname(folder, fname))


def get_region_colors(num_regs):
    return REG_CM(np.linspace(0.2, 1, num_regs))


def plot_cell_lengths(length, threshold):
    plt.figure()
    num_rois = length.shape[0]
    x = np.linspace(0, num_rois - 1, num_rois)

    height = np.sort(length)

    plt.bar(x, height, color="darkgray")
    plt.hlines([threshold], 0, num_rois - 1, linestyle="--", color="red", alpha=0.8)

    plt.xlabel("Cell number")
    plt.ylabel("Length [\u03bcm]")
    plt.tight_layout()


def plot_cropped_cell_lengths(length, threshold, reg_colors):
    num_regs = reg_colors.shape[0]

    length_filt = length[length >= threshold]

    length_filt = np.sort(length_filt)

    remainder = length_filt - threshold

    thresholds = np.linspace(0, threshold, num_regs + 1)
    num_cells = length_filt.shape[0]
    x = np.arange(num_cells)

    plt.figure()
    for color, t_l, t_h in zip(reg_colors, thresholds[:-1], thresholds[1:]):
        plt.bar(
            x,
            height=(t_h - t_l) * np.ones(num_cells),
            bottom=t_l * np.ones(num_cells),
            color=color,
        )

    top = thresholds[-1]
    plt.bar(
        x,
        height=remainder,
        bottom=top * np.ones(num_cells),
        color="darkgray",
        alpha=0.8,
    )

    plt.xlabel("Cell number")
    plt.ylabel("Length [\u03bcm]")
    plt.tight_layout()


def main():
    results_dir = generate_global_results_dir()
    g_fig_dir = generate_figures_dir()
    fig_dir = os.path.join(g_fig_dir, "morphology")

    cell_stats = load_pickle(results_dir, STATS_FNAME)

    length_pd = np.array(cell_stats["length"])

    set_fig_size(0.48, 1)
    plot_cell_lengths(length_pd, CELL_LENGTH_T)
    save_fig(fig_dir, "cell_length_all")

    reg_colors = get_region_colors(N_REGIONS)
    plot_cropped_cell_lengths(length_pd, CELL_LENGTH_T, reg_colors)
    save_fig(fig_dir, "cell_length_filtered")

    for exp_name in EXP_NAMES:
        df_exp = cell_stats[cell_stats["exp_name"] == exp_name]
        length_s = df_exp["length"]
        num_long = np.sum(length_s > CELL_LENGTH_T)
        print(f"{exp_name}: {num_long} cells")

    plt.show()


if __name__ == "__main__":
    main()
