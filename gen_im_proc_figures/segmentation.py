from fio import (
    generate_figures_dir,
    generate_exp_dir,
    generate_denoised_dir,
    load_cfg,
    load_s2p_df_rois,
)
from sigproc import get_pos, calc_c_pos, calc_pos_pca0, get_reg_mask
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np

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
CROP_ID = "OpticTectum"
ROIS_FNAME = "rois"

USE_DENOISED = True
USE_CHAN2 = False

ROI_CM = cm.rainbow
REG_CM = cm.viridis_r

N_REGIONS = 6
MICROM_PER_M = 1000000

N_FRAMES_BASELINE = 800

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
    plt.rc("figure", figsize=(scale * text_width, y_scale * scale * text_width))


def save_fig(fig_dir, fname, format="png"):
    plt.savefig(
        os.path.join(fig_dir, fname + f".{format}"),
        bbox_inches="tight",
    )


def get_mean_img_rgb(f):
    n_ch = 3
    mean_im = np.zeros((f.shape[1], f.shape[2], n_ch))

    mu_im = np.mean(f, axis=0)
    mu_im = mu_im / np.amax(mu_im)  # Normalize
    for i in range(n_ch):
        mean_im[:, :, i] = mu_im

    return mean_im


def plot_rois(mu_im, rois, colors, pixel_size, shading_strength=1):
    """
    Plots rois over anatomical image
    """

    _, ax = plt.subplots()
    mu_im_w_rois = np.copy(mu_im)
    if type(rois) == list and type(colors) == list:
        for roi, color in zip(rois, colors):
            mu_im_w_rois[roi] = (mu_im[roi] + color * shading_strength) / (
                1 + shading_strength
            )

    elif type(rois) == list or type(colors) == list:
        raise Exception("Error: either rois or colors is list but not both.")

    else:
        mu_im_w_rois[rois] = (mu_im[rois] + colors) / 2

    ax.imshow(mu_im_w_rois)
    ax.axis("off")

    bar_size = 10
    loc = "lower right"
    asb = AnchoredSizeBar(
        ax.transData,
        size=bar_size / pixel_size * 1e-6,
        size_vertical=1,
        label=f"{str(bar_size)} \u03bcm",
        loc=loc,
        pad=0.1,
        borderpad=2,
        frameon=False,
        color="white",
    )
    ax.add_artist(asb)

    plt.tight_layout()


def plot_roi_dff(f, rois, colors, volume_rate):

    f_rois = np.array([np.mean(f[:, roi], axis=1) for roi in rois])
    f0 = np.mean(f_rois[:, :N_FRAMES_BASELINE], axis=1)
    dff = ((f_rois.T - f0) / f0).T

    delta_y = np.amax(dff) - np.amin(dff)

    av_dff = np.mean(dff, axis=0)

    _, ax = plt.subplots()
    for roi_num, dff_roi in enumerate(dff):
        if roi_num == 0:
            ax.plot(av_dff, color="darkgray", alpha=0.7, label="Average")
        else:
            ax.plot(av_dff - delta_y * roi_num, color="darkgray", alpha=0.7)

        ax.plot(
            dff_roi - delta_y * roi_num,
            color=colors[roi_num],
            label=f"ROI {roi_num+1}",
            alpha=0.8,
        )

    scalebar_width_pixels = 10
    inv = ax.transData.inverted()
    points = inv.transform([(0, 0), (scalebar_width_pixels, scalebar_width_pixels)])
    scale_x = points[0, 1] - points[0, 0]
    scale_y = points[1, 1] - points[1, 0]

    bar_size = 1  # minute
    loc = "lower center"
    asb = AnchoredSizeBar(
        ax.transData,
        size=bar_size * volume_rate * 60,
        size_vertical=scale_y,
        label=f"{str(bar_size)} min",
        loc=loc,
        pad=0.1,
        borderpad=-2,
        frameon=False,
        color="black",
    )
    ax.add_artist(asb)

    bar_size = max(((delta_y / 5) // 0.5) * 0.5, 0.5)  # dff
    loc = "upper left"
    asb = AnchoredSizeBar(
        ax.transData,
        size=scale_x,
        size_vertical=bar_size / 1,
        label=f"{str(int(bar_size * 100))}% $\Delta F / F_0$",
        loc=loc,
        pad=0.1,
        borderpad=-2,
        frameon=False,
        color="black",
    )
    ax.add_artist(asb)

    ax.axis("off")

    plt.tight_layout()


def get_region_colors(num_regs):
    return REG_CM(np.linspace(0.2, 1, num_regs))


def get_region_pos(roi, cfg, num_reg):
    pos = get_pos(cfg.Ly, cfg.Lx)
    c_pos = calc_c_pos(roi, pos, cfg.Ly)
    pos_pca0 = calc_pos_pca0(c_pos, cfg, MICROM_PER_M)
    region_mask = get_reg_mask(pos_pca0, num_reg)
    region_pos = (
        np.zeros((pos.shape[0], region_mask.shape[0], region_mask.shape[1])) - 1
    )
    for reg_num in range(region_mask.shape[0]):
        region_pos[:, reg_num, region_mask[reg_num]] = c_pos[:, region_mask[reg_num]]

    return region_pos


def plot_pos_reg(pos_reg, num_regs):
    reg_colors = get_region_colors(num_regs)
    marker_size = 0.1
    marker_alpha = 0.8

    _, ax = plt.subplots()

    for reg_num in range(num_regs):
        mask = pos_reg[0, reg_num] >= 0
        ax.scatter(
            pos_reg[1, reg_num, mask],
            pos_reg[0, reg_num, mask],
            s=marker_size,
            alpha=marker_alpha,
            color=reg_colors[reg_num],
            marker="s",
        )

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    dy, dx = ymax - ymin, xmax - xmin

    if dy > dx:
        diff = dy - dx
        pad = int(diff / 2)
        ax.set_xlim(xmin - pad, xmax + pad)

    elif dy < dx:
        diff = -dy + dx
        pad = int(diff / 2)
        ax.set_ylim(ymin - pad, ymax + pad)

    bar_width = 1
    bar_size = 20  # micrometers
    loc = "lower right"
    asb = AnchoredSizeBar(
        ax.transData,
        size=bar_size,
        size_vertical=bar_width,
        label=f"{bar_size} \u03bcm",
        loc=loc,
        pad=0.1,
        borderpad=1,
        frameon=False,
        color="black",
    )
    ax.add_artist(asb)

    ax.axis("off")


def pos_regs_to_rois(pos_regs, cfg):
    ly, lx = cfg.Ly, cfg.Lx
    rois = []
    for reg_num in range(pos_regs.shape[1]):
        roi = np.zeros((ly, lx), dtype=bool)
        for pixel_num in range(pos_regs.shape[2]):
            y, x = int(pos_regs[0, reg_num, pixel_num]), int(
                pos_regs[1, reg_num, pixel_num]
            )
            roi[y, x] = True

        rois.append(roi)

    return rois


def main():
    fig_dir = generate_figures_dir()
    a_fig_dir = os.path.join(fig_dir, "anatomical")

    reg_colors = get_region_colors(N_REGIONS)

    for exp_name in EXP_NAMES:
        print(f"Processing {exp_name}")

        a_exp_fig_dir = os.path.join(a_fig_dir, exp_name)
        if not os.path.isdir(a_exp_fig_dir):
            os.makedirs(a_exp_fig_dir)

        exp_dir = generate_exp_dir(exp_name, CROP_ID)
        cfg = load_cfg(exp_dir)
        pixel_size = cfg.pixel_size
        volume_rate = cfg.volume_rate

        if USE_DENOISED:
            exp_dir = generate_denoised_dir(exp_dir, USE_CHAN2)

        s2p_df, rois = load_s2p_df_rois(exp_dir, cfg, ROIS_FNAME)

        f = s2p_df["x"]

        mu_im = np.power(get_mean_img_rgb(f), 0.9)

        print(rois[0].shape)

        n_rois = len(rois)
        tmp = ROI_CM(np.linspace(0, 1, n_rois))[:, :3]
        roi_colors = [tmp[i] for i in range(n_rois)]

        set_fig_size(0.48, 1)
        plot_rois(mu_im, rois, roi_colors, pixel_size, 0.45)
        save_fig(a_exp_fig_dir, "anatomical")

        plot_roi_dff(f, rois, roi_colors, volume_rate)
        save_fig(a_exp_fig_dir, "cell_traces")

        for roi_num, roi in enumerate(rois):
            region_pos = get_region_pos(roi, cfg, N_REGIONS)
            plot_pos_reg(region_pos, N_REGIONS)
            save_fig(a_exp_fig_dir, f"region_pos_roi{roi_num}")

            plot_rois(mu_im, [roi], [roi_colors[roi_num]], pixel_size, 0.5)
            save_fig(a_exp_fig_dir, f"anatomical_roi{roi_num}")

            sub_rois = pos_regs_to_rois(region_pos, cfg)
            plot_roi_dff(f, sub_rois, reg_colors, volume_rate)
            save_fig(a_exp_fig_dir, f"region_traces_roi{roi_num}")

        plt.close()


if __name__ == "__main__":
    main()
