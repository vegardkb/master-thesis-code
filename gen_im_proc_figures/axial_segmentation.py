from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import os
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


import sigproc
from fio import load_cfg

EXP_NAME = "20211112_18_30_27_GFAP_GCamp6s_F5_c2"
ROI_NUM = 0
EXP_DIR = f"Y:\\Vegard\\data\\{EXP_NAME}\\OpticTectum"
ROIS_PATH = f"{EXP_DIR}\\denoised\\rois.npy"

MICRON_PER_METER = 1e6

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE, dpi=300)  # fontsize of the figure title
plt.rcParams["font.family"] = "serif"
plt.rcParams["lines.linewidth"] = 0.75


def set_fig_size(scale, y_scale):
    a4_w = 8.3
    lmargin = 1.5
    rmargin = 1
    text_width = a4_w - lmargin - rmargin
    plt.rc("figure", figsize=(scale * text_width, y_scale * scale * text_width))


def save_fig(fig_dir, fname):
    plt.savefig(
        os.path.join(fig_dir, fname + ".png"),
        bbox_inches="tight",
    )


def plot_roi(c_pos, mask=None, num_regs=6):
    marker_size = 0.1
    marker_alpha = 0.8

    _, ax = plt.subplots()

    if mask is not None:
        reg_colors = cm.viridis_r(np.linspace(0.2, 1, num_regs))
        for reg_num in range(num_regs):
            c_pos_m = c_pos[:, mask[reg_num]]
            ax.scatter(
                c_pos_m[1],
                c_pos_m[0],
                s=marker_size,
                alpha=marker_alpha,
                color=reg_colors[reg_num],
                marker="s",
            )

    else:
        ax.scatter(
            c_pos[1],
            c_pos[0],
            s=marker_size,
            alpha=marker_alpha,
            color="gray",
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


def plot_pcs(pc1):
    fig, ax = plt.subplots()

    ax.quiver([0, 0], pc1, color="black", linestyle="dashed", label="PC1")
    ax.legend()


def main():
    fig_dir = "Y:\\Vegard\\data\\figures\\preprocess"
    num_regions = 6

    roi = np.load(ROIS_PATH)[ROI_NUM]
    cfg = load_cfg(EXP_DIR)

    print(f"roi.shape: {roi.shape}")

    pos = sigproc.get_pos(cfg.Ly, cfg.Lx)
    c_pos = sigproc.calc_c_pos(roi, pos, cfg.Ly)
    print(c_pos)

    set_fig_size(0.48, 1)
    plot_roi(c_pos)
    save_fig(fig_dir, "roi_gray")
    pos_pd = sigproc.calc_pos_pca0(c_pos, cfg, MICRON_PER_METER)
    reg_mask = sigproc.get_reg_mask(pos_pd, num_regions)

    pc1 = sigproc.calc_pc1(c_pos, cfg, MICRON_PER_METER)

    plot_pcs(pc1)
    save_fig(fig_dir, "roi_pcs")

    plot_roi(c_pos, mask=reg_mask, num_regs=num_regions)
    save_fig(fig_dir, "roi_reg")

    plt.show()


if __name__ == "__main__":
    main()
