import numpy as np
from matplotlib import pyplot as plt
import os

from fio import (
    generate_figures_dir,
)

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
plt.rcParams["lines.linewidth"] = 0.5


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


def light_stim_plot():
    t_bl = 10 * 60
    t_on = []

    t_last = t_bl
    for isi in [300, 120, 60, 30, 15]:
        for _ in range(5):
            t_on.append(t_last / 60)
            t_last += 10 + isi

    t_on = np.array(t_on)
    t_off = t_on + 10 / 60

    _, ax = plt.subplots()
    for on, off in zip(t_on, t_off):
        ax.fill_betweenx([0, 1], on, off, alpha=0.7, color="red")
    ax.set_xlim(0, 60)
    ax.set_xlabel("Time [min]")

    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.set_yticks([])


def main():
    fig_dir = os.path.join(generate_figures_dir(), "exp")
    set_fig_size(0.4, 0.4)
    light_stim_plot()
    save_fig(fig_dir, "light_stim")

    plt.show()


if __name__ == "__main__":
    main()
