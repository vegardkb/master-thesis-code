import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

from fio import (
    load_cfg,
    load_protocol,
)
from sigproc import (
    get_pos,
    calc_c_pos,
    calc_pos_pca0,
)

EXP_NAME = "20211112_18_30_27_GFAP_GCamp6s_F5_c2"
ROI_NUM = 2
EXP_DIR = f"Y:\\Vegard\\data\\{EXP_NAME}\\OpticTectum"
DFF_PATH = f"Y:\\Vegard\\data\\{EXP_NAME}\\OpticTectum\\denoised\\results\\ROI{ROI_NUM}\\dff\\rois\\dff_stat.npy"
ROIS_PATH = f"Y:\\Vegard\\data\\{EXP_NAME}\\OpticTectum\\denoised\\rois.npy"
PROTOCOL_NAME = "GFAP;Gcamp6s2021_11_12_data"

VOLUME_RATE = 4.86
CORRECTION_VOLUME_RATE = 30 / 5
N_T_START_CROP = 2000
N_T_START_DISCARD = 0

REG_CM = cm.viridis_r
N_REGIONS = 3


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


def get_im(frame, shape, c_pos, dff):
    im = np.zeros(shape)

    f = dff[frame]

    for i in range(c_pos.shape[1]):
        pos = c_pos[:, i]
        im[pos[0], pos[1]] = f[i]

    return im


def animate(c_pos, dff):
    fig, ax = plt.subplots()

    buf = 2
    c_pos = (c_pos.T - np.amin(c_pos, axis=1)).T + buf

    shape = (np.amax(c_pos[0]) + buf, np.amax(c_pos[1]) + buf)
    dff = (dff - np.amin(dff)) / (np.amax(dff) - np.amin(dff))

    ax.axis("off")
    im = ax.imshow(
        get_im(0, shape, c_pos, dff), animated=True, cmap=cm.inferno, vmin=0, vmax=1
    )
    time_text = ax.text(0.5, MEDIUM_SIZE, "0 sec", fontsize=MEDIUM_SIZE, color="white")

    def update(frame):
        im.set_array(get_im(frame, shape, c_pos, dff))
        time_text.set_text(f"{np.round(frame / VOLUME_RATE, 0)} sec")
        return (
            im,
            time_text,
        )

    ani = animation.FuncAnimation(fig, update, interval=5)
    plt.show()


def main():
    dff = np.load(DFF_PATH)
    roi = np.load(ROIS_PATH)[ROI_NUM]
    cfg = load_cfg(EXP_DIR)
    fig_dir = "Y:\\Vegard\\data\\figures\\preprocess"

    print(f"dff.shape: {dff.shape}")
    print(f"dff: {dff}")
    print(f"roi.shape: {roi.shape}")
    print(f"roi: {roi}")

    pos = get_pos(cfg.Ly, cfg.Lx)
    c_pos = calc_c_pos(roi, pos, cfg.Ly)

    print(f"c_pos.shape: {c_pos.shape}")
    print(f"c_pos: {c_pos}")

    animate(c_pos, dff)


if __name__ == "__main__":
    main()
