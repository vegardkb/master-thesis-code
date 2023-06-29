import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA

from fio import (
    load_cfg,
    load_protocol,
)
from sigproc import (
    get_pos,
    calc_c_pos,
)

EXP_NAME = "20220211_13_18_56_GFAP_GCamp6s_F2_C"
# EXP_NAME = "20220211_16_51_15_GFAP_GCamp6s_F4_PTZ"
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

    max_pc = 20

    pca = PCA(max_pc)
    pca.fit(dff)

    e_var = pca.explained_variance_ratio_

    plt.figure()
    plt.bar(np.linspace(1, max_pc + 1, max_pc), e_var)
    plt.plot(np.linspace(1, max_pc + 1, max_pc), np.cumsum(e_var), color="red")
    plt.ylabel("Explained variance")
    plt.xlabel("PC")

    num_frames = dff.shape[0]
    num_pixels = dff.shape[1]
    protocol = load_protocol(PROTOCOL_NAME)
    evt_start, evt_end, _ = extract_events_light_stimulus_events(
        protocol,
        VOLUME_RATE,
        num_frames,
        N_T_START_DISCARD,
        N_T_START_CROP,
        0,
        CORRECTION_VOLUME_RATE,
    )

    l_mask = np.zeros(num_frames, dtype=bool)
    for es, ee in zip(evt_start, evt_end):
        l_mask[es:ee] = True

    c = []
    for l in l_mask:
        if l:
            c.append("orange")
        else:
            c.append("gray")

    pca = PCA(3)
    dff_3 = pca.fit_transform(dff)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(dff_3[:, 0], dff_3[:, 1], dff_3[:, 2], s=2, c=c, alpha=0.6)

    plt.show()


if __name__ == "__main__":
    main()
