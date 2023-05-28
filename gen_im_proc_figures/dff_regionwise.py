import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from fio import (
    load_cfg,
    load_s2p_df_rois,
    generate_exp_dir,
    generate_denoised_dir,
    generate_results_dir,
    generate_roi_dff_dir,
    gen_npy_fname,
    load_protocol,
)
from sigproc import (
    get_pos,
    calc_c_pos,
    calc_pos_pca0,
)

EXP_NAME = "20211112_18_30_27_GFAP_GCamp6s_F5_c2"
ROI_NUM = 0
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


def get_reg_mask(pos_pca0, n_regions):
    """
    Divide into regions by position along proximal-distal axis, equal length of regions
    """

    thr = np.linspace(
        np.amin(pos_pca0) - 1e-10, np.amax(pos_pca0) + 1e-10, n_regions + 1
    )

    mask = []
    for i in range(n_regions):
        mask.append(np.logical_and(pos_pca0 > thr[i], pos_pca0 <= thr[i + 1]))

    return np.array(mask)


def get_reg_activity(reg_mask, x, max_dyn_range=False):
    """
    reg_mask: 2d boolean ndarray (n_regions, pix)
    x: 2d ndarray (t, pix)

    return:
     - y: 2d ndarray (t, n_regions)
    """
    num_frames = x.shape[0]
    num_reg = reg_mask.shape[0]
    y = np.zeros((num_frames, num_reg))
    for reg_num in range(reg_mask.shape[0]):
        x_reg = x[:, reg_mask[reg_num]]
        if max_dyn_range:
            dyn_range = np.amax(x_reg, axis=0) - np.amin(x_reg, axis=0)
            max_ind = np.argmax(dyn_range)
            y[:, reg_num] = x_reg[:, max_ind]

        else:
            y[:, reg_num] = np.mean(x_reg, axis=1)

    return y


def main():
    dff = np.load(DFF_PATH)
    roi = np.load(ROIS_PATH)[ROI_NUM]
    cfg = load_cfg(EXP_DIR)
    fig_dir = "Y:\\Vegard\\data\\figures\\preprocess"

    print(f"dff.shape: {dff.shape}")
    print(f"roi.shape: {roi.shape}")

    pos = get_pos(cfg.Ly, cfg.Lx)
    c_pos = calc_c_pos(roi, pos, cfg.Ly)
    pos_pd = calc_pos_pca0(c_pos, cfg, 1e6)

    set_fig_size(0.9, 0.7)

    num_frames = dff.shape[0]
    num_pixels = dff.shape[1]
    protocol = load_protocol(PROTOCOL_NAME)
    evt_start, _, _ = extract_events_light_stimulus_events(
        protocol,
        VOLUME_RATE,
        num_frames,
        N_T_START_DISCARD,
        N_T_START_CROP,
        0,
        CORRECTION_VOLUME_RATE,
    )

    light_on = evt_start
    light_off = evt_start + 10 * VOLUME_RATE

    t_whole_minutes = np.arange(num_frames) / VOLUME_RATE / 60

    fig, ax = plt.subplots()
    for pixel_num in range(dff.shape[1]):
        ax.plot(t_whole_minutes, dff[:, pixel_num] * 100, color="darkgray", alpha=0.6)
    ax.set_xlabel("time [min]")
    ax.set_ylabel(r"$\Delta F/F$ [%]")
    for lon, loff in zip(light_on, light_off):
        ax.axvline(lon / VOLUME_RATE / 60, color="orange", linestyle="dashed")
        ax.axvline(loff / VOLUME_RATE / 60, color="darkgray", linestyle="dashed")

    save_fig(fig_dir, "cell_pixels_trace")

    fig, ax = plt.subplots()
    ax.plot(t_whole_minutes, np.mean(dff, axis=1) * 100, color="black")
    ax.set_xlabel("time [min]")
    ax.set_ylabel(r"$\Delta F/F$ [%]")
    for lon, loff in zip(light_on, light_off):
        ax.axvline(lon / VOLUME_RATE / 60, color="orange", linestyle="dashed")
        ax.axvline(loff / VOLUME_RATE / 60, color="darkgray", linestyle="dashed")

    save_fig(fig_dir, "cell_average_trace")

    reg_mask = get_reg_mask(pos_pd, N_REGIONS)
    dff_reg = get_reg_activity(reg_mask, dff)
    reg_colors = REG_CM(np.linspace(0, 1, N_REGIONS))

    fig, ax = plt.subplots()
    for reg_num in range(N_REGIONS):
        ax.plot(t_whole_minutes, dff_reg[:, reg_num] * 100, color=reg_colors[reg_num])

    ax.set_xlabel("time [min]")
    ax.set_ylabel(r"$\Delta F/F$ [%]")
    for lon, loff in zip(light_on, light_off):
        ax.axvline(lon / VOLUME_RATE / 60, color="orange", linestyle="dashed")
        ax.axvline(loff / VOLUME_RATE / 60, color="darkgray", linestyle="dashed")

    save_fig(fig_dir, "regional_trace_whole")

    evt_start = light_on - int(5 * VOLUME_RATE)
    evt_end = light_on + int(120 * VOLUME_RATE)

    dff_first_trial = dff_reg[evt_start[0] : evt_end[0]]
    t_trial = np.arange(evt_end[0] - evt_start[0]) / VOLUME_RATE - 5

    fig, ax = plt.subplots()
    for reg_num in range(N_REGIONS):
        ax.plot(t_trial, dff_first_trial[:, reg_num] * 100, color=reg_colors[reg_num])

    ax.set_xlabel("time [sec]")
    ax.set_ylabel(r"$\Delta F/F$ [%]")
    lon, loff = 0, 10
    ax.axvline(lon, color="orange", linestyle="dashed")
    ax.axvline(loff, color="darkgray", linestyle="dashed")

    save_fig(fig_dir, "regional_trace_trial")

    plt.show()


if __name__ == "__main__":
    main()
