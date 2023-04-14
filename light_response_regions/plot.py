import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import copy
from scipy import signal
from sklearn.linear_model import LinearRegression

from fio import (
    generate_global_results_dir,
    gen_pickle_fname,
    generate_figures_dir,
    gen_npy_fname,
    gen_image_fname,
    gen_pickle_fname,
)


NEGATIVE_RGB = (165, 165, 165)  # gray
SUBTHRESHOLD_RGB = (146, 243, 240)  # light blue
AMPLIFYING_RGB = (209, 128, 7)  # orange-ish
ATTENUATING_RGB = (149, 243, 146)  # light green
UNDEFINED_RGB = (255, 255, 255)  # white
EARLY_RGB = (52, 179, 86)  # green
MIDDLE_RGB = (191, 107, 29)  # orange-brown ish
LATE_RGB = (140, 24, 24)  # red

DARKGRAY = (130 / 255, 129 / 255, 129 / 255)
LIGHTBLUE = (101 / 255, 221 / 255, 247 / 255)
ORANGE = (247 / 255, 151 / 255, 54 / 255)
RED = (194 / 255, 35 / 255, 23 / 255)

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

DFF_REGS_FNAME = "dff_light_response"
SECOND_DERIVATIVE_FNAME = "d2xdt2_light_response"
STATS_FNAME = "stats_light_response"
T_ONSET_FNAME = "t_onsets_light_response"

N_REGIONS = 8
DISTAL_REG = 0
PROXIMAL_REG = N_REGIONS - 1

CELL_LENGTH_THRESHOLDS = [40, 60]

# ISIS = [300, 120, 60, 30, 15]
ISIS = [300, 120, 60]

REG_CM = cm.viridis_r

SIG_SINGLE = 0.05
SIG_DOUBLE = 0.01
SIG_TRIPLE = 0.001

MARKER_SIZE = 8
MIN_SIZE = 2
MAX_SIZE = 40

VOLUME_RATE = 4.86
PRE_EVENT_T = 5
STIM_DURATION = 10
POST_STIM_T = 5

CTRL_PTZ_COLORS = ("darkgray", "brown")
AMP_CAT_COLORS = (DARKGRAY, LIGHTBLUE, ORANGE)
AMP_CAT_COLORS_DICT = {
    "negative": DARKGRAY,
    "attenuating": LIGHTBLUE,
    "amplifying": ORANGE,
}

T_ONSET_CAT_COLORS = (ORANGE, LIGHTBLUE, DARKGRAY)
T_ONSET_CAT_COLORS_DICT = {
    "light": ORANGE,
    "sequential": LIGHTBLUE,
    "undefined": DARKGRAY,
}

""" T_ONSET_CAT_COLORS_DICT = {
    "distal_proximal": ORANGE,
    "proximal_distal": LIGHTBLUE,
    "simultaneous": DARKGRAY,
} """

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc(
    "figure", titlesize=BIGGER_SIZE, figsize=(12, 12)
)  # fontsize of the figure title
plt.rcParams["font.family"] = "Times New Roman"


""" def ar2_model_fit(x):
    autocorr = signal.correlate(x, x, mode="same")
    lags = signal.correlation_lags(x.shape[0], x.shape[0], mode="same")
    rho1_2 = autocorr[np.logical_or(lags == 1, lags == 2)]
    rho1 = rho1_2[0]
    rho2 = rho1_2[1]

    phi1 = rho1 * ((1 - rho2) / (1 - rho1**2))
    phi2 = (rho2 - rho1**2) / (1 - rho1**2)

    print(f"phi1: {phi1}")
    print(f"phi2: {phi2}")

    return np.array([phi1, phi2]) """


def arp_model_fit(x, p):
    n = len(x)
    X = np.zeros((n - p, p))
    y = np.zeros(n - p)

    for t in range(n - p):
        X[t] = np.flip(x[t : t + p])
        y[t] = x[t + p]

    model = LinearRegression(fit_intercept=False).fit(X, y)
    phis = model.coef_
    print(f"phis: {phis}")

    return phis


def arp_model_pred(x, par):
    p = len(par)
    n = len(x)
    y = np.zeros(n)
    y[:p] = x[:p]
    for t in range(p, n):
        for i in range(p):
            y[t] += par[i] * x[t - 1 - i]

    return y


def arp_model_res(x, par):
    y = arp_model_pred(x, par)
    z = x - y
    return z


def load_pickle(dir, fname, num_regs):
    return pd.read_pickle(gen_pickle_fname(dir, fname + f"_{num_regs}_regions"))


def load_np(dir, fname, num_regs):
    return np.load(gen_npy_fname(dir, fname + f"_{num_regs}_regions"))


def get_dot_size(df, size_col, size_lim, logarithmic=False):
    s = MARKER_SIZE
    if size_col is not None:
        size_min, size_max = MIN_SIZE, MAX_SIZE
        scale_series = df[size_col]
        if size_lim is not None:
            size_min, size_max = size_lim[0], size_lim[1]

        if logarithmic:
            scale_series = np.log10(scale_series - scale_series.min() + 1)
            scale_min, scale_max = scale_series.min(), scale_series.max()
            s = (scale_series - scale_min) / (scale_max - scale_min) * (
                size_max - size_min
            ) + size_min

        else:
            scale_min, scale_max = scale_series.min(), scale_series.max()
            s = (scale_series - scale_min) / (scale_max - scale_min) * (
                size_max - size_min
            ) + size_min

    return s


def get_t_onset_masks(df, num_regs):
    distal_reg = 0
    proximal_reg = num_regs - 1
    middle_reg = int((distal_reg + proximal_reg) / 2)

    lon_s, lon_e = -0.5, 4
    loff_s, loff_e = 9.5, 14

    mean_t, med_t = 0, 0

    t_onset_distal = df[f"t_onset_r{distal_reg}"]
    t_onset_middle = df[f"t_onset_r{middle_reg}"]
    t_onset_proximal = df[f"t_onset_r{proximal_reg}"]

    for reg_d, reg_p in zip(range(num_regs - 1), range(1, num_regs)):
        df[f"t_onset_lag_{reg_d}_{reg_p}"] = (
            df[f"t_onset_r{reg_p}"] - df[f"t_onset_r{reg_d}"]
        )

    df["t_onset_lag_mean"] = df[
        [
            f"t_onset_lag_{reg_d}_{reg_p}"
            for reg_d, reg_p in zip(range(num_regs - 1), range(1, num_regs))
        ]
    ].mean(axis=1)
    df["t_onset_lag_median"] = df[
        [
            f"t_onset_lag_{reg_d}_{reg_p}"
            for reg_d, reg_p in zip(range(num_regs - 1), range(1, num_regs))
        ]
    ].median(axis=1)
    df["t_onset_lag_var"] = df[
        [
            f"t_onset_lag_{reg_d}_{reg_p}"
            for reg_d, reg_p in zip(range(num_regs - 1), range(1, num_regs))
        ]
    ].var(axis=1)

    light = (
        (
            ((t_onset_distal < lon_e) & (t_onset_distal > lon_s))
            | ((t_onset_distal < loff_e) & (t_onset_distal > loff_s))
        )
        & (
            ((t_onset_middle < lon_e) & (t_onset_middle > lon_s))
            | ((t_onset_middle < loff_e) & (t_onset_middle > loff_s))
        )
        & (
            ((t_onset_proximal < lon_e) & (t_onset_proximal > lon_s))
            | ((t_onset_proximal < loff_e) & (t_onset_proximal > loff_s))
        )
    )
    seq = (
        ~light
        & (df["t_onset_lag_mean"] > mean_t)
        & ((df["t_onset_lag_median"] > med_t))
    )
    undefined = ~seq & ~light

    category_masks = [light, seq, undefined]
    category_labels = ["light", "sequential", "undefined"]

    return category_masks, category_labels


def get_amp_masks(df, num_regs):
    distal_reg = 0
    proximal_reg = num_regs - 1
    distal_amp = df[f"amp_r{distal_reg}"]
    proximal_amp = df[f"amp_r{proximal_reg}"]
    """ slope = 0.35
    intercept = 1.75

    neg_mask = (distal_amp <= 0) | (proximal_amp <= 0)
    norm_mask = (proximal_amp <= distal_amp * slope + intercept) & (~neg_mask)
    amp_mask = (~norm_mask) & (~neg_mask) """

    neg_mask = (distal_amp <= 0) | (proximal_amp <= 0)
    att_mask = (proximal_amp <= distal_amp) & (~neg_mask)
    amp_mask = (proximal_amp > distal_amp) & (~neg_mask)

    category_masks = [neg_mask, att_mask, amp_mask]
    category_labels = [
        "negative",
        "attenuating",
        "amplifying",
    ]

    return category_masks, category_labels


def get_t_lag_masks(df, num_regs):
    distal_reg = 0
    proximal_reg = num_regs - 1
    proximal_lag = df[f"t_lag_distal_post_r{proximal_reg}"]

    t = 1.5
    sim_mask = (proximal_lag >= -t) & (proximal_lag <= t)
    d_p_mask = proximal_lag > t
    p_d_mask = proximal_lag < -t

    category_masks = [sim_mask, d_p_mask, p_d_mask]
    category_labels = [
        "simultaneous",
        "centripetal",
        "centrifugal",
    ]

    return category_masks, category_labels


def mask_sort_df(df, mask, sort_columns):
    masked_df = df[mask]
    sorted_df = masked_df.sort_values(by=sort_columns)
    return sorted_df


def plot_split_violin(
    data_tup,
    label_tup,
    color_tup,
    ylabel,
    xlabels,
    title=None,
):
    _, ax = plt.subplots()

    points_gauss = 100
    med_line_length = 0.1
    q_line_length = 0.06
    med_width = 1
    q_width = 0.6

    vs = []

    for data, label, color, side in zip(
        data_tup, label_tup, color_tup, ["left", "right"]
    ):
        v = ax.violinplot(
            data,
            points=points_gauss,
            showextrema=False,
        )
        vs.append(v)

        q1, q2, q3 = np.percentile(data, [25, 50, 75], axis=1)
        for i, b in enumerate(v["bodies"]):
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            if side == "left":
                b.get_paths()[0].vertices[:, 0] = np.clip(
                    b.get_paths()[0].vertices[:, 0], -np.inf, m
                )
                ax.hlines([q2[i]], m - med_line_length, m, color="black", lw=med_width)
                ax.hlines(
                    [q1[i], q3[i]], m - q_line_length, m, color="black", lw=q_width
                )
            else:
                b.get_paths()[0].vertices[:, 0] = np.clip(
                    b.get_paths()[0].vertices[:, 0], m, np.inf
                )
                ax.hlines([q2[i]], m, m + med_line_length, color="black", lw=med_width)
                ax.hlines(
                    [q1[i], q3[i]], m, m + q_line_length, color="black", lw=q_width
                )

            b.set_facecolor(color)
            b.set_edgecolor("black")
            b.set_alpha(0.7)

    ax.legend([vs[0]["bodies"][0], vs[1]["bodies"][0]], [label_tup[0], label_tup[1]])

    ax.set_xticks(np.arange(1, len(xlabels) + 1), labels=xlabels)
    ax.set_xlim(0.25, len(xlabels) + 0.75)
    ax.set_xlabel("Cell region")
    ax.set_ylabel(ylabel)

    ax.grid(visible=True, axis="y", alpha=0.7)

    if title is not None:
        ax.set_title(title)


def plot_scatter_categories(
    df,
    x_col,
    y_col,
    colors,
    cat_masks,
    cat_names,
    size_col=None,
    size_lim=None,
    labels=None,
    percent=False,
):
    s = get_dot_size(df, size_col, size_lim)

    plt.figure()
    for cat_mask, cat_name, color in zip(cat_masks, cat_names, colors):
        df_cat = df[cat_mask]
        if percent:
            plt.scatter(
                df_cat[x_col] * 100,
                df_cat[y_col] * 100,
                s=s,
                color=color,
                label=cat_name,
            )
        else:
            plt.scatter(df_cat[x_col], df_cat[y_col], s=s, color=color, label=cat_name)

    plt.legend()
    if labels is None:
        plt.xlabel(x_col)
        plt.ylabel(y_col)
    else:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])


def plot_scatter_cont_color(
    df, x_col, y_col, color_col, cmap, size_col=None, size_lim=None, clim=None
):
    s = get_dot_size(df, size_col, size_lim)

    plt.figure()
    plt.scatter(df[x_col], df[y_col], s=s, c=df[color_col], cmap=cmap)
    cbar = plt.colorbar()
    cbar.set_label(color_col)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if clim is not None:
        plt.clim(clim[0], clim[1])


def plot_split_bar(
    masks,
    labels,
    split_mask,
    colors,
):
    num_cat = len(labels)
    n1s = np.zeros(num_cat)
    n2s = np.zeros(num_cat)
    for i, mask in enumerate(masks):
        mask1 = mask & split_mask
        mask2 = mask & ~split_mask

        n1s[i] = np.sum(mask1)
        n2s[i] = np.sum(mask2)

    n1s = n1s / np.sum(n1s)
    n2s = n2s / np.sum(n2s)

    x = np.arange(num_cat)

    _, ax = plt.subplots()

    w = 0.25

    for x0, n1, n2 in zip(x, n1s, n2s):
        x_c = x0 - w / 2
        x_p = x0 + w / 2

        ax.bar(x_c, n1, edgecolor="black", width=w, color=colors[0])
        ax.bar(x_p, n2, edgecolor="black", width=w, color=colors[1])

    ax.set_xticks(x, labels)
    ax.set_ylabel("Frequency of category")
    plt.tight_layout()


def plot_cell_overview(df, cat_column, colors):
    df_sort = df.sort_values(by=["ptz", "exp_name", "roi_number", "evt_num"])
    amp_cat_s = df_sort[cat_column]
    ptz_s = df_sort["ptz"]
    exp_s = df_sort["exp_name"]

    num_evts = 5
    cell_im = np.zeros((num_evts, 3))
    empty_row = np.ones((num_evts, 3))
    amp_cat_im = []

    y_ticks = []
    y_labels = []

    num_cells_exp = 0

    e_ind = 0
    im_ind = 0
    prev_ptz = None
    prev_exp = None

    ptz_count = 0
    ctrl_count = 0

    for amp_cat, ptz, exp in zip(amp_cat_s, ptz_s, exp_s):
        if prev_ptz is None:
            prev_ptz = ptz
            prev_exp = exp

        if exp != prev_exp:
            print("Fish change")
            y_ticks.append(im_ind - int(num_cells_exp / 2))
            if prev_ptz:
                y_labels.append("ptz " + str(ptz_count))
                ptz_count += 1
            else:
                y_labels.append("ctrl " + str(ctrl_count))
                ctrl_count += 1
            amp_cat_im.append(empty_row)
            im_ind += 1
            prev_exp = exp
            num_cells_exp = 0

        if ptz != prev_ptz:
            print("Group change")
            amp_cat_im.append(empty_row)
            im_ind += 1
            prev_ptz = ptz

        if e_ind == num_evts:
            e_ind = 0
            amp_cat_im.append(copy.deepcopy(cell_im))
            im_ind += 1
            num_cells_exp += 1

        cell_im[e_ind] = colors[amp_cat]
        e_ind += 1

    y_ticks.append(im_ind - int(num_cells_exp / 2))
    if prev_ptz:
        y_labels.append("ptz " + str(ptz_count))
        ptz_count += 1
    else:
        y_labels.append("ctrl " + str(ctrl_count))
        ctrl_count += 1

    amp_cat_im = np.array(amp_cat_im)

    plt.figure()
    plt.imshow(amp_cat_im, aspect="auto")
    plt.yticks(y_ticks, y_labels)
    plt.ylabel("Cells each fish")
    plt.xlabel("Event number")
    plt.tight_layout()


def plot_trace(x, mask, title, ylim=None):
    """
    x: ndarray (evts, time, regions)
    """
    num_regs = x.shape[2]
    x_regs = x[mask]
    av_x = np.mean(x_regs, axis=0)
    reg_colors = REG_CM(np.linspace(0, 1, num_regs))

    t = np.arange(x_regs.shape[1]) / VOLUME_RATE - PRE_EVENT_T

    _, ax = plt.subplots(figsize=(12, 12))
    for reg_num in range(num_regs):
        ax.plot(t, av_x[:, reg_num] * 100, color=reg_colors[reg_num])

    ax.axvline(0, 0.05, 0.95, color="orange", linestyle="--", alpha=0.8)
    ax.axvline(10, 0.05, 0.95, color="darkgray", linestyle="--", alpha=0.8)
    ax.axhline(0, color="black", alpha=0.8)

    ax.set_title(title)
    ax.set_ylabel(r"$\Delta F / F_0$ [%]")
    ax.set_xlabel("Time [sec]")

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    plt.tight_layout()


def plot_split_violin_lag_corr(df, num_regs, interval="whole"):
    ptz_mask = df["ptz"] == True
    ctrl_mask = df["ptz"] == False

    if interval == "whole":
        lag_reg_ptz = [
            df[ptz_mask][f"t_lag_distal_r{reg_num}"] for reg_num in range(1, num_regs)
        ]
        lag_reg_ctrl = [
            df[ctrl_mask][f"t_lag_distal_r{reg_num}"] for reg_num in range(1, num_regs)
        ]

        corr_reg_ptz = [
            df[ptz_mask][f"corr_distal_r{reg_num}"] for reg_num in range(1, num_regs)
        ]
        corr_reg_ctrl = [
            df[ctrl_mask][f"corr_distal_r{reg_num}"] for reg_num in range(1, num_regs)
        ]

        group = "whole"

    elif interval == "post":
        lag_reg_ptz = [
            df[ptz_mask][f"t_lag_distal_post_r{reg_num}"]
            for reg_num in range(1, num_regs)
        ]
        lag_reg_ctrl = [
            df[ctrl_mask][f"t_lag_distal_post_r{reg_num}"]
            for reg_num in range(1, num_regs)
        ]

        corr_reg_ptz = [
            df[ptz_mask][f"corr_distal_post_r{reg_num}"]
            for reg_num in range(1, num_regs)
        ]
        corr_reg_ctrl = [
            df[ctrl_mask][f"corr_distal_post_r{reg_num}"]
            for reg_num in range(1, num_regs)
        ]

        group = "post"
    else:
        print(f"Not a valid interval: {interval}")
        return

    if num_regs == 3:
        labels = ["Middle reg", "Proximal reg"]
    else:
        labels = [f"Middle reg {reg_num}" for reg_num in range(1, num_regs - 1)]
        labels.append("Proximal reg")

    plot_split_violin(
        (lag_reg_ctrl, lag_reg_ptz),
        ("ctrl", "ptz"),
        CTRL_PTZ_COLORS,
        "Lag [s]",
        labels,
        title=group,
    )

    plot_split_violin(
        (corr_reg_ctrl, corr_reg_ptz),
        ("ctrl", "ptz"),
        CTRL_PTZ_COLORS,
        "Correlation",
        labels,
        title=group,
    )


def plot_ex_lr(
    t,
    x,
    t_mask,
    num_regs,
    region_colors,
    labels,
    title,
):
    _, ax = plt.subplots(figsize=(10, 10))
    for reg_num, reg_color, label in zip(range(num_regs), region_colors, labels):
        ax.plot(t[t_mask], x[t_mask, reg_num], color=reg_color, label=label)
        ax.plot(
            t[np.logical_not(t_mask)],
            x[np.logical_not(t_mask), reg_num],
            color=reg_color,
            alpha=0.4,
        )

    ax.set_xlabel("Time [sec]")
    ax.set_ylabel(r"$\Delta$F/F")
    ax.set_title(title)

    vlines = [0, 10]
    vlines_colors = ["orange", "darkgray"]
    vlines_alpha = 0.7

    for i, vline in enumerate(vlines):
        ax.axvline(
            vline,
            0.05,
            0.95,
            color=vlines_colors[i],
            alpha=vlines_alpha,
        )


def plot_ex_corr(x, t_mask, num_regs, region_colors, labels, max_lag, title):
    print(f"x.shape: {x.shape}")
    plt.figure()
    x_ref = x[t_mask, DISTAL_REG]
    for reg_num, reg_color, label in zip(range(num_regs), region_colors, labels):
        x_reg = x[t_mask, reg_num]
        mu_x = np.mean(x_reg)
        mu_ref = np.mean(x_ref)
        norm_x = np.sqrt(np.sum(np.power(x_reg - mu_x, 2)))
        norm_ref = np.sqrt(np.sum(np.power(x_ref - mu_ref, 2)))
        corr = signal.correlate(
            x_reg - np.mean(x_reg), x_ref - np.mean(x_ref), mode="same"
        )
        norm = norm_x * norm_ref

        corr = corr / norm
        lags = np.arange(x_reg.shape[0]) / VOLUME_RATE
        lags = lags - np.amax(lags) / 2

        lag_mask = np.absolute(lags) < max_lag
        lags = lags[lag_mask]
        corr = corr[lag_mask]

        max_ind = np.argmax(corr)
        plt.plot(lags, corr, color=reg_color, label=label, alpha=0.9)
        plt.scatter(
            [lags[max_ind]], [corr[max_ind]], color=reg_color, marker="x", alpha=0.9
        )

    plt.legend()
    plt.xlabel("Lag [sec]")
    plt.ylabel("Correlation")
    plt.grid()
    plt.title(title)


def plot_schematic_time_lag_distal(
    num_regs,
    results_dir,
):
    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)
    d2xdt2 = load_np(results_dir, SECOND_DERIVATIVE_FNAME, num_regs)
    num_evts = dff.shape[0]
    num_frames = dff.shape[1]

    for evt_num in range(num_evts):
        dff_evt = dff[evt_num]
        d2xdt2_evt = d2xdt2[evt_num]
        region_colors = REG_CM(np.linspace(0, 1, num_regs))

        t = (
            np.linspace(
                -PRE_EVENT_T * VOLUME_RATE,
                -PRE_EVENT_T * VOLUME_RATE + num_frames - 1,
                num_frames,
            )
            / VOLUME_RATE
        )
        t_interest = (-PRE_EVENT_T, 20)
        t_mask = np.logical_and(t >= t_interest[0], t < t_interest[1])

        labels_tmp = ["Distal"]
        for i in range(1, num_regs - 1):
            labels_tmp.append(f"Middle {i}")
        labels_tmp.append("Proximal")

        labels = [labels_tmp[reg_num] for reg_num in range(num_regs)]

        p = 25
        phis = arp_model_fit(dff_evt[:, DISTAL_REG], p)
        res_evt = np.zeros(dff_evt.shape)
        for reg_num in range(num_regs):
            res_evt[:, reg_num] = arp_model_res(dff_evt[:, reg_num], phis)

        pred_evt = np.zeros(dff_evt.shape)
        for reg_num in range(num_regs):
            pred_evt[:, reg_num] = arp_model_pred(dff_evt[:, reg_num], phis)

        plot_ex_lr(
            t,
            dff_evt,
            t_mask,
            num_regs,
            region_colors,
            labels,
            "ex dff lr",
        )
        plot_ex_lr(
            t,
            d2xdt2_evt,
            t_mask,
            num_regs,
            region_colors,
            labels,
            "ex d2xdt2 lr",
        )
        plot_ex_lr(
            t,
            res_evt,
            t_mask,
            num_regs,
            region_colors,
            labels,
            f"ex AR({p}) res lr",
        )

        max_lag = 10
        plot_ex_corr(
            dff_evt, t_mask, num_regs, region_colors, labels, max_lag, "ex dff corr"
        )
        plot_ex_corr(
            d2xdt2_evt,
            t_mask,
            num_regs,
            region_colors,
            labels,
            max_lag,
            "ex d2xdt2 corr",
        )
        plot_ex_corr(
            res_evt,
            t_mask,
            num_regs,
            region_colors,
            labels,
            max_lag,
            f"ex AR({p}) res corr",
        )
        plt.show()


def plot_fr_regions(df_ts, t_col, size_col, mask, num_regs, title):
    reg_colors = REG_CM(np.linspace(0, 1, num_regs))

    df_t = df_ts[mask]
    num_bins = int(50)
    thresh = np.linspace(-2.5, 15, num_bins + 1)
    t = np.array([(t_l + t_h) / 2 for t_l, t_h in zip(thresh[:-1], thresh[1:])])
    y_dc = np.flip(np.arange(num_regs))
    # y_dc = np.zeros(num_regs)

    fr = np.zeros((num_bins, num_regs))
    for reg_num in range(num_regs):
        df_t_reg = df_t[df_t["region"] == reg_num]

        for bin_num, t_l, t_h in zip(range(num_bins), thresh[:-1], thresh[1:]):
            df_t_bin = df_t_reg[(df_t_reg[t_col] > t_l) & (df_t_reg[t_col] <= t_h)]
            fr[bin_num, reg_num] = df_t_bin[size_col].sum()

    fr = fr / np.amax(fr, axis=0)
    # fr = fr / np.amax(fr)

    plt.figure()
    for reg_num in range(num_regs):
        plt.plot(t, y_dc[reg_num] + fr[:, reg_num], color=reg_colors[reg_num])

    plt.hlines(y_dc, np.amin(t), np.amax(t), color="darkgray", alpha=0.7)

    plt.xlabel("Time [sec]")
    plt.ylabel("Region/Average event density")
    plt.title(title)


def plot_av_d2xdt2_regions(d2xdt2, mask, title):
    num_regs = d2xdt2.shape[2]
    reg_colors = REG_CM(np.linspace(0, 1, num_regs))

    x = np.mean(
        d2xdt2[
            mask,
            int(PRE_EVENT_T * VOLUME_RATE / 2) : int(
                (PRE_EVENT_T + STIM_DURATION + POST_STIM_T) * VOLUME_RATE
            ),
        ],
        axis=0,
    )
    num_samples = x.shape[0]
    t = np.linspace(-PRE_EVENT_T / 2, STIM_DURATION + POST_STIM_T, num_samples)

    y_dc = np.flip(np.arange(num_regs))
    # y_dc = np.zeros(num_regs)

    x = x / np.amax(np.absolute(x))

    plt.figure()
    for reg_num in range(num_regs):
        plt.plot(t, y_dc[reg_num] + x[:, reg_num], color=reg_colors[reg_num])

    plt.hlines(y_dc, np.amin(t), np.amax(t), color="darkgray", alpha=0.7)

    plt.xlabel("Time [sec]")
    plt.ylabel("Region/Average 2nd derivative")
    plt.title(title)


def calc_relative_diff(series_list, ind_ref):
    rel_diff_series_list = []
    ref_series = series_list[ind_ref]
    for series in series_list:
        rel_diff_series_list.append((series - ref_series) / ref_series * 100)

    return rel_diff_series_list


def amp_regions_generating_code(num_regs, results_dir, fig_dir):
    distal_reg = 0
    proximal_reg = num_regs - 1
    distal_reg_threshold = 20

    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)

    ptz_mask = df_stat["ptz"] == True
    ctrl_mask = df_stat["ptz"] == False

    amp_reg_ptz = [
        df_stat[ptz_mask][f"amp_r{reg_num}"] * 100 for reg_num in range(num_regs)
    ]
    amp_reg_ctrl = [
        df_stat[ctrl_mask][f"amp_r{reg_num}"] * 100 for reg_num in range(num_regs)
    ]

    labels = ["Distal reg"]
    for reg_num in range(1, num_regs - 1):
        labels.append(f"Middle reg {reg_num}")
    labels.append("Proximal reg")

    plot_split_violin(
        (amp_reg_ctrl, amp_reg_ptz),
        ("ctrl", "ptz"),
        CTRL_PTZ_COLORS,
        r"Amplitude $\Delta F / F_0$ [%]",
        labels,
    )

    df_amp_ptz = pd.concat(amp_reg_ptz, axis=1)
    df_amp_ctrl = pd.concat(amp_reg_ctrl, axis=1)

    print(df_amp_ptz.head())

    df_amp_ptz = df_amp_ptz[df_amp_ptz[f"amp_r{distal_reg}"] > distal_reg_threshold]
    df_amp_ctrl = df_amp_ctrl[df_amp_ctrl[f"amp_r{distal_reg}"] > distal_reg_threshold]

    for df in [df_amp_ctrl, df_amp_ptz]:
        for reg_num in range(num_regs):
            reg_amp, ref_amp = df[f"amp_r{reg_num}"], df[f"amp_r{distal_reg}"]
            df[f"rel_diff_amp_r{reg_num}"] = (reg_amp - ref_amp) / ref_amp * 100

    rel_diff_amp_ctrl = [
        df_amp_ctrl[f"rel_diff_amp_r{reg_num}"] for reg_num in range(1, num_regs)
    ]
    rel_diff_amp_ptz = [
        df_amp_ptz[f"rel_diff_amp_r{reg_num}"] for reg_num in range(1, num_regs)
    ]

    plot_split_violin(
        (rel_diff_amp_ctrl, rel_diff_amp_ptz),
        ("ctrl", "ptz"),
        CTRL_PTZ_COLORS,
        "Relative difference amplitude [%]",
        labels[1:],
    )

    for reg_num in range(1, num_regs):
        df_stat[f"diff_amp_r{reg_num}"] = (
            df_stat[f"amp_r{reg_num}"] - df_stat[f"amp_r{distal_reg}"]
        )
        df_stat[f"diff_peak_r{reg_num}"] = (
            df_stat[f"peak_r{reg_num}"] - df_stat[f"peak_r{distal_reg}"]
        )

    plot_scatter_categories(
        df_stat,
        f"amp_r{distal_reg}",
        f"amp_r{proximal_reg}",
        CTRL_PTZ_COLORS,
        (ctrl_mask, ptz_mask),
        ("ctrl", "ptz"),
        labels=(
            r"Distal amplitude $\Delta F / F_0$ [%]",
            r"Proximal amplitude $\Delta F / F_0$ [%]",
        ),
        percent=True,
    )

    """ plot_scatter_categories(
        df_stat,
        f"peak_r{distal_reg}",
        f"peak_r{proximal_reg}",
        CTRL_PTZ_COLORS,
        (ctrl_mask, ptz_mask),
        ("ctrl", "ptz"),
        labels=(
            r"Distal peak $\Delta F / F_0$ [%]",
            r"Proximal peak $\Delta F / F_0$ [%]",
        ),
        percent=True,
    ) """

    masks, labels = get_amp_masks(df_stat, num_regs)

    df_stat["amp_cat"] = ""
    for mask, label in zip(masks, labels):
        df_stat.loc[mask, "amp_cat"] = label

    print(df_stat["amp_cat"])
    print(df_stat.head())
    plot_scatter_categories(
        df_stat,
        f"amp_r{distal_reg}",
        f"amp_r{proximal_reg}",
        AMP_CAT_COLORS,
        masks,
        labels,
        labels=(
            r"Distal amplitude $\Delta F / F_0$ [%]",
            r"Proximal amplitude $\Delta F / F_0$ [%]",
        ),
        percent=True,
    )
    """ plot_scatter_categories(
        df_stat,
        f"peak_r{distal_reg}",
        f"peak_r{proximal_reg}",
        AMP_CAT_COLORS,
        masks,
        labels,
        labels=(
            r"Distal peak $\Delta F / F_0$ [%]",
            r"Proximal peak $\Delta F / F_0$ [%]",
        ),
        percent=True,
    ) """
    plot_scatter_cont_color(
        df_stat,
        f"amp_r{distal_reg}",
        f"amp_r{proximal_reg}",
        f"bl_r{proximal_reg}",
        cm.inferno,
        # clim=(-0.5, 5),
    )

    """ plt.figure()
    plt.scatter(
        df_stat[f"bl_r{distal_reg}"],
        df_stat[f"amp_r{distal_reg}"],
        color="orange",
        label="amp",
        s=5,
    )
    plt.scatter(
        df_stat[f"bl_r{distal_reg}"],
        df_stat[f"peak_r{distal_reg}"],
        color="red",
        label="peak",
        s=5,
    )
    plt.xlabel("Baseline dff")
    plt.ylabel("amp/peak dff")
    plt.title("distal region")
    plt.legend()

    plt.figure()
    plt.scatter(
        df_stat[f"bl_r{proximal_reg}"],
        df_stat[f"amp_r{proximal_reg}"],
        color="orange",
        label="amp",
        s=5,
    )
    plt.scatter(
        df_stat[f"bl_r{proximal_reg}"],
        df_stat[f"peak_r{proximal_reg}"],
        color="red",
        label="peak",
        s=5,
    )
    plt.xlabel("Baseline dff")
    plt.ylabel("amp/peak dff")
    plt.title("proximal region")
    plt.legend() """

    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)

    for mask, label in zip(masks, labels):
        plot_trace(dff, mask, label)

    plot_split_bar(masks, labels, ctrl_mask, CTRL_PTZ_COLORS)

    plot_cell_overview(df_stat, "amp_cat", AMP_CAT_COLORS_DICT)


def t_lag_regions_generating_code(num_regs, results_dir, fig_dir):
    distal_reg = 0
    proximal_reg = num_regs - 1

    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)
    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)
    d2xdt2 = load_np(results_dir, SECOND_DERIVATIVE_FNAME, num_regs)

    # plot_split_violin_lag_corr(df_stat, num_regs, interval="whole")
    plot_split_violin_lag_corr(df_stat, num_regs, interval="post")

    masks, labels = get_t_lag_masks(df_stat, num_regs)
    group_masks, group_labels = [df_stat["ptz"] == ptz for ptz in [False, True]], [
        "ctrl",
        "ptz",
    ]
    for mask, label in zip(masks, labels):
        for group_mask, group_label in zip(group_masks, group_labels):
            c_mask = mask & group_mask
            c_label = group_label + " " + label

            plot_trace(dff, c_mask, c_label)

    corr_threshold = 0.6

    """ df_h = df_stat[df_stat[f"corr_distal_r{proximal_reg}"] >= corr_threshold]
    df_l = df_stat[df_stat[f"corr_distal_r{proximal_reg}"] < corr_threshold]

    plot_split_violin_lag_corr(df_h, num_regs)
    plot_split_violin_lag_corr(df_l, num_regs) """

    """ mask_ptz = df_stat["ptz"] == True
    mask_ctrl = df_stat["ptz"] == False

    mask_h = df_stat[f"corr_distal_r{proximal_reg}"] >= corr_threshold
    mask_l = df_stat[f"corr_distal_r{proximal_reg}"] < corr_threshold

    mask_p_mid = (df_stat[f"t_lag_distal_r{proximal_reg}"] >= 2) & (
        df_stat[f"t_lag_distal_r{proximal_reg}"] < 7
    )
    mask_p_sim = (df_stat[f"t_lag_distal_r{proximal_reg}"] >= -2) & (
        df_stat[f"t_lag_distal_r{proximal_reg}"] < 2
    )

    mask_ptz_sim = mask_ptz & mask_p_sim
    mask_ptz_mid = mask_ptz & mask_p_mid

    mask_ctrl_sim = mask_ctrl & mask_p_sim
    mask_ctrl_mid = mask_ctrl & mask_p_mid

    plot_trace(dff, mask_ptz_sim, f"ptz simultaneous (n={np.sum(mask_ptz_sim)})")
    plot_trace(dff, mask_ctrl_sim, f"ctrl simultaneous (n={np.sum(mask_ctrl_sim)})")
    plot_trace(dff, mask_ptz_mid, f"ptz lag (n={np.sum(mask_ptz_mid)})")
    plot_trace(dff, mask_ctrl_mid, f"ctrl lag (n={np.sum(mask_ctrl_mid)})")

    df_stat_ctrl = df_stat[mask_ctrl]
    df_stat_ptz = df_stat[mask_ptz]

    plot_scatter_cont_color(
        df_stat_ctrl,
        f"amp_r{distal_reg}",
        f"t_lag_distal_r{proximal_reg}",
        f"amp_r{proximal_reg}",
        cm.inferno,
    )

    plot_scatter_cont_color(
        df_stat_ptz,
        f"amp_r{distal_reg}",
        f"t_lag_distal_r{proximal_reg}",
        f"amp_r{proximal_reg}",
        cm.inferno,
    ) """


def create_empty_t_onsets_dict():
    t_onsets_dict = {
        "exp_name": [],
        "crop_id": [],
        "roi_number": [],
        "isi": [],
        "ptz": [],
        "evt_num": [],
        "time_com": [],
        "time_start": [],
        "time_end": [],
        "time_peak": [],
        "dxdt_end": [],
        "diff_dxdt": [],
        "peak": [],
        "region": [],
    }
    return t_onsets_dict


def t_onset_regions_generating_code(num_regs, results_dir, fig_dir):
    proximal_reg = num_regs - 1

    df_t_onsets = load_pickle(results_dir, T_ONSET_FNAME, num_regs)
    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)
    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)
    d2xdt2 = load_np(results_dir, SECOND_DERIVATIVE_FNAME, num_regs)

    ctrl_mask = df_stat["ptz"] == False

    print(f"df_t_onsets.shape: {df_t_onsets.shape}")
    print(f"df_stat.shape: {df_stat.shape}")
    print(f"dff.shape: {dff.shape}")

    t_col = "time_com"
    size_col = "diff_dxdt"

    reg_colors = REG_CM(np.linspace(0, 1, num_regs))
    s_all = np.array(get_dot_size(df_t_onsets, size_col, [5, 30], logarithmic=False))

    masks, labels = get_t_onset_masks(df_stat, num_regs)
    df_stat["t_onset_cat"] = ""

    for ptz in [False, True]:
        group = "ptz" if ptz else "ctrl"
        group_mask = df_t_onsets["ptz"] == ptz
        masked_df_ts = df_t_onsets[group_mask]
        masked_s = s_all[group_mask]

        plt.figure()
        for reg_num in range(num_regs):
            df_ts_reg = masked_df_ts[masked_df_ts["region"] == reg_num]
            x = df_ts_reg[t_col]
            y = num_regs - reg_num + np.linspace(0.25, 0.75, x.size)
            s = masked_s[masked_df_ts["region"] == reg_num]

            plt.scatter(x, y, s=s, color=reg_colors[reg_num])

        plt.title(group)

        plot_fr_regions(df_t_onsets, t_col, size_col, group_mask, num_regs, group)

        group_mask_stat = df_stat["ptz"] == ptz

        for mask, label in zip(masks, labels):
            group_label = group + " " + label
            c_mask = mask & group_mask_stat
            df_stat.loc[c_mask, "t_onset_cat"] = label
            plot_trace(dff, c_mask, group_label)
            plt.ylim(-20, 190)
            plt.xlim(-10, 120)

            masked_df_stat = df_stat[c_mask]

            s_maxes = [
                np.array(
                    get_dot_size(
                        masked_df_stat,
                        f"diff_dxdt_r{reg_num}",
                        [1, 30],
                        logarithmic=False,
                    )
                )
                for reg_num in range(num_regs)
            ]

            plt.figure()
            for reg_num in range(num_regs):
                x = masked_df_stat[f"t_onset_r{reg_num}"]
                y = num_regs - reg_num + np.linspace(0.25, 0.75, x.size)
                s = s_maxes[reg_num]

                plt.scatter(x, y, s=s, color=reg_colors[reg_num])

            plt.title(group_label)

    """ for exp_name in EXP_NAMES:
        group = exp_name
        group_mask = df_t_onsets["exp_name"] == exp_name
        masked_df_ts = df_t_onsets[group_mask]
        masked_s = s_all[group_mask]

        plt.figure()
        for reg_num in range(num_regs):
            df_ts_reg = masked_df_ts[masked_df_ts["region"] == reg_num]
            x = df_ts_reg["time_com"]
            y = num_regs - reg_num + np.linspace(0.25, 0.75, x.size)
            s = masked_s[masked_df_ts["region"] == reg_num]

            plt.scatter(x, y, s=s, color=reg_colors[reg_num])

        plt.title(group)

        stat_group_mask = df_stat["exp_name"] == exp_name
        plot_fr_regions(df_t_onsets, group_mask, num_regs, group)
        plot_trace(dff, stat_group_mask, group) """

    """ s_maxes = [
        np.array(
            get_dot_size(df_stat, f"diff_dxdt_r{reg_num}", [1, 30], logarithmic=False)
        )
        for reg_num in range(num_regs)
    ]
    for ptz in [False, True]:
        group = "ptz" if ptz else "ctrl"
        masked_df_stat = df_stat[df_stat["ptz"] == ptz]

        plt.figure()
        for reg_num in range(num_regs):
            x = masked_df_stat[f"t_onset_r{reg_num}"]
            y = num_regs - reg_num + np.linspace(0.25, 0.75, x.size)
            s = s_maxes[reg_num][df_stat["ptz"] == ptz]

            plt.scatter(x, y, s=s, color=reg_colors[reg_num])

        plt.title(group) """

    plot_split_bar(masks, labels, ctrl_mask, CTRL_PTZ_COLORS)

    plot_cell_overview(df_stat, "t_onset_cat", T_ONSET_CAT_COLORS_DICT)

    """ for mask, label in zip(masks, labels):
        masked_df_ts = df_t_onsets[mask]
        masked_s = s_all[mask]

        plt.figure()
        for reg_num in range(num_regs):
            df_ts_reg = masked_df_ts[masked_df_ts["region"] == reg_num]
            x = df_ts_reg["time_com"]
            y = num_regs - reg_num + np.linspace(0.25, 0.75, x.size)
            s = masked_s[masked_df_ts["region"] == reg_num]

            plt.scatter(x, y, s=s, color=reg_colors[reg_num])

        plt.title(label) """


def amp_t_onset_cluster_masks(df_stat, num_regs):

    amp_masks, amp_labels = get_amp_masks(df_stat, num_regs)
    t_onset_masks, t_onset_labels = get_t_onset_masks(df_stat, num_regs)

    comb_masks, comb_labels = [], []

    plt.figure()
    for amp_mask, amp_label in zip(amp_masks, amp_labels):
        for t_onset_mask, t_onset_label in zip(t_onset_masks, t_onset_labels):
            comb_mask, comb_label = (
                amp_mask & t_onset_mask,
                amp_label + "\n" + t_onset_label,
            )

            comb_masks.append(comb_mask)
            comb_labels.append(comb_label)

    return comb_masks, comb_labels


def amp_t_onset_generating_code(num_regs, results_dir, fig_dir):

    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)
    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)

    masks, labels = amp_t_onset_cluster_masks(df_stat, num_regs)
    ymin, ymax = -100, 300
    for mask, label in zip(masks, labels):
        plot_trace(dff, mask, label, ylim=(ymin, ymax))

    plot_split_bar(masks, labels, df_stat["ptz"] == False, CTRL_PTZ_COLORS)


def main():
    results_dir = generate_global_results_dir()
    fig_dir = generate_figures_dir()
    reg_colors = REG_CM(np.linspace(0, 1, N_REGIONS))

    num_reg = 3
    # amp_regions_generating_code(num_reg, results_dir, fig_dir)
    # plot_schematic_time_lag_distal(num_reg, results_dir)
    # t_lag_regions_generating_code(num_reg, results_dir, fig_dir)

    num_reg = 9
    # t_onset_regions_generating_code(num_reg, results_dir, fig_dir)
    amp_t_onset_generating_code(num_reg, results_dir, fig_dir)

    # t_decay_regions_generating_code(results_dict, reg_colors, evt_num, cell_length)

    # t_onset_cat_generating_code(results_dict, reg_colors, evt_num, cell_length)

    plt.show()


if __name__ == "__main__":
    main()
