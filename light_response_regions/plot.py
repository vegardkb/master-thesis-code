import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import copy
from scipy import signal, stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import chi2, norm, ranksums
from sklearn.metrics import log_loss
from sklearn.cluster import KMeans
import os
import math

from fio import (
    generate_global_results_dir,
    gen_pickle_fname,
    generate_figures_dir,
    gen_npy_fname,
    gen_image_fname,
    gen_pickle_fname,
)

DARKGRAY = (130 / 255, 129 / 255, 129 / 255)
LIGHTBLUE = (101 / 255, 221 / 255, 247 / 255)
DARKBLUE = (99 / 255, 52 / 255, 217 / 255)
ORANGE = (247 / 255, 151 / 255, 54 / 255)
RED = (194 / 255, 35 / 255, 23 / 255)
BROWN = (125 / 255, 69 / 255, 10 / 255)
GREEN = (126 / 255, 186 / 255, 30 / 255)

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
POS_REGS_FNAME = "pos_regs"

N_REGIONS = 8
DISTAL_REG = 0
PROXIMAL_REG = N_REGIONS - 1

CELL_LENGTH = 55

REG_CM = cm.viridis_r

SIG_SINGLE = 0.05
SIG_DOUBLE = 0.01
SIG_TRIPLE = 0.001

MARKER_SIZE = 16
MIN_SIZE = 1
MAX_SIZE = 20
MARKER_ALPHA = 0.6

VOLUME_RATE = 4.86
PRE_EVENT_T = 5

CTRL_PTZ_COLORS = ("gray", "brown")

PEAK_CAT_COLORS = (DARKGRAY, GREEN, BROWN)

T_ONSET_CAT_COLORS = (ORANGE, LIGHTBLUE, DARKBLUE, DARKGRAY)

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
    plt.savefig(
        os.path.join(fig_dir, fname + ".svg"),
        bbox_inches="tight",
    )


def get_region_labels(num_regs):
    reg_labels = []
    reg_labels.append("Distal")
    if num_regs == 3:
        reg_labels.append("Middle")

    else:
        for i in range(1, num_regs - 1):
            reg_labels.append(f"Middle {i}")

    reg_labels.append("Proximal")
    return reg_labels


def get_region_colors(num_regs):
    return REG_CM(np.linspace(0.2, 1, num_regs))


def get_group_labels():
    return ["Control", "PTZ"]


def get_sample_size(df, mask):
    sample_size_str = "num animals = "
    exp_names = df["exp_name"][mask]
    sample_size_str += str(len(exp_names.unique()))

    sample_size_str += "\nnum cells = "

    roi_nums = df["roi_number"][mask]
    exp_roi = exp_names + roi_nums.apply(str)
    sample_size_str += str(len(exp_roi.unique()))

    sample_size_str += f"\nnum trials = {np.sum(mask)}"

    return sample_size_str


def print_plot_prop_speed(df, mask, label_pretty):
    df_mask = df[mask]

    speed = (1 / df_mask["t_onset_slope"]).abs()  # micrometer/sec

    mean_speed = speed.mean()
    std_speed = speed.std()

    print(f"Speed all : {mean_speed} +- {std_speed}")

    speed_groups = []

    for g_label, g_mask in zip(
        ["Control", "PTZ"], [df_mask["ptz"] == False, df_mask["ptz"] == True]
    ):
        g_speed = speed[g_mask]
        speed_groups.append(g_speed)

        mean_speed = g_speed.mean()
        std_speed = g_speed.std()

        print(f"Speed {g_label} : {mean_speed} +- {std_speed}")

    rs_result = ranksums(speed_groups[0], speed_groups[1])
    p_val = rs_result.pvalue

    print(f"P-value group difference : {p_val}")

    ctrl_mask, ptz_mask = ctrl_ptz_mask(df)
    ctrl_speeds = speed[ctrl_mask]
    ptz_speeds = speed[ptz_mask]

    _, ax = plt.subplots()

    w = 0.25

    x0 = 1

    x_c = x0 - 3 / 4 * w
    x_p = x0 + 3 / 4 * w

    ax.bar(
        x_c,
        ctrl_speeds.mean(),
        edgecolor="black",
        width=w,
        color=CTRL_PTZ_COLORS[0],
        alpha=0.75,
        label="Control",
    )
    ax.bar(
        x_p,
        ptz_speeds.mean(),
        edgecolor="black",
        width=w,
        color=CTRL_PTZ_COLORS[1],
        alpha=0.75,
        label="PTZ",
    )

    x_small = np.linspace(-w / 4, w / 4, len(ctrl_speeds))
    ax.scatter(
        x_c + x_small,
        ctrl_speeds,
        s=MARKER_SIZE,
        color=CTRL_PTZ_COLORS[0],
        alpha=MARKER_ALPHA,
    )
    x_small = np.linspace(-w / 4, w / 4, len(ptz_speeds))
    ax.scatter(
        x_p + x_small,
        ptz_speeds,
        s=MARKER_SIZE,
        color=CTRL_PTZ_COLORS[1],
        alpha=MARKER_ALPHA,
    )

    sig = assess_significance(p_val)

    ax.text(x0, 1.1 * mean_speed, sig)

    ax.set_xticks([x0], [label_pretty])
    ax.set_ylabel("Speed [$\mu m / sec$]")
    ax.legend()
    plt.tight_layout()


def ctrl_ptz_mask(df):
    ctrl_mask = df["ptz"] == False
    ptz_mask = df["ptz"] == True
    return ctrl_mask, ptz_mask


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


def linear_model(x, y):
    X = np.reshape(x, (-1, 1))

    linreg = LinearRegression().fit(X, y)
    y_pred = linreg.predict(X)

    return linreg, X, y_pred


def linear_mixed_model(x, a, y):
    X = np.stack([x, a], axis=1)
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    X = poly.fit_transform(X)

    linreg = LinearRegression().fit(X, y)
    y_pred = linreg.predict(X)

    return linreg, X, y_pred


def log_likelihood(y, y_pred, sigma):
    return np.sum(norm.logpdf(y, loc=y_pred, scale=sigma))


def rsquared(y, y_pred):
    res_var = np.var(y - y_pred)
    tot_var = np.var(y)
    return (tot_var - res_var) / tot_var


def likelihood_ratio_test(x, a, y):
    _, _, lin_y = linear_model(x, y)
    _, _, mix_y = linear_mixed_model(x, a, y)
    df = 2

    sigma_lin = np.std(lin_y - y, ddof=2)
    sigma_mix = np.std(mix_y - y, ddof=4)

    lin_log_likelihood = log_likelihood(y, lin_y, sigma_lin)
    mix_log_likelihood = log_likelihood(y, mix_y, sigma_mix)

    G = -2 * (lin_log_likelihood - mix_log_likelihood)
    p_val = 1 - chi2.cdf(G, df)

    return p_val


def assess_significance(pvalue):
    sig = "ns"
    if pvalue < SIG_SINGLE:
        sig = "*"
    if pvalue < SIG_DOUBLE:
        sig = sig + "*"
    if pvalue < SIG_TRIPLE:
        sig = sig + "*"

    return sig


def stat_mask_to_t_onset_mask(df_stat, df_t_onset, mask):
    """
    Need to identify exp_name, roi_num, evt_num. Then we can mask df_t_onset

    This is most likely super slow.. Please use faster methods.
    """
    onset_mask = df_t_onset["exp_name"] == "no"

    for row_ind, stat_row in df_stat.iterrows():
        mask_val = mask[row_ind]
        if mask_val:
            onset_mask = onset_mask | (
                (df_t_onset["exp_name"] == stat_row["exp_name"])
                & (df_t_onset["roi_number"] == stat_row["roi_number"])
                & (df_t_onset["evt_num"] == stat_row["evt_num"])
            )

    return onset_mask


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
    lon_s, lon_e = 0, 4
    loff_s, loff_e = 10, 14

    mean_t, med_t = 0, 0
    r_t = 0.25

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

    sync = np.ones(df.shape[0], dtype=bool)
    for reg_num in range(num_regs):
        t_onset_reg = df[f"t_onset_r{reg_num}"]
        sync = sync & (
            ((t_onset_reg < lon_e) & (t_onset_reg > lon_s))
            | ((t_onset_reg < loff_e) & (t_onset_reg > loff_s))
        )
    wave = ~sync & (
        (df["t_onset_lag_median"] > med_t)
        & (df["t_onset_lag_mean"] > mean_t)
        & (df["t_onset_rsq"] > r_t)
    )
    wave_r = (
        ~wave
        & ~sync
        & (
            (df["t_onset_lag_median"] < med_t)
            & (df["t_onset_lag_mean"] < mean_t)
            & (df["t_onset_rsq"] > r_t)
        )
    )
    undefined = ~wave & ~sync & ~wave_r

    category_masks = [sync, wave, wave_r, undefined]
    category_labels = ["simultaneous", "calcium_wave", "calcium_wave_r", "undefined"]
    categor_labels_pretty = [
        "simultaneous",
        "centripetal\ncalcium wave",
        "centrifugal\ncalcium wave",
        "undefined",
    ]

    return category_masks, category_labels, categor_labels_pretty


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
        "distal_dominant",
        "proximal_dominant",
    ]
    categor_labels_pretty = [
        "negative",
        "distal-dominant",
        "proximal-dominant",
    ]
    return category_masks, category_labels, categor_labels_pretty


def get_peak_masks(df, num_regs):
    distal_reg = 0
    proximal_reg = num_regs - 1
    distal_peak = df[f"peak_r{distal_reg}"]
    proximal_peak = df[f"peak_r{proximal_reg}"]
    distal_amp = df[f"amp_r{distal_reg}"]
    proximal_amp = df[f"amp_r{proximal_reg}"]

    neg_mask = (distal_amp <= 0) | (proximal_amp <= 0)
    att_mask = (proximal_peak <= distal_peak) & (~neg_mask)
    amp_mask = (proximal_peak > distal_peak) & (~neg_mask)

    category_masks = [neg_mask, att_mask, amp_mask]
    category_labels = [
        "negative",
        "distal_dominant",
        "proximal_dominant",
    ]
    categor_labels_pretty = [
        "negative",
        "distal-dominant",
        "proximal-dominant",
    ]
    return category_masks, category_labels, categor_labels_pretty


def get_t_lag_masks(df, num_regs, lag_type="res"):
    distal_reg = 0
    proximal_reg = num_regs - 1
    proximal_lag = df[f"t_lag_{lag_type}_r{proximal_reg}"]

    t = 2.5
    sim_mask = (proximal_lag >= -t) & (proximal_lag <= t)
    d_p_mask = proximal_lag > t
    p_d_mask = proximal_lag < -t

    category_masks = [sim_mask, d_p_mask, p_d_mask]
    category_labels = [
        "simultaneous",
        "centripetal",
        "centrifugal",
    ]
    category_labels_pretty = [
        "simultaneous",
        "centripetal",
        "centrifugal",
    ]

    return category_masks, category_labels, category_labels_pretty


def get_amp_t_onset_masks(df_stat_t, df_stat_amp, num_regs_t, num_regs_amp):
    amp_masks, amp_labels, amp_labels_pretty = get_amp_masks(df_stat_amp, num_regs_amp)
    t_onset_masks, t_onset_labels, t_onset_labels_pretty = get_t_onset_masks(
        df_stat_t, num_regs_t
    )

    comb_masks, comb_labels, comb_labels_pretty = [], [], []

    plt.figure()
    for amp_mask, amp_label, amp_label_pretty in zip(
        amp_masks, amp_labels, amp_labels_pretty
    ):
        if amp_label == "negative":
            continue

        for t_onset_mask, t_onset_label, t_onset_label_pretty in zip(
            t_onset_masks, t_onset_labels, t_onset_labels_pretty
        ):
            if t_onset_label == "undefined":
                continue

            comb_mask, comb_label = (
                amp_mask & t_onset_mask,
                amp_label + "\n" + t_onset_label,
            )
            comb_mask = amp_mask & t_onset_mask
            comb_label = amp_label + "_" + t_onset_label
            comb_label_pretty = amp_label_pretty + "\n" + t_onset_label_pretty

            comb_masks.append(comb_mask)
            comb_labels.append(comb_label)
            comb_labels_pretty.append(comb_label_pretty)

    return comb_masks, comb_labels, comb_labels_pretty


def get_peak_t_onset_masks(df_stat_t, df_stat_peak, num_regs_t, num_regs_peak):
    peak_masks, peak_labels, peak_labels_pretty = get_peak_masks(
        df_stat_peak, num_regs_peak
    )
    t_onset_masks, t_onset_labels, t_onset_labels_pretty = get_t_onset_masks(
        df_stat_t, num_regs_t
    )

    comb_masks, comb_labels, comb_labels_pretty = [], [], []

    plt.figure()
    for peak_mask, peak_label, peak_label_pretty in zip(
        peak_masks, peak_labels, peak_labels_pretty
    ):
        if peak_label == "negative":
            continue

        for t_onset_mask, t_onset_label, t_onset_label_pretty in zip(
            t_onset_masks, t_onset_labels, t_onset_labels_pretty
        ):
            if t_onset_label == "undefined":
                continue

            comb_mask, comb_label = (
                peak_mask & t_onset_mask,
                peak_label + "\n" + t_onset_label,
            )
            comb_mask = peak_mask & t_onset_mask
            comb_label = peak_label + "_" + t_onset_label
            comb_label_pretty = peak_label_pretty + "\n" + t_onset_label_pretty

            comb_masks.append(comb_mask)
            comb_labels.append(comb_label)
            comb_labels_pretty.append(comb_label_pretty)

    return comb_masks, comb_labels, comb_labels_pretty


def plot_light_on_off(ax, lw=0.7, ymin=0.05, ymax=0.95):
    ax.axvline(0, ymin, ymax, color="orange", linestyle="--", linewidth=lw, alpha=0.75)
    ax.axvline(
        10, ymin, ymax, color="darkgray", linestyle="--", linewidth=lw, alpha=0.75
    )


def plot_amp_times(ax, x, amp5s_color, amprnd_color, lw=0.7):
    ax.axvline(
        5,
        np.amin(x),
        np.amax(x),
        color=amp5s_color,
        alpha=0.75,
        linewidth=lw,
    )
    ax.axvline(
        110,
        np.amin(x),
        np.amax(x),
        color=amprnd_color,
        alpha=0.75,
        linewidth=lw,
    )
    """ ax.plot(
        t_peak,
        np.flip(x),
        color=amp_color,
        alpha=0.75,
        linewidth=lw,
    ) """


def add_scale_bar(
    ax, size_h, size_v, label, loc="upper center", pad=0.1, borderpad=-2, color="black"
):
    asb = AnchoredSizeBar(
        ax.transData,
        size=size_h,
        size_vertical=size_v,
        label=label,
        loc=loc,
        pad=pad,
        borderpad=borderpad,
        frameon=False,
        color=color,
    )
    ax.add_artist(asb)


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
    axlim=None,
    xlim=None,
    ylim=None,
    title=None,
    mask_mask=None,
    mixreg=None,
):
    s = get_dot_size(df, size_col, size_lim)

    plt.figure()
    for ind_cat, cat_mask, cat_name, color in zip(
        range(len(cat_masks)), cat_masks, cat_names, colors
    ):
        if mask_mask is not None:
            cat_mask = copy.deepcopy(cat_mask)
            cat_mask[~mask_mask] = False
        df_cat = df[cat_mask]
        if percent:
            plt.scatter(
                df_cat[x_col] * 100,
                df_cat[y_col] * 100,
                s=s,
                color=color,
                label=cat_name,
                alpha=MARKER_ALPHA,
            )
        else:
            plt.scatter(
                df_cat[x_col],
                df_cat[y_col],
                s=s,
                color=color,
                label=cat_name,
                alpha=MARKER_ALPHA,
            )

        if mixreg:
            x = np.array([df_cat[x_col].min(), df_cat[x_col].max()])
            a = np.ones(x.shape) * ind_cat
            X = np.stack([x, a], axis=1)
            poly = PolynomialFeatures(interaction_only=True, include_bias=False)
            X = poly.fit_transform(X)
            y_pred = mixreg.predict(X)

            print(f"X: {X}")
            print(f"y_pred: {y_pred}")

            if percent:
                plt.plot(
                    x * 100,
                    y_pred * 100,
                    linestyle="dashed",
                    color=color,
                    alpha=MARKER_ALPHA,
                )
            else:
                plt.plot(x, y_pred, linestyle="dashed", color=color, alpha=MARKER_ALPHA)

    plt.legend()
    if labels is None:
        plt.xlabel(x_col)
        plt.ylabel(y_col)
    else:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])

    if axlim is not None:
        plt.xlim((axlim[0], axlim[1]))
        plt.ylim((axlim[0], axlim[1]))

    if xlim is not None:
        plt.xlim((xlim[0], xlim[1]))
    if ylim is not None:
        plt.ylim((ylim[0], ylim[1]))

    if title is not None:
        plt.title(title)


def scatter_hist(x, y, ax, ax_histx, ax_histy, binsize, s, color, alpha, label):
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    ax.scatter(x, y, s=s, color=color, alpha=alpha, label=label)

    x_min, x_max = np.amin(x), np.amax(x)
    x_bins = np.linspace(x_min, x_max, int((x_max - x_min) / binsize))
    y_min, y_max = np.amin(y), np.amax(y)
    y_bins = np.linspace(y_min, y_max, int((y_max - y_min) / binsize))

    x_num = []
    x_0 = []
    for t_l, t_h in zip(x_bins[:-1], x_bins[1:]):
        x_num.append(np.sum(np.logical_and(x >= t_l, x < t_h)))
        x_0.append((t_l + t_h) / 2)

    y_num = []
    y_0 = []
    for t_l, t_h in zip(y_bins[:-1], y_bins[1:]):
        y_num.append(np.sum(np.logical_and(y >= t_l, y < t_h)))
        y_0.append((t_l + t_h) / 2)

    ax_histx.plot(x_0, x_num, color=color)
    ax_histy.plot(y_num, y_0, color=color)

    xnum_max = max(x_num)
    ynum_max = max(y_num)
    return xnum_max, ynum_max


def plot_scatter_hist_categories(
    df,
    x_col,
    y_col,
    colors,
    cat_masks,
    cat_names,
    labels=None,
    xlim=None,
    ylim=None,
    binsize=0.1,
    percent=False,
    size_col=None,
    size_lim=None,
    mixreg=False,
    tick_inc=10,
    x_ticks=[-100, 0, 100, 200],
    y_ticks=[-100, 0, 100, 200, 300, 400, 500],
):
    fig = plt.figure()
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.05,
        hspace=0.05,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    s = get_dot_size(df, size_col, size_lim)
    dot_alpha = 0.5
    line_alpha = 0.8

    xmaxes, ymaxes = [], []

    for ind_cat, cat_mask, cat_name, color in zip(
        range(len(cat_masks)), cat_masks, cat_names, colors
    ):
        df_cat = df[cat_mask]
        if percent:
            xmax, ymax = scatter_hist(
                df_cat[x_col] * 100,
                df_cat[y_col] * 100,
                ax,
                ax_histx,
                ax_histy,
                binsize,
                s,
                color,
                dot_alpha,
                cat_name,
            )
        else:
            xmax, ymax = scatter_hist(
                df_cat[x_col],
                df_cat[y_col],
                ax,
                ax_histx,
                ax_histy,
                binsize,
                s,
                color,
                dot_alpha,
                cat_name,
            )
        xmaxes.append(xmax), ymaxes.append(ymax)
        if mixreg:
            x = np.array([df_cat[x_col].min(), df_cat[x_col].max()])
            a = np.ones(x.shape) * ind_cat
            X = np.stack([x, a], axis=1)
            poly = PolynomialFeatures(interaction_only=True, include_bias=False)
            X = poly.fit_transform(X)
            y_pred = mixreg.predict(X)

            print(f"X: {X}")
            print(f"y_pred: {y_pred}")

            if percent:
                ax.plot(
                    x * 100,
                    y_pred * 100,
                    linestyle="dashed",
                    color=color,
                    alpha=line_alpha,
                )
            else:
                ax.plot(x, y_pred, linestyle="dashed", color=color, alpha=line_alpha)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    histx_ticks = [
        i * tick_inc for i in range(max(1, math.ceil(max(xmaxes) / tick_inc) + 1))
    ]
    histy_ticks = [
        i * tick_inc for i in range(max(1, math.ceil(max(ymaxes) / tick_inc) + 1))
    ]

    ax_histx.set_yticks(histx_ticks)
    ax_histy.set_xticks(histy_ticks)

    ax.legend(loc="upper left")
    if labels is None:
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    else:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

    ax_histx.set_ylabel("Counts")
    ax_histy.set_xlabel("Counts")

    ax_histx.axhline(color="darkgray", alpha=0.7)
    ax_histy.axvline(color="darkgray", alpha=0.7)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])


def plot_split_bar(
    exp_names: pd.Series,
    masks,
    labels,
    split_mask,
    colors,
    test_sig=False,
    scale=1,
):
    num_cat = len(labels)
    ns = []
    for s_mask in [split_mask, ~split_mask]:
        s_exp_names = exp_names[s_mask]
        unique_names = s_exp_names.unique()
        n = np.zeros((num_cat, len(unique_names)))

        for i, mask in enumerate(masks):
            for j, name in enumerate(unique_names):
                n[i, j] = (
                    np.sum(mask & (exp_names == name))
                    / np.sum(exp_names == name)
                    * scale
                )

        ns.append(n)

    n1s = ns[0]
    n2s = ns[1]
    m1s = np.mean(n1s, axis=1)
    m2s = np.mean(n2s, axis=1)

    x = np.arange(num_cat)

    _, ax = plt.subplots()

    w = 0.25

    for i, x0, m1, m2 in zip(range(num_cat), x, m1s, m2s):
        x_c = x0 - w / 2
        x_p = x0 + w / 2

        n1 = n1s[i]
        n2 = n2s[i]

        if i == 0:
            ax.bar(
                x_c,
                m1,
                edgecolor="black",
                width=w,
                color=colors[0],
                alpha=0.75,
                label="Control",
            )
            ax.bar(
                x_p,
                m2,
                edgecolor="black",
                width=w,
                color=colors[1],
                alpha=0.75,
                label="PTZ",
            )
        else:
            ax.bar(
                x_c,
                m1,
                edgecolor="black",
                width=w,
                color=colors[0],
                alpha=0.75,
            )
            ax.bar(
                x_p,
                m2,
                edgecolor="black",
                width=w,
                color=colors[1],
                alpha=0.75,
            )

        x_small = np.linspace(-w / 4, w / 4, n1.shape[0])
        ax.scatter(
            x_c + x_small, n1, s=MARKER_SIZE, color=colors[0], alpha=MARKER_ALPHA
        )
        x_small = np.linspace(-w / 4, w / 4, n2.shape[0])
        ax.scatter(
            x_p + x_small, n2, s=MARKER_SIZE, color=colors[1], alpha=MARKER_ALPHA
        )

        if test_sig:
            rs_result = ranksums(n1, n2)
            p_val = rs_result.pvalue
            sig = assess_significance(p_val)

            ax.text(x0, 0.8, sig)

    ax.set_ylim(0, 1.1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Frequency of category")
    ax.legend()
    plt.tight_layout()


def reg_analysis(
    exp_names: pd.Series,
    masks,
    labels,
    split_mask,
    split_labels,
    colors,
    scale=1,
):
    num_cat = len(labels)
    ys = []
    xs = []
    for s_mask in [split_mask, ~split_mask]:
        s_exp_names = exp_names[s_mask]
        unique_names = s_exp_names.unique()
        y = np.zeros((num_cat, len(unique_names)))
        x = np.zeros((num_cat, len(unique_names)))

        for i, mask in enumerate(masks):
            for j, name in enumerate(unique_names):
                y[i, j] = (
                    np.sum(mask & (exp_names == name))
                    / np.sum(exp_names == name)
                    * scale
                )
                x[i, j] = i

        ys.append(y.reshape(-1))
        xs.append(x.reshape(-1))

    _, ax = plt.subplots()

    print("\nTest slope non-zero:")
    sig_str = ""
    rng = np.random.default_rng()
    for x, y, color, s_label in zip(xs, ys, colors, split_labels):
        lin_res = stats.linregress(x, y)

        print(f"pvalue {s_label}: {lin_res.pvalue}")
        sig = assess_significance(lin_res.pvalue)

        num_dots = x.shape[0]

        ax.scatter(
            x + (rng.random(num_dots) - 0.5) / 4,
            y,
            s=MARKER_SIZE,
            color=color,
            alpha=MARKER_ALPHA,
            label=s_label,
        )

        ax.plot(
            [x[0], x[-1]],
            [lin_res.intercept, lin_res.intercept + lin_res.slope * x[-1]],
            color=color,
            alpha=MARKER_ALPHA,
            linestyle="dashed",
        )

    ax.set_ylabel("Frequency of calcium wave")
    ax.set_xticks(np.arange(num_cat), labels)
    ax.legend()


def plot_split_bar_ptz_inside_out(
    ptz, var_list, var_name_list, hatch_list, ylabel, ylim=None
):
    ctrl = np.logical_not(ptz)

    n1s = [var[ctrl] for var in var_list]
    n2s = [var[ptz] for var in var_list]

    m1s = [np.mean(n1) for n1 in n1s]
    m2s = [np.mean(n2) for n2 in n2s]

    num_var = len(var_list)

    w = 0.25

    _, ax = plt.subplots()

    h_color = "gray"
    h_alpha = 0.99
    h_lw = 0.3

    e_color = "black"
    e_lw = 0.3

    f_alpha = 0.8

    m_size = 8
    m_alpha = 0.75

    legend_elements = []
    legend_elements.append(
        Patch(
            facecolor=CTRL_PTZ_COLORS[0],
            edgecolor=e_color,
            lw=e_lw,
            alpha=f_alpha,
            label="Control",
        )
    )
    legend_elements.append(
        Patch(
            facecolor=CTRL_PTZ_COLORS[1],
            edgecolor=e_color,
            lw=e_lw,
            alpha=f_alpha,
            label="PTZ",
        )
    )
    for var_name, h_str in zip(var_name_list, hatch_list):
        legend_elements.append(
            Patch(
                facecolor="none",
                edgecolor=e_color,
                lw=h_lw,
                alpha=h_alpha,
                label=var_name,
                hatch=h_str,
            )
        )

    xs = []
    for group, ns, ms, x0, f_color in zip(
        ["Control", "PTZ"], [n1s, n2s], [m1s, m2s], range(2), CTRL_PTZ_COLORS
    ):
        dx = np.linspace(-w / 1.5, w / 1.5, num=num_var, endpoint=True)

        x = x0 + dx

        for var_num, h_str in zip(range(num_var), hatch_list):
            # draw hatch
            if h_str:
                ax.bar(
                    x[var_num],
                    ms[var_num],
                    width=w,
                    edgecolor=f_color,
                    color="none",
                    alpha=h_alpha,
                    hatch=h_str,
                    lw=h_lw,
                    zorder=1,
                )
            # draw edge
            ax.bar(
                x[var_num],
                ms[var_num],
                width=w,
                edgecolor=e_color,
                color=f_color,
                alpha=f_alpha,
                linewidth=e_lw,
                zorder=0,
            )
            xs.append(x[var_num])

            n = ns[var_num]
            x_small = np.linspace(-w / 4, w / 4, n.shape[0])
            ax.scatter(x[var_num] + x_small, n, s=m_size, color=f_color, alpha=m_alpha)

            if var_num > 0:
                rs_result = ranksums(n, ns[0])
                p_val = rs_result.pvalue
                sig = assess_significance(p_val)

                ax.text(x[var_num], 0.8, sig)

    print("Control vs PTZ")
    for i, var_name in enumerate(var_name_list):
        rs_result = ranksums(n1s[i], n2s[i])
        print(f"{var_name}: {rs_result.pvalue}")

    ax.set_ylim(top=1.1)
    ax.axhline(color="black")
    ax.set_ylim(-0.2, 1)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    # ax.set_xticks(xs)
    ax.set_xticks([])
    ax.spines[["right", "top", "bottom"]].set_visible(False)
    ax.set_ylabel(ylabel)
    ax.legend(handles=legend_elements)
    plt.tight_layout()


def plot_split_bar_ptz(ptz, var_list, var_name_list, ylabel, ylim=None):
    ctrl = np.logical_not(ptz)

    n1s = [var[ctrl] for var in var_list]
    n2s = [var[ptz] for var in var_list]

    m1s = [np.mean(n1) for n1 in n1s]
    m2s = [np.mean(n2) for n2 in n2s]

    num_var = len(var_list)

    w = 0.25
    marker_s = 8
    marker_alpha = 0.75
    bar_alpha = 0.75

    _, ax = plt.subplots()

    for i, x0, m1, m2, n1, n2 in zip(
        range(num_var),
        np.arange(num_var),
        m1s,
        m2s,
        n1s,
        n2s,
    ):
        x_c = x0 - w / 2
        x_p = x0 + w / 2

        if i == 0:
            ax.bar(
                x_c,
                m1,
                edgecolor="black",
                width=w,
                color=CTRL_PTZ_COLORS[0],
                alpha=bar_alpha,
                label="Control",
            )
            ax.bar(
                x_p,
                m2,
                edgecolor="black",
                width=w,
                color=CTRL_PTZ_COLORS[1],
                alpha=bar_alpha,
                label="PTZ",
            )
        else:
            ax.bar(
                x_c,
                m1,
                edgecolor="black",
                width=w,
                color=CTRL_PTZ_COLORS[0],
                alpha=bar_alpha,
            )
            ax.bar(
                x_p,
                m2,
                edgecolor="black",
                width=w,
                color=CTRL_PTZ_COLORS[1],
                alpha=bar_alpha,
            )

        x_small = np.linspace(-w / 4, w / 4, n1.shape[0])
        ax.scatter(
            x_c + x_small, n1, s=marker_s, color=CTRL_PTZ_COLORS[0], alpha=marker_alpha
        )
        x_small = np.linspace(-w / 4, w / 4, n2.shape[0])
        ax.scatter(
            x_p + x_small, n2, s=marker_s, color=CTRL_PTZ_COLORS[1], alpha=marker_alpha
        )

        rs_result = ranksums(n1, n2)
        p_val = rs_result.pvalue
        sig = assess_significance(p_val)

        ax.text(x0, 0.8, sig)

    ax.set_ylim(top=1.1)
    ax.axhline(color="black")
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.set_xticks(np.arange(num_var), var_name_list)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()


def plot_slope_plot(
    df,
    x_col,
    y_col,
    colors,
    cat_masks,
    cat_names,
):

    x = df[x_col]
    y = df[y_col]

    fig, ax = plt.subplots()
    for c_mask, c_name, color in zip(cat_masks, cat_names, colors):
        x_c = x[c_mask]
        y_c = y[c_mask]
        ax.plot(
            [0, 1],
            [x_c.mean(), y_c.mean()],
            color=color,
            label=c_name,
            marker="o",
            alpha=0.8,
        )

        for x_p, y_p in zip(x_c, y_c):
            ax.plot([0, 1], [x_p, y_p], color=color, alpha=0.1)

    plt.legend()
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Region")
    ax.set_xticks([0, 1], ["Distal", "Proximal"])


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

    ax.set_xlabel("Micrometers")
    ax.set_ylabel("Micrometers")

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


def plot_trace(
    x,
    mask,
    title=None,
    ylim=None,
    xlim=None,
    t_max=None,
    ylabel=None,
    plot_std=False,
    plot_sem=False,
    plot_dist=False,
    plot_all=False,
    xticks=None,
    yticks=None,
):
    """
    x: ndarray (evts, time, regions)
    """
    num_regs = x.shape[2]
    x_regs = x[mask] * 100
    t = np.arange(x_regs.shape[1]) / VOLUME_RATE - PRE_EVENT_T

    if t_max is not None:
        t_mask = t < t_max
        t = t[t_mask]
        x_regs = x_regs[:, t_mask]

    av_x = np.mean(x_regs, axis=0)
    if plot_dist:
        av_x = np.median(x_regs, axis=0)

    reg_colors = get_region_colors(num_regs)
    av_alpha = 0.95
    std_alpha = 0.15
    lw_std = 0.25
    fill_alpha = 0.075
    single_alpha = 0.2
    single_lw = 0.3

    reg_labels = get_region_labels(num_regs)

    _, ax = plt.subplots()

    plot_light_on_off(ax)
    ax.axhline(0, color="black", alpha=0.8)

    if plot_std:
        std_x = np.std(x_regs, axis=0)
        for reg_num in range(num_regs):
            ax.plot(
                t,
                av_x[:, reg_num] + std_x[:, reg_num],
                color=reg_colors[reg_num],
                alpha=std_alpha,
                lw=lw_std,
            )
            ax.plot(
                t,
                av_x[:, reg_num] - std_x[:, reg_num],
                color=reg_colors[reg_num],
                alpha=std_alpha,
                lw=lw_std,
            )
            ax.fill_between(
                t,
                av_x[:, reg_num] - std_x[:, reg_num],
                av_x[:, reg_num] + std_x[:, reg_num],
                color=reg_colors[reg_num],
                alpha=fill_alpha,
            )

    if plot_sem:
        sem_x = np.std(x_regs, axis=0) / np.sqrt(x_regs.shape[0])
        for reg_num in range(num_regs):
            ax.plot(
                t,
                av_x[:, reg_num] + sem_x[:, reg_num],
                color=reg_colors[reg_num],
                alpha=std_alpha,
                lw=lw_std,
            )
            ax.plot(
                t,
                av_x[:, reg_num] - sem_x[:, reg_num],
                color=reg_colors[reg_num],
                alpha=std_alpha,
                lw=lw_std,
            )
            ax.fill_between(
                t,
                av_x[:, reg_num] - sem_x[:, reg_num],
                av_x[:, reg_num] + sem_x[:, reg_num],
                color=reg_colors[reg_num],
                alpha=fill_alpha,
            )

    if plot_dist:
        for i, prop in enumerate([30, 20, 10, 5]):
            pos_percentile_x = np.percentile(x_regs, 50 + prop / 2, axis=0)
            neg_percentile_x = np.percentile(x_regs, 50 - prop / 2, axis=0)
            for reg_num in range(num_regs):
                if i == 0:
                    ax.plot(
                        t,
                        pos_percentile_x[:, reg_num],
                        color=reg_colors[reg_num],
                        alpha=std_alpha,
                        lw=lw_std,
                    )
                    ax.plot(
                        t,
                        neg_percentile_x[:, reg_num],
                        color=reg_colors[reg_num],
                        alpha=std_alpha,
                        lw=lw_std,
                    )
                ax.fill_between(
                    t,
                    neg_percentile_x[:, reg_num],
                    pos_percentile_x[:, reg_num],
                    color=reg_colors[reg_num],
                    alpha=fill_alpha,
                )

    if plot_all:
        num_trials = x_regs.shape[0]
        for trial_num in range(num_trials):
            for reg_num in range(num_regs):
                ax.plot(
                    t,
                    x_regs[trial_num, :, reg_num],
                    color=reg_colors[reg_num],
                    alpha=single_alpha,
                    lw=single_lw,
                )

    for reg_num in range(num_regs):
        if (not (reg_num == 0 or reg_num == (num_regs - 1))) and num_regs > 3:
            ax.plot(
                t,
                av_x[:, reg_num],
                color=reg_colors[reg_num],
                alpha=av_alpha,
            )
        else:
            ax.plot(
                t,
                av_x[:, reg_num],
                color=reg_colors[reg_num],
                alpha=av_alpha,
                label=reg_labels[reg_num],
            )

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        if plot_std:
            ax.set_ylabel(r"$\Delta F / F_0$ (%, mean $\pm$ STD)")
        elif plot_sem:
            ax.set_ylabel(r"$\Delta F / F_0$ (%, mean $\pm$ SEM)")
        else:
            ax.set_ylabel(r"$\Delta F / F_0$ (%)")
    ax.set_xlabel("Time [sec]")

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])

    if title is not None:
        ax.set_title(title)

    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    ax.legend()


def plot_trace_stacked(
    x,
    mask,
    t_min=None,
    t_max=None,
    shade_peaks=False,
    second_derivative=False,
):

    num_regs = x.shape[2]
    x_regs = x[mask]
    t = np.arange(x_regs.shape[1]) / VOLUME_RATE - PRE_EVENT_T

    if t_max is not None:
        t_mask = t <= t_max
        t = t[t_mask]
        x_regs = x_regs[:, t_mask]

    if t_min is not None:
        t_mask = t >= t_min
        t = t[t_mask]
        x_regs = x_regs[:, t_mask]

    av_x = np.mean(x_regs, axis=0)

    delta_y = np.amax(np.absolute(av_x))
    raw_scale_factor = 0.25
    if not second_derivative:
        delta_y = delta_y * raw_scale_factor

    colors = get_region_colors(num_regs)

    _, ax = plt.subplots()

    plot_light_on_off(ax)

    for reg_num in range(num_regs):
        dff_reg = av_x[:, reg_num]
        ax.plot(
            t,
            dff_reg - delta_y * reg_num,
            color=colors[reg_num],
            label=f"ROI {reg_num+1}",
            alpha=0.8,
        )
        if shade_peaks:
            ax.fill_between(
                t,
                np.zeros(t.shape) - delta_y * reg_num,
                dff_reg - delta_y * reg_num,
                where=dff_reg > 0,
                color=colors[reg_num],
                alpha=0.4,
            )

    scalebar_width_pixels = 10
    inv = ax.transData.inverted()
    points = inv.transform([(0, 0), (scalebar_width_pixels, scalebar_width_pixels)])
    scale_x = points[0, 1] - points[0, 0]
    scale_y = points[1, 1] - points[1, 0]

    bar_size = 1  # seconds
    loc = "lower center"
    asb = AnchoredSizeBar(
        ax.transData,
        size=bar_size * VOLUME_RATE,
        size_vertical=scale_y,
        label=f"{str(bar_size)} sec",
        loc=loc,
        pad=0.1,
        borderpad=-2,
        frameon=False,
        color="black",
    )
    ax.add_artist(asb)

    ylabel = "% $\Delta F / F_0$/s" if second_derivative else "% $\Delta F / F_0$"

    bar_size = max(((delta_y / 5) // 0.5) * 0.5, 0.5)  # dff
    if second_derivative:
        bar_size = max(((delta_y / 5) // 0.05) * 0.05, 0.05)
    loc = "upper left"
    asb = AnchoredSizeBar(
        ax.transData,
        size=scale_x,
        size_vertical=bar_size / 1,
        label=f"{str(int(bar_size * 100))}{ylabel}",
        loc=loc,
        pad=0.1,
        borderpad=-2,
        frameon=False,
        color="black",
    )
    ax.add_artist(asb)

    ax.axis("off")

    plt.tight_layout()


def plot_heatmap(
    dff,
    mask,
    title=None,
    cmap=cm.inferno,
    t_max=None,
    norm=False,
    include_baseline=False,
    bl=None,
    light_lw=0.5,
    vmax=0.8,
):
    dff_regs = dff[mask]
    av_dff = np.mean(dff_regs, axis=0) * 100
    av_dff = np.flip(av_dff, axis=1)

    t = np.arange(av_dff.shape[0]) / VOLUME_RATE - PRE_EVENT_T
    x = np.linspace(0, CELL_LENGTH, av_dff.shape[1])

    if t_max is not None:
        t_mask = t < t_max
        t = t[t_mask]
        av_dff = av_dff[t_mask]

    extent = np.min(t), np.max(t), np.min(x), np.max(x)

    if norm:
        dyn_range = np.amax(av_dff, axis=0) - np.amin(av_dff, axis=0)
        av_dff = av_dff / dyn_range

    if include_baseline:
        av_dff = av_dff + np.flip(bl)

    fig, ax = plt.subplots(frameon=False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    im = ax.imshow(
        av_dff.T,
        cmap=cmap,
        extent=extent,
        origin="lower",
        interpolation="none",
        aspect="equal",
        vmin=max(-25, np.amin(av_dff)),
        vmax=vmax * np.amax(av_dff),
    )
    ax.set_ylabel("Proximal <-> Distal")
    ax.set_xlabel("Time [sec]")

    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label(r"$\Delta F / F_0$ [%]")
    ax.set_yticks([])
    plot_light_on_off(ax, lw=light_lw, ymin=0, ymax=1)

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    return fig, ax


def plot_ex_lr(
    t,
    x,
    t_mask,
    num_regs,
    region_colors,
    labels,
    title=None,
    ylabel=None,
):
    _, ax = plt.subplots()

    for reg_num, reg_color, label in zip(range(num_regs), region_colors, labels):
        if (not (reg_num == 0 or reg_num == (num_regs - 1))) and num_regs > 3:
            ax.plot(t[t_mask], x[t_mask, reg_num], color=reg_color)
            ax.plot(
                t[np.logical_not(t_mask)],
                x[np.logical_not(t_mask), reg_num],
                color=reg_color,
                alpha=0.4,
            )
        else:
            ax.plot(t[t_mask], x[t_mask, reg_num], color=reg_color, label=label)
            ax.plot(
                t[np.logical_not(t_mask)],
                x[np.logical_not(t_mask), reg_num],
                color=reg_color,
                alpha=0.4,
            )

    ax.set_xlabel("Time [sec]")
    ax.set_ylabel(r"$\Delta F / F_0$ [%]")
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
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

    ax.legend()


def plot_ex_corr(x, t_mask, num_regs, region_colors, labels, max_lag, title=None):
    print(f"x.shape: {x.shape}")
    plt.figure()
    x_ref = x[t_mask, DISTAL_REG]
    n = x_ref.shape[0]
    lim = 1.96 / np.sqrt(n)
    for reg_num, reg_color, label in zip(range(num_regs), region_colors, labels):
        x_reg = x[t_mask, reg_num]
        corr = signal.correlate(x_reg, x_ref, mode="same") / n

        lags = np.arange(x_reg.shape[0]) / VOLUME_RATE
        lags = lags - np.amax(lags) / 2

        lag_mask = np.absolute(lags) < max_lag
        lags = lags[lag_mask]
        corr = corr[lag_mask]

        max_ind = np.argmax(corr)
        if not (reg_num == 0 or reg_num == (num_regs - 1)) and num_regs > 3:
            plt.plot(lags, corr, color=reg_color, alpha=0.9)
        else:
            plt.plot(lags, corr, color=reg_color, label=label, alpha=0.9)

        plt.scatter(
            [lags[max_ind]], [corr[max_ind]], color=reg_color, marker="x", alpha=0.9
        )

    plt.hlines(
        [lim, -lim], -max_lag, max_lag, linestyle="dashed", color="black", alpha=0.7
    )
    plt.xlabel("Lag [sec]")
    plt.ylabel("Correlation")
    plt.legend()
    if title is not None:
        plt.title(title)


def plot_event_scatter(
    df_ts,
    s,
    alpha,
    mask,
    t_col,
    reg_colors,
    num_regs,
    title=None,
    band_width=0.5,
    xlim=None,
    xticks=None,
    yticks=None,
):
    masked_df_ts = df_ts[mask]
    masked_s = s[mask]
    y_0s = []

    _, ax = plt.subplots()

    plot_light_on_off(ax)

    for reg_num in range(num_regs):
        df_ts_reg = masked_df_ts[masked_df_ts["region"] == reg_num]
        x = df_ts_reg[t_col]
        y_0 = num_regs - reg_num + 0.5
        y = y_0 + np.linspace(-band_width / 2, band_width / 2, x.size)
        s = masked_s[masked_df_ts["region"] == reg_num]

        ax.scatter(x, y, s=s, color=reg_colors[reg_num], alpha=alpha)

        y_0s.append(y_0)

    y_labels = ["Distal"]
    for _ in range(1, num_regs - 1):
        y_labels.append("")
    y_labels.append("Proximal")

    ax.set_yticks(y_0s, y_labels)
    ax.set_xlabel("Time [sec]")

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])

    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    if title is not None:
        ax.set_title(title)


def plot_event_scatter_max(
    df,
    mask,
    reg_colors,
    num_regs,
    title=None,
    band_width=0.5,
    xlim=None,
    xticks=None,
    yticks=None,
):
    masked_df = df[mask]
    y_0s = []

    _, ax = plt.subplots()

    plot_light_on_off(ax)

    s = get_dot_size(None, None, None)

    for reg_num in range(num_regs):

        x = masked_df[f"t_onset_r{reg_num}"]
        y_0 = num_regs - reg_num + 0.5
        y = y_0 + np.linspace(-band_width / 2, band_width / 2, x.size)

        ax.scatter(x, y, s=s, color=reg_colors[reg_num], alpha=MARKER_ALPHA)

        y_0s.append(y_0)

    y_labels = ["Distal"]
    for _ in range(1, num_regs - 1):
        y_labels.append("")
    y_labels.append("Proximal")

    ax.set_yticks(y_0s, y_labels)
    ax.set_xlabel("Time [sec]")

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])

    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    if title is not None:
        ax.set_title(title)


def amp_regions_generating_code(num_regs, results_dir, fig_dir):
    distal_reg = 0
    proximal_reg = num_regs - 1
    fig_dir = os.path.join(fig_dir, "amplitude")

    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)
    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)

    ctrl_mask, ptz_mask = ctrl_ptz_mask(df_stat)

    group_labels = get_group_labels()

    set_fig_size(0.8, 0.7)
    for group, g_mask in zip(group_labels, [ctrl_mask, ptz_mask]):
        plot_trace(dff, g_mask, plot_std=True, ylim=(-50, 200))
        save_fig(fig_dir, f"trace_av_std_{group}")
        plot_trace(dff, g_mask, plot_sem=True, ylim=(-25, 150))
        save_fig(fig_dir, f"trace_av_sem_{group}")
        plot_trace(dff, g_mask, plot_all=True, ylim=(-100, 300))
        save_fig(fig_dir, f"trace_av_singles_{group}")

    print("\namp")
    x = df_stat[f"amp_r{distal_reg}"].to_numpy()
    y = df_stat[f"amp_r{proximal_reg}"].to_numpy()
    a = df_stat["ptz"].to_numpy()

    mixreg, _, y_mix = linear_mixed_model(x, a, y)
    p = likelihood_ratio_test(x, a, y)
    print(f"likelihood-ratio-test: {p}")

    rs_distal = ranksums(x[a], x[np.logical_not(a)])
    rs_proximal = ranksums(y[a], y[np.logical_not(a)])

    print(f"rank-sum distal amp: {rs_distal.pvalue}")
    print(f"rank-sum proximal amp: {rs_proximal.pvalue}")

    set_fig_size(0.8, 1)
    plot_scatter_hist_categories(
        df_stat,
        f"amp_r{distal_reg}",
        f"amp_r{proximal_reg}",
        CTRL_PTZ_COLORS,
        (ctrl_mask, ptz_mask),
        ("Control", "PTZ"),
        labels=(
            r"Distal amplitude $\Delta F / F_0$ [%]",
            r"Proximal amplitude $\Delta F / F_0$ [%]",
        ),
        binsize=15,
        percent=True,
        mixreg=mixreg,
    )
    save_fig(fig_dir, "amp_scatter_hist_distal_proximal_ptz_ctrl")

    print("\namp5s")
    x = df_stat[f"amp5s_r{distal_reg}"].to_numpy()
    y = df_stat[f"amp5s_r{proximal_reg}"].to_numpy()
    a = df_stat["ptz"].to_numpy()

    mixreg, _, y_mix = linear_mixed_model(x, a, y)
    p = likelihood_ratio_test(x, a, y)
    print(f"likelihood-ratio-test: {p}")

    rs_distal = ranksums(x[a], x[np.logical_not(a)])
    rs_proximal = ranksums(y[a], y[np.logical_not(a)])

    print(f"rank-sum distal amp: {rs_distal.pvalue}")
    print(f"rank-sum proximal amp: {rs_proximal.pvalue}")

    set_fig_size(0.8, 1)
    plot_scatter_hist_categories(
        df_stat,
        f"amp5s_r{distal_reg}",
        f"amp5s_r{proximal_reg}",
        CTRL_PTZ_COLORS,
        (ctrl_mask, ptz_mask),
        ("Control", "PTZ"),
        labels=(
            r"Distal amplitude $\Delta F / F_0$ [%]",
            r"Proximal amplitude $\Delta F / F_0$ [%]",
        ),
        binsize=15,
        percent=True,
        mixreg=mixreg,
    )
    save_fig(fig_dir, "amp5s_scatter_hist_distal_proximal_ptz_ctrl")

    set_fig_size(0.8, 1)
    plot_slope_plot(
        df_stat,
        f"amp_r{distal_reg}",
        f"amp_r{proximal_reg}",
        CTRL_PTZ_COLORS,
        (ctrl_mask, ptz_mask),
        ("Control", "PTZ"),
    )
    save_fig(fig_dir, "amp_slope_distal_proximal_ptz_ctrl")


def peak_regions_generating_code(num_regs, results_dir, fig_dir):
    distal_reg = 0
    proximal_reg = num_regs - 1
    fig_dir = os.path.join(fig_dir, "peak")
    distal_reg_threshold = 20

    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)

    ctrl_mask, ptz_mask = ctrl_ptz_mask(df_stat)

    peak_reg_ptz = [
        df_stat[ptz_mask][f"peak_r{reg_num}"] * 100 for reg_num in range(num_regs)
    ]
    peak_reg_ctrl = [
        df_stat[ctrl_mask][f"peak_r{reg_num}"] * 100 for reg_num in range(num_regs)
    ]

    labels = get_region_labels(num_regs)

    set_fig_size(0.9, 0.7)
    plot_split_violin(
        (peak_reg_ctrl, peak_reg_ptz),
        ("Control", "PTZ"),
        CTRL_PTZ_COLORS,
        r"Peak $\Delta F / F_0$ [%]",
        labels,
    )
    save_fig(fig_dir, "peak_reg_violin")

    df_peak_ptz = pd.concat(peak_reg_ptz, axis=1)
    df_peak_ctrl = pd.concat(peak_reg_ctrl, axis=1)

    print(df_peak_ptz.head())

    df_peak_ptz = df_peak_ptz[df_peak_ptz[f"peak_r{distal_reg}"] > distal_reg_threshold]
    df_peak_ctrl = df_peak_ctrl[
        df_peak_ctrl[f"peak_r{distal_reg}"] > distal_reg_threshold
    ]

    for df in [df_peak_ctrl, df_peak_ptz]:
        for reg_num in range(num_regs):
            reg_peak, ref_peak = df[f"peak_r{reg_num}"], df[f"peak_r{distal_reg}"]
            df[f"rel_diff_peak_r{reg_num}"] = (reg_peak - ref_peak) / ref_peak * 100

    rel_diff_peak_ctrl = [
        df_peak_ctrl[f"rel_diff_peak_r{reg_num}"] for reg_num in range(1, num_regs)
    ]
    rel_diff_peak_ptz = [
        df_peak_ptz[f"rel_diff_peak_r{reg_num}"] for reg_num in range(1, num_regs)
    ]

    plot_split_violin(
        (rel_diff_peak_ctrl, rel_diff_peak_ptz),
        ("Control", "PTZ"),
        CTRL_PTZ_COLORS,
        "Relative difference peaklitude [%]",
        labels[1:],
    )
    save_fig(fig_dir, "rel_peak_reg_violin")

    for reg_num in range(1, num_regs):
        df_stat[f"diff_peak_r{reg_num}"] = (
            df_stat[f"peak_r{reg_num}"] - df_stat[f"peak_r{distal_reg}"]
        )
        df_stat[f"diff_peak_r{reg_num}"] = (
            df_stat[f"peak_r{reg_num}"] - df_stat[f"peak_r{distal_reg}"]
        )

    x = df_stat[f"peak_r{distal_reg}"].to_numpy()
    y = df_stat[f"peak_r{proximal_reg}"].to_numpy()
    a = df_stat["ptz"].to_numpy()

    mixreg, _, y_pred = linear_mixed_model(x, a, y)
    p = likelihood_ratio_test(x, a, y)
    rsq = rsquared(y, y_pred)

    set_fig_size(0.7, 1)
    plot_scatter_categories(
        df_stat,
        f"peak_r{distal_reg}",
        f"peak_r{proximal_reg}",
        CTRL_PTZ_COLORS,
        (ctrl_mask, ptz_mask),
        ("Control", "PTZ"),
        labels=(
            r"Distal peak $\Delta F / F_0$ [%]",
            r"Proximal peak $\Delta F / F_0$ [%]",
        ),
        percent=True,
        title=f"p-value = {p:.03g}, R^2 = {rsq:.03g}",
        mixreg=mixreg,
        # axlim=(-180, 550),
    )
    save_fig(fig_dir, "peak_scatter_distal_proximal_ptz_ctrl")

    masks, labels, labels_pretty = get_peak_masks(df_stat, num_regs)

    df_stat["peak_cat"] = ""
    for mask, label in zip(masks, labels_pretty):
        df_stat.loc[mask, "peak_cat"] = label

    print(df_stat["peak_cat"])
    print(df_stat.head())
    plot_scatter_categories(
        df_stat,
        f"peak_r{distal_reg}",
        f"peak_r{proximal_reg}",
        PEAK_CAT_COLORS,
        masks,
        labels_pretty,
        labels=(
            r"Distal peak $\Delta F / F_0$ [%]",
            r"Proximal peak $\Delta F / F_0$ [%]",
        ),
        percent=True,
        # axlim=(-180, 550),
    )
    save_fig(fig_dir, "peak_scatter_distal_proximal_cat")

    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)

    set_fig_size(0.75, 0.6)
    for mask, label in zip(masks, labels):
        plot_trace(dff, mask, ylim=(-60, 160), t_max=60)
        save_fig(fig_dir, f"trace_{label}")

    set_fig_size(0.7, 1)
    plot_split_bar(
        df_stat["exp_name"],
        masks,
        labels_pretty,
        ctrl_mask,
        CTRL_PTZ_COLORS,
        test_sig=True,
    )
    save_fig(fig_dir, "cell_category_bar")


def plot_schematic_time_lag_distal(
    num_regs,
    results_dir,
    fig_dir,
):

    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)
    num_evts = dff.shape[0]
    num_frames = dff.shape[1]
    fig_dir = os.path.join(fig_dir, "time_lag")

    num_regs_t = 6
    num_regs_amp = 3
    df_stat_t = load_pickle(results_dir, STATS_FNAME, num_regs_t)
    df_stat_amp = load_pickle(results_dir, STATS_FNAME, num_regs_amp)

    masks, _, _ = get_amp_t_onset_masks(
        df_stat_t, df_stat_amp, num_regs_t, num_regs_amp
    )

    mask = masks[-1]

    rng = np.random.default_rng()
    random_evts = rng.choice(np.arange(num_evts), num_evts, replace=False)
    for evt_num in random_evts:
        if not mask[evt_num]:
            continue

        dff_evt = dff[evt_num]
        region_colors = get_region_colors(num_regs)

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

        p = 10
        phis = arp_model_fit(dff_evt[:, DISTAL_REG], p)
        res_evt = np.zeros(dff_evt.shape)
        for reg_num in range(num_regs):
            res_evt[:, reg_num] = arp_model_res(dff_evt[:, reg_num], phis)

        mu_dff, std_dff = np.mean(dff_evt[t_mask], axis=0), np.std(
            dff_evt[t_mask], axis=0
        )
        mu_res, std_res = np.mean(res_evt[t_mask], axis=0), np.std(
            res_evt[t_mask], axis=0
        )
        dff_norm = (dff_evt - mu_dff) / std_dff
        res_norm = (res_evt - mu_res) / std_res

        set_fig_size(0.48, 0.8)
        plot_ex_lr(
            t,
            dff_norm,
            t_mask,
            num_regs,
            region_colors,
            labels,
            ylabel="Relative fluorescence [arbitrary unit]",
        )
        save_fig(fig_dir, "ex_light_response_norm")
        plot_ex_lr(
            t,
            res_norm,
            t_mask,
            num_regs,
            region_colors,
            labels,
            ylabel="Relative fluorescence [arbitrary unit]",
        )
        save_fig(fig_dir, "ex_light_response_res")

        max_lag = 10
        plot_ex_corr(dff_norm, t_mask, num_regs, region_colors, labels, max_lag)
        save_fig(fig_dir, "ex_corr_func_norm")
        plot_ex_corr(
            res_norm,
            t_mask,
            num_regs,
            region_colors,
            labels,
            max_lag,
        )
        save_fig(fig_dir, "ex_corr_func_res")
        plt.show()
        cont = input("continue? (y/n)")
        if cont == "n":
            break


def t_lag_regions_generating_code(num_regs, results_dir, fig_dir):
    fig_dir = os.path.join(fig_dir, "time_lag")

    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)
    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)

    ctrl_mask, ptz_mask = ctrl_ptz_mask(df_stat)

    reg_labels = get_region_labels(num_regs)[1:]

    num_trials = len(df_stat.index)

    for lag_type in ["dff", "res"]:
        num_sig = np.zeros(num_trials, dtype=int)
        for reg_num in range(num_regs):
            num_sig = (
                num_sig
                + (
                    df_stat[f"corr_{lag_type}_r{reg_num}"]
                    > df_stat[f"limit_{lag_type}_r{reg_num}"]
                ).to_numpy()
            )

        print(num_sig)
        significant = np.ones(num_trials, dtype=bool)

        lag_reg_ptz = [
            df_stat[ptz_mask & significant][f"t_lag_{lag_type}_r{reg_num}"]
            for reg_num in range(1, num_regs)
        ]
        lag_reg_ctrl = [
            df_stat[ctrl_mask & significant][f"t_lag_{lag_type}_r{reg_num}"]
            for reg_num in range(1, num_regs)
        ]

        corr_reg_ptz = [
            df_stat[ptz_mask & significant][f"corr_{lag_type}_r{reg_num}"]
            for reg_num in range(1, num_regs)
        ]
        corr_reg_ctrl = [
            df_stat[ctrl_mask & significant][f"corr_{lag_type}_r{reg_num}"]
            for reg_num in range(1, num_regs)
        ]

        set_fig_size(0.48, 0.7)
        plot_split_violin(
            (lag_reg_ctrl, lag_reg_ptz),
            ("Control", "PTZ"),
            CTRL_PTZ_COLORS,
            "Lag [s]",
            reg_labels,
        )
        save_fig(fig_dir, f"lag_{lag_type}_violin")

        plot_split_violin(
            (corr_reg_ctrl, corr_reg_ptz),
            ("Control", "PTZ"),
            CTRL_PTZ_COLORS,
            "Correlation",
            reg_labels,
        )
        save_fig(fig_dir, f"corr_{lag_type}_violin")

        masks, labels, _ = get_t_lag_masks(df_stat, num_regs, lag_type=lag_type)

        set_fig_size(0.45, 0.7)
        for mask, label in zip(masks, labels):
            for group_label, group_mask in zip(
                get_group_labels(), [ctrl_mask, ptz_mask]
            ):
                c_label = group_label + "_" + label
                c_mask = mask & group_mask & significant

                plot_trace(dff, c_mask, ylim=(-50, 150), t_max=60)
                save_fig(fig_dir, f"trace_{lag_type}_{c_label}")


def schematic_calcium_event_detection(num_regs, results_dir, fig_dir):
    fig_dir = os.path.join(fig_dir, "time_onset")
    reg_colors = get_region_colors(num_regs)

    df_t_onsets = load_pickle(results_dir, T_ONSET_FNAME, num_regs)
    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)
    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)
    d2xdt2 = load_np(results_dir, SECOND_DERIVATIVE_FNAME, num_regs)
    pos_regs = load_np(results_dir, POS_REGS_FNAME, num_regs)

    s = get_dot_size(df_t_onsets, "diff_dxdt", [1.5, 40])

    df_shuffled = df_stat.sample(frac=1)
    for ind, row in df_shuffled.iterrows():
        exp_name = row["exp_name"]
        roi_number = row["roi_number"]
        evt_num = row["evt_num"]
        mask = (
            (df_stat["exp_name"] == exp_name)
            & (df_stat["roi_number"] == roi_number)
            & (df_stat["evt_num"] == evt_num)
        )
        onset_mask = stat_mask_to_t_onset_mask(df_stat, df_t_onsets, mask)
        if not np.any(onset_mask):
            continue

        pos_reg = pos_regs[ind]
        for pixel_num in range(pos_reg.shape[-1]):
            if not np.any(pos_reg[:, :, pixel_num]):
                pos_reg = pos_reg[:, :, :pixel_num]
                break

        set_fig_size(0.48, 0.7)
        plot_pos_reg(pos_reg, num_regs)
        save_fig(fig_dir, f"ex_roi_detection", im_format="svg")

        set_fig_size(0.48, 0.7)
        plot_trace_stacked(
            dff,
            mask,
            t_max=15,
        )
        save_fig(fig_dir, "ex_dff_detection", im_format="svg")

        plot_trace_stacked(
            d2xdt2, mask, t_max=15, shade_peaks=True, second_derivative=True
        )
        save_fig(fig_dir, "ex_d2xdt2_detection", im_format="svg")

        set_fig_size(0.49, 0.7)
        plot_event_scatter(
            df_t_onsets,
            s,
            0.9,
            onset_mask,
            "time_com",
            reg_colors,
            num_regs,
            band_width=1e-6,
            xlim=(-5, 15),
            xticks=[-5, 0, 5, 10, 15],
        )
        save_fig(fig_dir, "ex_event_scatter_detection", im_format="svg")

        plt.show()


def schematic_calcium_wave_detection(num_regs, results_dir, fig_dir):
    fig_dir = os.path.join(fig_dir, "time_onset")
    reg_colors = get_region_colors(num_regs)

    df_t_onsets = load_pickle(results_dir, T_ONSET_FNAME, num_regs)
    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)

    s = get_dot_size(df_t_onsets, "diff_dxdt", [1.5, 40])

    masks, _, _ = get_t_onset_masks(df_stat, num_regs)

    sim_mask = masks[0]
    wave_mask = masks[1]
    wave_r_mask = masks[2]
    undef_mask = masks[3]

    cli_input = ""

    df_shuffled = df_stat.sample(frac=1)
    for p_mask, pattern in zip(
        [sim_mask, wave_mask, wave_r_mask, undef_mask],
        ["sim", "wave", "wave_r", "undef"],
    ):
        print(pattern)
        for ind, row in df_shuffled.iterrows():
            if not p_mask[ind]:
                continue

            exp_name = row["exp_name"]
            roi_number = row["roi_number"]
            evt_num = row["evt_num"]
            mask = (
                (df_stat["exp_name"] == exp_name)
                & (df_stat["roi_number"] == roi_number)
                & (df_stat["evt_num"] == evt_num)
            )
            onset_mask = stat_mask_to_t_onset_mask(df_stat, df_t_onsets, mask)
            if not np.any(onset_mask):
                continue

            set_fig_size(0.3, 0.7)
            plot_event_scatter(
                df_t_onsets,
                s,
                0.9,
                onset_mask,
                "time_com",
                reg_colors,
                num_regs,
                band_width=1e-6,
                xlim=(-2.5, 15),
                xticks=[0, 5, 10, 15],
            )
            save_fig(fig_dir, f"ex_event_scatter_{pattern}")

            plot_event_scatter_max(
                df_stat,
                mask,
                reg_colors,
                num_regs,
                band_width=1e-6,
                xlim=(-2.5, 15),
                xticks=[0, 5, 10, 15],
            )
            save_fig(fig_dir, f"ex_event_scatter_max_{pattern}")

            plt.show()

            cli_input = input("Quit? (y/*)")
            if cli_input == "y":
                break


def t_onset_regions_generating_code(num_regs, results_dir, fig_dir):
    fig_dir = os.path.join(fig_dir, "time_onset")

    df_t_onsets = load_pickle(results_dir, T_ONSET_FNAME, num_regs)
    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)
    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)

    ctrl_mask, ptz_mask = ctrl_ptz_mask(df_stat)

    print(f"df_t_onsets.shape: {df_t_onsets.shape}")
    print(f"df_stat.shape: {df_stat.shape}")
    print(f"dff.shape: {dff.shape}")

    t_col = "time_com"
    size_col = "diff_dxdt"

    reg_colors = get_region_colors(num_regs)
    s_all = np.array(get_dot_size(df_t_onsets, size_col, [0.1, 30], logarithmic=False))

    masks, labels, labels_pretty = get_t_onset_masks(df_stat, num_regs)
    onset_masks = [
        stat_mask_to_t_onset_mask(df_stat, df_t_onsets, mask) for mask in masks
    ]

    df_stat["t_onset_cat"] = ""
    for mask, label in zip(masks, labels_pretty):
        df_stat.loc[mask, "t_onset_cat"] = label

    for ptz, group_mask_stat in zip([False, True], [ctrl_mask, ptz_mask]):
        set_fig_size(0.48, 0.7)
        group = "PTZ" if ptz else "Control"
        group_mask = stat_mask_to_t_onset_mask(df_stat, df_t_onsets, group_mask_stat)

        plot_event_scatter(
            df_t_onsets,
            s_all,
            MARKER_ALPHA,
            group_mask,
            t_col,
            reg_colors,
            num_regs,
        )
        save_fig(fig_dir, f"event_scatter_{group}")

        set_fig_size(0.48, 0.7)
        plot_trace(
            dff,
            group_mask_stat,
            xlim=(-4, 15.75),
            ylim=(-10, 140),
            xticks=[0, 5, 10, 15],
        )
        save_fig(fig_dir, f"av_trace_{group}")

        set_fig_size(0.48, 0.7)
        for mask, label in zip(onset_masks, labels):
            group_label = group + "_" + label
            c_mask = mask & group_mask

            plot_event_scatter(
                df_t_onsets, s_all, MARKER_ALPHA, c_mask, t_col, reg_colors, num_regs
            )
            save_fig(fig_dir, f"event_scatter_{group_label}")

        set_fig_size(0.48, 0.7)
        for mask, label in zip(masks, labels):
            group_label = group + "_" + label
            c_mask = mask & group_mask_stat

            sample_size = get_sample_size(df_stat, c_mask)
            print(f"\n{group_label}:\n{sample_size}")

            plot_trace(dff, c_mask, ylim=(-20, 250), t_max=60)
            save_fig(fig_dir, f"trace_{group_label}")

    """ set_fig_size(0.7, 0.8)
    plot_split_bar(
        # df_stat["exp_name"] + df_stat["roi_number"].astype("string"),
        df_stat["exp_name"],
        masks,
        labels_pretty,
        ctrl_mask,
        CTRL_PTZ_COLORS,
        test_sig=False,
    )
    save_fig(fig_dir, "category_bar")

    plot_scatter_categories(
        df_stat,
        "t_onset_slope",
        "t_onset_rsq",
        T_ONSET_CAT_COLORS,
        masks,
        labels_pretty,
    )
    save_fig(fig_dir, "t_slope_vs_rsquared_wave")
    for g_label, g_mask_stat in zip(["Control", "PTZ"], [ctrl_mask, ptz_mask]):
        plot_scatter_categories(
            df_stat,
            "t_onset_slope",
            "t_onset_rsq",
            T_ONSET_CAT_COLORS,
            masks,
            labels_pretty,
            mask_mask=g_mask_stat,
        )
        save_fig(fig_dir, f"t_slope_vs_rsquared_{g_label}")

    set_fig_size(0.7, 0.8)
    print_plot_prop_speed(df_stat, masks[1], labels_pretty[1])
    save_fig(fig_dir, "calcium_wave_speed_wave")

    print("\nReverse:")
    print_plot_prop_speed(df_stat, masks[2], labels_pretty[2])
    save_fig(fig_dir, "outward_calcium_wave_speed")

    wave_evt_num_masks = []
    for evt_num in range(5):
        wave_evt_num_masks.append(masks[1] & (df_stat["evt_num"] == evt_num))

    set_fig_size(0.7, 0.8)
    reg_analysis(
        df_stat["exp_name"],
        wave_evt_num_masks,
        [f"Trial {i+1}" for i in range(5)],
        ctrl_mask,
        ["Control", "PTZ"],
        CTRL_PTZ_COLORS,
        scale=5,
    )
    save_fig(fig_dir, "wave_freq_trial") """


def amp_t_onset_generating_code(num_regs, results_dir, fig_dir):
    distal_reg = 0
    proximal_reg = num_regs - 1
    fig_dir = os.path.join(fig_dir, "amp_time")

    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)

    masks, labels, labels_pretty = get_t_onset_masks(df_stat, num_regs)
    df_stat["t_onset_cat"] = ""
    for mask, label in zip(masks, labels):
        df_stat.loc[mask, "t_onset_cat"] = label

    ctrl_mask, ptz_mask = ctrl_ptz_mask(df_stat)

    print("\namp\n")
    for ptz, g_mask in zip([False, True], [ctrl_mask, ptz_mask]):
        group = "ptz" if ptz else "Control"
        print(f"\n{group}")

        light_seq_mask = np.logical_and(np.logical_or(masks[0], masks[1]), g_mask)
        df_stat_light_seq = df_stat[light_seq_mask]
        x = df_stat_light_seq[f"amp_r{distal_reg}"].to_numpy()
        y = df_stat_light_seq[f"amp_r{proximal_reg}"].to_numpy()
        a = (df_stat_light_seq["t_onset_cat"] == "calcium_wave").to_numpy()

        mixreg, _, _ = linear_mixed_model(x, a, y)
        p = likelihood_ratio_test(x, a, y)
        print(f"likelihood-ratio-test: {p}")

        rs_distal = ranksums(x[a], x[np.logical_not(a)])
        rs_proximal = ranksums(y[a], y[np.logical_not(a)])

        print(f"rank-sum distal amp: {rs_distal.pvalue}")
        print(f"rank-sum proximal amp: {rs_proximal.pvalue}")

        set_fig_size(0.48, 1)
        plot_scatter_hist_categories(
            df_stat,
            f"amp_r{distal_reg}",
            f"amp_r{proximal_reg}",
            (T_ONSET_CAT_COLORS[0], T_ONSET_CAT_COLORS[1]),
            (masks[0] & g_mask, masks[1] & g_mask),
            (labels_pretty[0], labels_pretty[1]),
            labels=(
                r"Distal amplitude $\Delta F / F_0$ [%]",
                r"Proximal amplitude $\Delta F / F_0$ [%]",
            ),
            binsize=15,
            percent=True,
            mixreg=mixreg,
            xlim=(-130, 290),
            ylim=(-110, 520),
        )
        save_fig(fig_dir, f"amp_scatter_hist_distal_proximal_sync_wave_{group}")

        set_fig_size(0.8, 1)
        plot_slope_plot(
            df_stat,
            f"amp_r{distal_reg}",
            f"amp_r{proximal_reg}",
            (T_ONSET_CAT_COLORS[0], T_ONSET_CAT_COLORS[1]),
            (masks[0] & g_mask, masks[1] & g_mask),
            (labels_pretty[0], labels_pretty[1]),
        )
        save_fig(fig_dir, f"amp_slope_distal_proximal_{group}")

    sfig_dir = os.path.join(fig_dir, "5_15s")
    print("\namp5s\n")
    for ptz, g_mask in zip([False, True], [ctrl_mask, ptz_mask]):
        group = "ptz" if ptz else "Control"
        print(f"\n{group}")

        light_seq_mask = np.logical_and(np.logical_or(masks[0], masks[1]), g_mask)
        df_stat_light_seq = df_stat[light_seq_mask]
        x = df_stat_light_seq[f"amp5s_r{distal_reg}"].to_numpy()
        y = df_stat_light_seq[f"amp5s_r{proximal_reg}"].to_numpy()
        a = (df_stat_light_seq["t_onset_cat"] == "calcium_wave").to_numpy()

        mixreg, _, _ = linear_mixed_model(x, a, y)
        p = likelihood_ratio_test(x, a, y)
        print(f"likelihood-ratio-test: {p}")

        rs_distal = ranksums(x[a], x[np.logical_not(a)])
        rs_proximal = ranksums(y[a], y[np.logical_not(a)])

        print(f"rank-sum distal amp: {rs_distal.pvalue}")
        print(f"rank-sum proximal amp: {rs_proximal.pvalue}")

        plot_scatter_hist_categories(
            df_stat,
            f"amp5s_r{distal_reg}",
            f"amp5s_r{proximal_reg}",
            (T_ONSET_CAT_COLORS[0], T_ONSET_CAT_COLORS[1]),
            (masks[0] & g_mask, masks[1] & g_mask),
            (labels_pretty[0], labels_pretty[1]),
            labels=(
                r"Distal amplitude $\Delta F / F_0$ [%]",
                r"Proximal amplitude $\Delta F / F_0$ [%]",
            ),
            binsize=15,
            percent=True,
            mixreg=mixreg,
            xlim=(-50, 195),
            ylim=(-90, 195),
        )
        save_fig(sfig_dir, f"amp5s_scatter_hist_distal_proximal_sync_wave_{group}")


def peak_t_onset_generating_code(num_regs, results_dir, fig_dir):
    distal_reg = 0
    proximal_reg = num_regs - 1
    fig_dir = os.path.join(fig_dir, "peak_time")

    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)

    masks, labels, labels_pretty = get_t_onset_masks(df_stat, num_regs)
    df_stat["t_onset_cat"] = ""
    for mask, label in zip(masks, labels):
        df_stat.loc[mask, "t_onset_cat"] = label

    ctrl_mask, ptz_mask = ctrl_ptz_mask(df_stat)

    print("\npeak\n")
    for ptz, g_mask in zip([False, True], [ctrl_mask, ptz_mask]):
        group = "ptz" if ptz else "Control"
        print(f"\n{group}")

        light_seq_mask = np.logical_and(np.logical_or(masks[0], masks[1]), g_mask)
        df_stat_light_seq = df_stat[light_seq_mask]
        x = df_stat_light_seq[f"peak_r{distal_reg}"].to_numpy()
        y = df_stat_light_seq[f"peak_r{proximal_reg}"].to_numpy()
        a = (df_stat_light_seq["t_onset_cat"] == "calcium_wave").to_numpy()

        mixreg, _, _ = linear_mixed_model(x, a, y)
        p = likelihood_ratio_test(x, a, y)
        print(f"likelihood-ratio-test: {p}")

        rs_distal = ranksums(x[a], x[np.logical_not(a)])
        rs_proximal = ranksums(y[a], y[np.logical_not(a)])

        print(f"rank-sum distal peak: {rs_distal.pvalue}")
        print(f"rank-sum proximal peak: {rs_proximal.pvalue}")

        set_fig_size(0.48, 1)
        plot_scatter_hist_categories(
            df_stat,
            f"peak_r{distal_reg}",
            f"peak_r{proximal_reg}",
            (T_ONSET_CAT_COLORS[0], T_ONSET_CAT_COLORS[1]),
            (masks[0] & g_mask, masks[1] & g_mask),
            (labels_pretty[0], labels_pretty[1]),
            labels=(
                r"Distal peaklitude $\Delta F / F_0$ [%]",
                r"Proximal peaklitude $\Delta F / F_0$ [%]",
            ),
            binsize=15,
            percent=True,
            mixreg=mixreg,
            xlim=(-50, 550),
            ylim=(-50, 690),
        )
        save_fig(fig_dir, f"peak_scatter_hist_distal_proximal_sync_wave_{group}")

    print("\npeak5s\n")
    for ptz, g_mask in zip([False, True], [ctrl_mask, ptz_mask]):
        group = "ptz" if ptz else "Control"
        print(f"\n{group}")

        light_seq_mask = np.logical_and(np.logical_or(masks[0], masks[1]), g_mask)
        df_stat_light_seq = df_stat[light_seq_mask]
        x = df_stat_light_seq[f"peak5s_r{distal_reg}"].to_numpy()
        y = df_stat_light_seq[f"peak5s_r{proximal_reg}"].to_numpy()
        a = (df_stat_light_seq["t_onset_cat"] == "calcium_wave").to_numpy()

        mixreg, _, _ = linear_mixed_model(x, a, y)
        p = likelihood_ratio_test(x, a, y)
        print(f"likelihood-ratio-test: {p}")

        rs_distal = ranksums(x[a], x[np.logical_not(a)])
        rs_proximal = ranksums(y[a], y[np.logical_not(a)])

        print(f"rank-sum distal peak: {rs_distal.pvalue}")
        print(f"rank-sum proximal peak: {rs_proximal.pvalue}")

        plot_scatter_hist_categories(
            df_stat,
            f"peak5s_r{distal_reg}",
            f"peak5s_r{proximal_reg}",
            (T_ONSET_CAT_COLORS[0], T_ONSET_CAT_COLORS[1]),
            (masks[0] & g_mask, masks[1] & g_mask),
            (labels_pretty[0], labels_pretty[1]),
            labels=(
                r"Distal peaklitude $\Delta F / F_0$ [%]",
                r"Proximal peaklitude $\Delta F / F_0$ [%]",
            ),
            binsize=15,
            percent=True,
            mixreg=mixreg,
            xlim=(-50, 450),
            ylim=(-60, 450),
        )
        save_fig(fig_dir, f"peak5s_scatter_hist_distal_proximal_sync_wave_{group}")


def bl_t_onset_generating_code(num_regs, results_dir, fig_dir):
    distal_reg = 0
    proximal_reg = num_regs - 1
    fig_dir = os.path.join(fig_dir, "bl_time")

    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)

    masks, labels, labels_pretty = get_t_onset_masks(df_stat, num_regs)
    df_stat["t_onset_cat"] = ""
    for mask, label in zip(masks, labels):
        df_stat.loc[mask, "t_onset_cat"] = label

    ctrl_mask, ptz_mask = ctrl_ptz_mask(df_stat)

    set_fig_size(0.48, 1)
    print("\nbl\n")
    for ptz, g_mask in zip([False, True], [ctrl_mask, ptz_mask]):
        group = "ptz" if ptz else "Control"
        print(f"\n{group}")

        light_seq_mask = np.logical_and(np.logical_or(masks[0], masks[1]), g_mask)
        df_stat_light_seq = df_stat[light_seq_mask]
        x = df_stat_light_seq[f"bl_r{distal_reg}"].to_numpy()
        y = df_stat_light_seq[f"bl_r{proximal_reg}"].to_numpy()
        a = (df_stat_light_seq["t_onset_cat"] == "calcium_wave").to_numpy()

        mixreg, _, _ = linear_mixed_model(x, a, y)
        p = likelihood_ratio_test(x, a, y)
        print(f"likelihood-ratio-test: {p}")

        rs_distal = ranksums(x[a], x[np.logical_not(a)])
        rs_proximal = ranksums(y[a], y[np.logical_not(a)])

        print(f"rank-sum distal bl: {rs_distal.pvalue}")
        print(f"rank-sum proximal bl: {rs_proximal.pvalue}")

        plot_scatter_hist_categories(
            df_stat,
            f"bl_r{distal_reg}",
            f"bl_r{proximal_reg}",
            (T_ONSET_CAT_COLORS[0], T_ONSET_CAT_COLORS[1]),
            (masks[0] & g_mask, masks[1] & g_mask),
            (labels_pretty[0], labels_pretty[1]),
            labels=(
                r"Distal baseline $\Delta F / F_0$ [%]",
                r"Proximal baseline $\Delta F / F_0$ [%]",
            ),
            binsize=15,
            percent=True,
            mixreg=mixreg,
            xlim=(-80, 320),
            ylim=(-80, 590),
        )
        save_fig(fig_dir, f"bl_scatter_hist_distal_proximal_sync_wave_{group}")


def clustering_generating_code(num_regs, results_dir, fig_dir):
    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)
    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)

    t = np.arange(dff.shape[1]) / VOLUME_RATE - PRE_EVENT_T
    t_mask = t < 20

    dff_2d = np.reshape(dff[:, t_mask], (dff.shape[0], -1))

    fig_dir = os.path.join(fig_dir, "clustering")

    group_labels = get_group_labels()
    group_masks = [df_stat["ptz"] == b for b in [False, True]]

    k = 4
    set_fig_size(0.48, 0.7)
    for g_mask, label in zip(group_masks, group_labels):
        kmeans = KMeans(n_clusters=k).fit(dff_2d[g_mask])
        c_identity = kmeans.predict(dff_2d)

        for c_num in range(k):
            c_mask = c_identity == c_num
            mask = c_mask & g_mask

            sample_size = get_sample_size(df_stat, mask)
            print(f"\n{label} cluster {c_num+1}:\n{sample_size}")

            plot_trace(dff, mask, plot_all=True, ylim=(-20, 200), t_max=60)
            save_fig(fig_dir, f"trace_lines_{label}_cluster_{c_num}")


def plot_amp5s_amprnd(
    x,
    amp5s=None,
    amprnd=None,
    amp5s_color=None,
    amprnd_color=None,
    amp5s_single=None,
    amprnd_single=None,
):
    _, ax = plt.subplots()

    if amp5s is not None:
        ax.plot(x, amp5s, color=amp5s_color, label=r"$\Delta F / F$ 5 sec")
    if amprnd is not None:
        ax.plot(x, amprnd, color=amprnd_color, label=r"$\Delta F / F$ 110 sec")

    if amp5s_single is not None:
        for amp5s_s in amp5s_single:
            ax.plot(x, amp5s_s, color=amp5s_color, alpha=0.3)

    if amprnd_single is not None:
        for amprnd_s in amprnd_single:
            ax.plot(x, amprnd_s, color=amprnd_color, alpha=0.3)

    ax.legend()

    ax.set_xlabel("Distal <-> Proximal [$\mu m$]")
    ax.set_ylabel(r"$\Delta F / F_0$ [%]")


def calc_lag_corr_lim(x, x_ref, fs):
    x_norm = (x - np.mean(x)) / np.std(x)
    x_ref_norm = (x_ref - np.mean(x_ref)) / np.std(x_ref)
    n = x.shape[0]
    corr = signal.correlate(x_norm, x_ref_norm, mode="same") / n

    lags = np.arange(n) / fs
    lags = lags - np.amax(lags) / 2

    sig_lim = 2.33 / np.sqrt(n)

    return lags, corr, sig_lim


def plot_autocorr_amp5s_amprnd(
    x,
    amp5s,
    amprnd,
    amp5s_color,
    amprnd_color,
    amp5s_single=None,
    amprnd_single=None,
):
    fig, ax = plt.subplots()

    space_sr = 1 / (x[1] - x[0])
    lags, a_amp5s, sig_lim = calc_lag_corr_lim(amp5s, amp5s, space_sr)
    _, a_amprnd, _ = calc_lag_corr_lim(amprnd, amprnd, space_sr)

    pos_mask = lags > 0

    alpha_sig = 0.7
    alpha_single = 0.3

    ax.axhline(sig_lim, linestyle="dashed", color="gray", alpha=alpha_sig)
    ax.axhline(-sig_lim, linestyle="dashed", color="gray", alpha=alpha_sig)

    if amp5s_single is not None:
        for amp5s_s in amp5s_single:
            _, a_amp5s_s, _ = calc_lag_corr_lim(amp5s_s, amp5s_s, space_sr)
            ax.plot(
                lags[pos_mask],
                a_amp5s_s[pos_mask],
                color=amp5s_color,
                alpha=alpha_single,
            )

    if amprnd_single is not None:
        for amprnd_s in amprnd_single:
            _, a_amprnd_s, _ = calc_lag_corr_lim(amprnd_s, amprnd_s, space_sr)
            ax.plot(
                lags[pos_mask],
                a_amprnd_s[pos_mask],
                color=amprnd_color,
                alpha=alpha_single,
            )

    ax.plot(
        lags[pos_mask],
        a_amp5s[pos_mask],
        color=amp5s_color,
        label=r"$\Delta F / F$ 5 sec",
    )
    ax.plot(
        lags[pos_mask],
        a_amprnd[pos_mask],
        color=amprnd_color,
        label=r"$\Delta F / F$ 60 sec",
    )

    ax.set_ylabel("Correlation")
    ax.set_xlabel(f"Lag [micrometer]")
    ax.legend()


def calc_bl_amp5s_amprnd(bl_singles, amp5s_singles, amprnd_singles):
    bl = np.mean(np.array(bl_singles), axis=0)
    amp5s = np.mean(np.array(amp5s_singles), axis=0)
    amprnd = np.mean(np.array(amprnd_singles), axis=0)

    return bl, amp5s, amprnd


def calc_reliability(xs, ys=None):
    x = copy.deepcopy(xs)
    num_trials = len(x)
    rs = []
    if ys is not None:
        y = copy.deepcopy(ys)
        for ind in range(num_trials):
            x1 = x[ind]
            x2 = y[ind]
            r, _ = stats.pearsonr(x1, x2)
            rs.append(r)

    else:
        for ind1 in range(num_trials):
            x1 = x[ind1]
            for ind2 in range(ind1):
                x2 = x[ind2]

                r, _ = stats.pearsonr(x1, x2)
                rs.append(r)

    return np.mean(np.array(rs))


def rep_ex_micro_generating_code(num_regs, results_dir, fig_dir):
    fig_dir = os.path.join(fig_dir, "micro")

    amprnd_color = "gray"
    amp5s_color = "green"

    amp_lw = 0.5

    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)
    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)
    df_stat6 = load_pickle(results_dir, STATS_FNAME, 6)

    pos_regs = load_np(results_dir, POS_REGS_FNAME, num_regs)

    print(f"dff.shape: {dff.shape}")

    x = np.linspace(0, CELL_LENGTH, dff.shape[2])

    print(df_stat.head())
    df_shuffled = df_stat.sample(frac=1)
    print(df_shuffled["exp_name"].size)
    for ind, row in df_shuffled.iterrows():
        exp_name = row["exp_name"]

        roi_number = row["roi_number"]
        event_number = row["evt_num"]
        print(f"\nexp_name: {exp_name}\nroi_number: {roi_number}\n")

        cell_mask = (df_stat["exp_name"] == exp_name) & (
            df_stat["roi_number"] == roi_number
        )
        trial_mask = cell_mask & (df_stat["evt_num"] == event_number)

        bl = np.array([row[f"bl_r{reg_num}"] for reg_num in range(num_regs)]) * 100
        amp5s = (
            np.array([row[f"amp5s_r{reg_num}"] for reg_num in range(num_regs)]) * 100
        )
        amprnd = (
            np.array([row[f"amprnd_r{reg_num}"] for reg_num in range(num_regs)]) * 100
        )

        filt_bl = (
            np.array([row[f"filt_bl_r{reg_num}"] for reg_num in range(num_regs)]) * 100
        )
        filt_amp5s = (
            np.array([row[f"filt_amp5s_r{reg_num}"] for reg_num in range(num_regs)])
            * 100
        )
        filt_amprnd = (
            np.array([row[f"filt_amprnd_r{reg_num}"] for reg_num in range(num_regs)])
            * 100
        )

        pos_reg = pos_regs[ind]
        for pixel_num in range(pos_reg.shape[-1]):
            if not np.any(pos_reg[:, :, pixel_num]):
                pos_reg = pos_reg[:, :, :pixel_num]
                break

        set_fig_size(0.48, 0.7)
        plot_pos_reg(pos_reg, num_regs)
        save_fig(fig_dir, f"ex_roi")

        set_fig_size(0.48, 1)
        _, ax = plot_heatmap(
            dff, trial_mask, t_max=62, light_lw=amp_lw, include_baseline=True, bl=bl
        )
        plot_amp_times(ax, x, amp5s_color, amprnd_color, lw=amp_lw)
        add_scale_bar(
            ax, 0.5, 10, f"10 \u03bcm", loc="lower right", borderpad=1, color="darkgray"
        )
        save_fig(fig_dir, f"ex_heatmap_w_bl")

        _, ax = plot_heatmap(
            dff, trial_mask, t_max=62, light_lw=amp_lw, include_baseline=False, bl=bl
        )
        plot_amp_times(ax, x, amp5s_color, amprnd_color, lw=amp_lw)
        add_scale_bar(
            ax, 0.5, 10, f"10 \u03bcm", loc="lower right", borderpad=1, color="darkgray"
        )
        save_fig(fig_dir, f"ex_heatmap")

        plot_amp5s_amprnd(
            x,
            amp5s=amp5s,
            amprnd=amprnd,
            amp5s_color=amp5s_color,
            amprnd_color=amprnd_color,
        )
        save_fig(fig_dir, f"ex_amps")

        plot_amp5s_amprnd(
            x,
            amp5s=filt_amp5s,
            amprnd=filt_amprnd,
            amp5s_color=amp5s_color,
            amprnd_color=amprnd_color,
        )
        save_fig(fig_dir, f"ex_filt_amps")

        plot_autocorr_amp5s_amprnd(x, amp5s, amprnd, amp5s_color, amprnd_color)
        save_fig(fig_dir, f"ex_autocorr_amps")

        plot_autocorr_amp5s_amprnd(
            x, filt_amp5s, filt_amprnd, amp5s_color, amprnd_color
        )
        save_fig(fig_dir, f"ex_autocorr_filt_amps")

        bl_singles = []
        amp5s_singles = []
        amprnd_singles = []
        filt_bl_singles = []
        filt_amp5s_singles = []
        filt_amprnd_singles = []
        rows = df_stat[cell_mask]
        rows_6 = df_stat6[cell_mask]
        # masks, _, labels_pretty = get_t_onset_masks(rows_6, 6)

        for _, row in rows.iterrows():
            bl_singles.append(
                np.array([row[f"bl_r{reg_num}"] for reg_num in range(num_regs)]) * 100
            )
            amp5s_singles.append(
                np.array([row[f"amp5s_r{reg_num}"] for reg_num in range(num_regs)])
                * 100
            )
            amprnd_singles.append(
                np.array([row[f"amprnd_r{reg_num}"] for reg_num in range(num_regs)])
                * 100
            )

            filt_bl_singles.append(
                np.array([row[f"filt_bl_r{reg_num}"] for reg_num in range(num_regs)])
                * 100
            )
            filt_amp5s_singles.append(
                np.array([row[f"filt_amp5s_r{reg_num}"] for reg_num in range(num_regs)])
                * 100
            )
            filt_amprnd_singles.append(
                np.array(
                    [row[f"filt_amprnd_r{reg_num}"] for reg_num in range(num_regs)]
                )
                * 100
            )

        amp5s, amprnd = calc_bl_amp5s_amprnd(amp5s_singles, amprnd_singles)

        filt_amp5s, filt_amprnd = calc_bl_amp5s_amprnd(
            filt_amp5s_singles, filt_amprnd_singles
        )

        _, ax = plot_heatmap(
            dff, trial_mask, t_max=62, light_lw=amp_lw, include_baseline=True, bl=bl
        )
        plot_amp_times(ax, x, amp5s_color, amprnd_color, lw=amp_lw)
        add_scale_bar(
            ax, 0.5, 10, f"10 \u03bcm", loc="lower right", borderpad=1, color="darkgray"
        )
        save_fig(fig_dir, f"ex_heatmap_multi_w_bl")

        _, ax = plot_heatmap(
            dff, trial_mask, t_max=62, light_lw=amp_lw, include_baseline=False, bl=bl
        )
        plot_amp_times(ax, x, amp5s_color, amprnd_color, lw=amp_lw)
        add_scale_bar(
            ax, 0.5, 10, f"10 \u03bcm", loc="lower right", borderpad=1, color="darkgray"
        )
        save_fig(fig_dir, f"ex_heatmap_multi")

        reliability = calc_reliability(amp5s_singles)
        plot_amp5s_amprnd(
            x,
            amp5s=amp5s,
            amp5s_color=amp5s_color,
            amp5s_single=amp5s_singles,
        )
        print(f"reliability amp5s: {reliability}")
        save_fig(fig_dir, f"ex_amp5s_multi")

        reliability = calc_reliability(filt_amp5s_singles)
        plot_amp5s_amprnd(
            x,
            amp5s=filt_amp5s,
            amp5s_color=amp5s_color,
            amp5s_single=filt_amp5s_singles,
        )
        print(f"reliability filt_amp5s: {reliability}")
        save_fig(fig_dir, f"ex_filt_amp5s_multi")

        reliability = calc_reliability(amprnd_singles)
        plot_amp5s_amprnd(
            x,
            amprnd=amprnd,
            amprnd_color=amprnd_color,
            amprnd_single=amprnd_singles,
        )
        print(f"reliability amprnd: {reliability}")
        save_fig(fig_dir, f"ex_amprnd_multi")

        reliability = calc_reliability(filt_amprnd_singles)
        plot_amp5s_amprnd(
            x,
            amprnd=filt_amprnd,
            amprnd_color=amprnd_color,
            amprnd_single=filt_amprnd_singles,
        )
        print(f"reliability filt_amprnd: {reliability}")
        save_fig(fig_dir, f"ex_filt_amprnd_multi")

        plot_autocorr_amp5s_amprnd(
            x,
            amp5s,
            amprnd,
            amp5s_color,
            amprnd_color,
            amp5s_single=amp5s_singles,
            amprnd_single=amprnd_singles,
        )
        save_fig(fig_dir, f"ex_autocorr_amps_multi")

        plot_autocorr_amp5s_amprnd(
            x,
            filt_amp5s,
            filt_amprnd,
            amp5s_color,
            amprnd_color,
            amp5s_single=filt_amp5s_singles,
            amprnd_single=filt_amprnd_singles,
        )
        save_fig(fig_dir, f"ex_autocorr_filt_amps_multi")

        plt.show()


def plot_reliability_split_bar(
    fig_dir, df_stat, num_regs, var_keys, var_labels, shuffle, mask=None
):
    fig_dir = os.path.join(fig_dir, "micro")

    df = df_stat
    if mask is not None:
        df = df_stat[mask]

    ptz = []
    rel = {key: [] for key in var_keys}
    if shuffle:
        rel["shuffled"] = []

    exp_name_set = df["exp_name"].unique()
    roi_num_set = df["roi_number"].unique()

    for exp_name in exp_name_set:
        for roi_num in roi_num_set:
            cell_mask = (df_stat["exp_name"] == exp_name) & (
                df["roi_number"] == roi_num
            )

            if np.sum(cell_mask) < 2:
                continue

            df_cell = df[cell_mask]
            singles = {key: [] for key in var_keys}
            for _, trial in df_cell.iterrows():
                for key in var_keys:
                    singles[key].append(
                        np.array(
                            [trial[f"{key}_r{reg_num}"] for reg_num in range(num_regs)]
                        )
                        * 100
                    )

            for key in var_keys:
                rel[key].append(calc_reliability(singles[key]))

            if shuffle:
                shuffle_lst = []
                for key in var_keys:
                    for arr in singles[key]:
                        shuffle_lst.append(arr)

                rel["shuffled"].append(calc_reliability(shuffle_lst))

            ptz.append(df_cell["ptz"].unique()[0])

    ptz = np.array(ptz)

    rel_lst = [np.array(lst) for lst in rel.values()]
    if shuffle:
        var_labels.append("shuffled")

    for mask, group_label in zip(
        [df_stat["ptz"] == False, df_stat["ptz"] == True], ["control", "ptz"]
    ):
        sample_size = get_sample_size(df_stat, mask)
        print(f"\n{group_label}:\n{sample_size}")

    plot_split_bar_ptz_inside_out(
        ptz,
        rel_lst,
        var_labels,
        ["", "///"],
        "Inter-Trial Correlation (mean)",
    )


def micro_fig1(num_regs, results_dir, fig_dir):
    fig_dir = os.path.join(fig_dir, "micro", "fig1")
    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)

    amp_lw = 0.5

    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)
    dff_6 = load_np(results_dir, DFF_REGS_FNAME, 6)
    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)
    df_stat_6 = load_pickle(results_dir, STATS_FNAME, 6)

    pos_regs = load_np(results_dir, POS_REGS_FNAME, num_regs)

    print(f"dff.shape: {dff.shape}")

    print(df_stat.head())
    df_shuffled = df_stat.sample(frac=1)
    print(df_shuffled["exp_name"].size)
    for ind, row in df_shuffled.iterrows():
        exp_name = row["exp_name"]

        roi_number = row["roi_number"]
        print(f"\nexp_name: {exp_name}\nroi_number: {roi_number}\n")

        cell_mask = (df_stat["exp_name"] == exp_name) & (
            df_stat["roi_number"] == roi_number
        )
        cell_mask_6 = (df_stat_6["exp_name"] == exp_name) & (
            df_stat_6["roi_number"] == roi_number
        )

        pos_reg = pos_regs[ind]
        for pixel_num in range(pos_reg.shape[-1]):
            if not np.any(pos_reg[:, :, pixel_num]):
                pos_reg = pos_reg[:, :, :pixel_num]
                break

        set_fig_size(0.48, 0.7)
        plot_pos_reg(pos_reg, num_regs)
        save_fig(fig_dir, f"ex_roi")

        rows = df_stat[cell_mask]
        rows_6 = df_stat_6[cell_mask_6].copy()
        masks, labels, _ = get_t_onset_masks(rows_6, 6)

        for ind, row in rows.iterrows():
            trial_num = row["evt_num"]
            trial_mask = cell_mask & (df_stat["evt_num"] == trial_num)

            trial_type = -1
            for i, mask in enumerate(masks):
                if mask[ind]:
                    trial_type = i
                    break

            print(f"Trial {trial_num}: {labels[trial_type]}")

            bl = np.array([row[f"bl_r{reg_num}"] for reg_num in range(num_regs)]) * 100

            set_fig_size(0.48, 1)
            _, ax = plot_heatmap(
                dff, trial_mask, t_max=62, light_lw=amp_lw, include_baseline=True, bl=bl
            )
            add_scale_bar(
                ax,
                0.5,
                10,
                f"10 \u03bcm",
                loc="lower right",
                borderpad=1,
                color="darkgray",
            )
            save_fig(fig_dir, f"ex_heatmap_w_bl_trial_{trial_num}")

            _, ax = plot_heatmap(
                dff,
                trial_mask,
                t_max=62,
                light_lw=amp_lw,
                include_baseline=False,
                bl=bl,
            )
            add_scale_bar(
                ax,
                0.5,
                10,
                f"10 \u03bcm",
                loc="lower right",
                borderpad=1,
                color="darkgray",
            )
            save_fig(fig_dir, f"ex_heatmap_{trial_num}")

            plot_trace(dff_6, trial_mask, t_max=62)
            save_fig(fig_dir, f"ex_trace_{trial_num}")

        plt.show()


def micro_fig2(num_regs, results_dir, fig_dir):
    fig_dir = os.path.join(fig_dir, "micro", "fig2")

    amprnd_color = "gray"
    amp5s_color = "green"

    amp_lw = 0.5

    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)
    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)

    pos_regs = load_np(results_dir, POS_REGS_FNAME, num_regs)

    print(f"dff.shape: {dff.shape}")

    x = np.linspace(0, CELL_LENGTH, dff.shape[2])

    print(df_stat.head())
    df_shuffled = df_stat.sample(frac=1)
    print(df_shuffled["exp_name"].size)
    for ind, row in df_shuffled.iterrows():
        exp_name = row["exp_name"]

        roi_number = row["roi_number"]
        event_number = row["evt_num"]
        print(f"\nexp_name: {exp_name}\nroi_number: {roi_number}\n")

        cell_mask = (df_stat["exp_name"] == exp_name) & (
            df_stat["roi_number"] == roi_number
        )
        trial_mask = cell_mask & (df_stat["evt_num"] == event_number)

        bl = np.array([row[f"bl_r{reg_num}"] for reg_num in range(num_regs)]) * 100
        amp5s = (
            np.array([row[f"amp5s_r{reg_num}"] for reg_num in range(num_regs)]) * 100
        )
        amprnd = (
            np.array([row[f"amprnd_r{reg_num}"] for reg_num in range(num_regs)]) * 100
        )
        filt_amp5s = (
            np.array([row[f"filt_amp5s_r{reg_num}"] for reg_num in range(num_regs)])
            * 100
        )
        filt_amprnd = (
            np.array([row[f"filt_amprnd_r{reg_num}"] for reg_num in range(num_regs)])
            * 100
        )

        pos_reg = pos_regs[ind]
        for pixel_num in range(pos_reg.shape[-1]):
            if not np.any(pos_reg[:, :, pixel_num]):
                pos_reg = pos_reg[:, :, :pixel_num]
                break

        set_fig_size(0.48, 0.7)
        plot_pos_reg(pos_reg, num_regs)
        save_fig(fig_dir, f"ex_roi")

        set_fig_size(0.4, 1)
        _, ax = plot_heatmap(
            dff, trial_mask, t_max=115, light_lw=amp_lw, include_baseline=True, bl=bl
        )
        plot_amp_times(ax, x, amp5s_color, amprnd_color, lw=amp_lw)
        add_scale_bar(
            ax, 0.5, 10, f"10 \u03bcm", loc="lower right", borderpad=1, color="darkgray"
        )
        save_fig(fig_dir, f"ex_heatmap_w_bl")

        _, ax = plot_heatmap(
            dff, trial_mask, t_max=115, light_lw=amp_lw, include_baseline=False, bl=bl
        )
        plot_amp_times(ax, x, amp5s_color, amprnd_color, lw=amp_lw)
        add_scale_bar(
            ax, 0.5, 10, f"10 \u03bcm", loc="lower right", borderpad=1, color="darkgray"
        )
        save_fig(fig_dir, f"ex_heatmap")

        set_fig_size(0.25, 1)

        plot_amp5s_amprnd(
            x,
            amp5s=amp5s,
            amp5s_color=amp5s_color,
        )
        save_fig(fig_dir, f"ex_amp5s")

        plot_amp5s_amprnd(
            x,
            amprnd=amprnd,
            amprnd_color=amprnd_color,
        )
        save_fig(fig_dir, f"ex_amprnd")

        plot_amp5s_amprnd(
            x,
            amp5s=filt_amp5s,
            amp5s_color=amp5s_color,
        )
        save_fig(fig_dir, f"ex_filt_amp5s")

        plot_amp5s_amprnd(
            x,
            amprnd=filt_amprnd,
            amprnd_color=amprnd_color,
        )
        save_fig(fig_dir, f"ex_filt_amprnd")

        plt.show()


def micro_fig3(num_regs, results_dir, fig_dir):
    fig_dir = os.path.join(fig_dir, "micro", "fig3")

    amprnd_color = "gray"
    amp5s_color = "green"

    amp_lw = 0.5

    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)
    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)

    set_fig_size(0.4, 1)
    plot_reliability_split_bar(
        fig_dir,
        df_stat,
        num_regs,
        ["filt_amprnd", "filt_amp5s"],
        ["$\Delta F/F$ 110 sec", "$\Delta F/F$ 5 sec"],
        False,
    )
    save_fig(fig_dir, "reliability_bar_amp_filt")

    print(f"dff.shape: {dff.shape}")

    x = np.linspace(0, CELL_LENGTH, dff.shape[2])

    print(df_stat.head())
    df_shuffled = df_stat.sample(frac=1)
    print(df_shuffled["exp_name"].size)
    for _, row in df_shuffled.iterrows():
        exp_name = row["exp_name"]

        roi_number = row["roi_number"]
        print(f"\nexp_name: {exp_name}\nroi_number: {roi_number}\n")

        cell_mask = (df_stat["exp_name"] == exp_name) & (
            df_stat["roi_number"] == roi_number
        )

        bl_singles = []
        amp5s_singles = []
        amprnd_singles = []
        filt_bl_singles = []
        filt_amp5s_singles = []
        filt_amprnd_singles = []
        rows = df_stat[cell_mask]

        for _, row in rows.iterrows():
            bl_singles.append(
                np.array([row[f"bl_r{reg_num}"] for reg_num in range(num_regs)]) * 100
            )
            amp5s_singles.append(
                np.array([row[f"amp5s_r{reg_num}"] for reg_num in range(num_regs)])
                * 100
            )
            amprnd_singles.append(
                np.array([row[f"amprnd_r{reg_num}"] for reg_num in range(num_regs)])
                * 100
            )

            filt_bl_singles.append(
                np.array([row[f"filt_bl_r{reg_num}"] for reg_num in range(num_regs)])
                * 100
            )
            filt_amp5s_singles.append(
                np.array([row[f"filt_amp5s_r{reg_num}"] for reg_num in range(num_regs)])
                * 100
            )
            filt_amprnd_singles.append(
                np.array(
                    [row[f"filt_amprnd_r{reg_num}"] for reg_num in range(num_regs)]
                )
                * 100
            )

        bl, _, _ = calc_bl_amp5s_amprnd(bl_singles, amp5s_singles, amprnd_singles)

        _, filt_amp5s, filt_amprnd = calc_bl_amp5s_amprnd(
            filt_bl_singles, filt_amp5s_singles, filt_amprnd_singles
        )

        set_fig_size(0.6, 0.5)
        _, ax = plot_heatmap(
            dff, cell_mask, t_max=115, light_lw=amp_lw, include_baseline=True, bl=bl
        )
        plot_amp_times(ax, x, amp5s_color, amprnd_color, lw=amp_lw)
        add_scale_bar(
            ax, 0.5, 10, f"10 \u03bcm", loc="lower right", borderpad=1, color="darkgray"
        )
        save_fig(fig_dir, f"ex_heatmap_multi_w_bl")

        _, ax = plot_heatmap(
            dff, cell_mask, t_max=115, light_lw=amp_lw, include_baseline=False, bl=bl
        )
        plot_amp_times(ax, x, amp5s_color, amprnd_color, lw=amp_lw)
        add_scale_bar(
            ax, 0.5, 10, f"10 \u03bcm", loc="lower right", borderpad=1, color="darkgray"
        )
        save_fig(fig_dir, f"ex_heatmap_multi")

        set_fig_size(0.25, 1)
        reliability = calc_reliability(filt_amp5s_singles)
        plot_amp5s_amprnd(
            x,
            amp5s=filt_amp5s,
            amp5s_color=amp5s_color,
            amp5s_single=filt_amp5s_singles,
        )
        print(f"reliability filt_amp5s: {reliability}")
        save_fig(fig_dir, f"ex_filt_amp5s_multi")

        reliability = calc_reliability(filt_amprnd_singles)
        plot_amp5s_amprnd(
            x,
            amprnd=filt_amprnd,
            amprnd_color=amprnd_color,
            amprnd_single=filt_amprnd_singles,
        )
        print(f"reliability filt_amprnd: {reliability}")
        save_fig(fig_dir, f"ex_filt_amprnd_multi")

        plt.show()


def micro_sfig3(num_regs, results_dir, fig_dir):
    fig_dir = os.path.join(fig_dir, "micro", "sfig")

    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)

    space_sr = (num_regs - 1) / CELL_LENGTH
    x = np.linspace(0, CELL_LENGTH, num_regs)

    set_fig_size(0.4, 0.7)
    for group, mask, color in zip(
        get_group_labels(), ctrl_ptz_mask(df_stat), CTRL_PTZ_COLORS
    ):
        _, ax = plt.subplots()
        df_g = df_stat[mask]
        amp = np.array(
            [df_g[f"filt_amp5s_r{reg_num}"].to_numpy() for reg_num in range(num_regs)]
        )  # region x num_trials
        num_trials = amp.shape[1]

        mu_amp = np.mean(amp, axis=1)
        sem_amp = np.std(amp, axis=1) / np.sqrt(num_trials)

        ax.plot(x, mu_amp, alpha=0.9, color=color, label=group)
        for trial_num in range(num_trials):
            ax.plot(x, amp[:, trial_num], alpha=0.1, color=color, linewidth=0.4)

        ax.set_ylabel("$\Delta F/F$ 5 sec")
        ax.set_xlabel("Distal <-> Proximal [$\mu m$]")
        ax.legend()
        save_fig(fig_dir, f"filt_amp5s_singles_{group}")

    set_fig_size(0.5, 0.6)
    _, ax = plt.subplots()
    for group, mask, color in zip(
        get_group_labels(), ctrl_ptz_mask(df_stat), CTRL_PTZ_COLORS
    ):
        df_g = df_stat[mask]
        amp = np.array(
            [df_g[f"filt_amp5s_r{reg_num}"].to_numpy() for reg_num in range(num_regs)]
        )  # region x num_trials
        num_trials = amp.shape[1]

        amp_norm = (amp - np.mean(amp, axis=0)) / np.std(amp, axis=0)

        corrs = []
        for trial_num in range(num_trials):
            x = amp_norm[:, trial_num]
            corrs.append(signal.correlate(x, x, mode="same") / num_regs)
        corrs = np.array(corrs).T

        lags = np.arange(num_regs) / space_sr
        lags = lags - np.amax(lags) / 2

        sig_lim = 2.33 / np.sqrt(num_regs)

        corrs = corrs[lags > 0]
        lags = lags[lags > 0]

        mu_corr = np.mean(corrs, axis=1)
        sem_corr = np.std(corrs, axis=1) / np.sqrt(num_trials)

        ax.plot(lags, mu_corr, alpha=0.9, color=color, label=group)
        ax.fill_between(
            lags, mu_corr - sem_corr, mu_corr + sem_corr, alpha=0.3, color=color
        )

    ax.axhline(sig_lim, linestyle="dashed", color="gray", alpha=0.7)
    ax.axhline(-sig_lim, linestyle="dashed", color="gray", alpha=0.7)
    ax.set_ylabel("Autocorrelation (mean $\pm$ SEM)")
    ax.set_xlabel("Lag [$\mu m$]")
    ax.legend()
    save_fig(fig_dir, "autocorr_sem")


def micro_generating_code(num_regs, results_dir, fig_dir):
    fig_dir = os.path.join(fig_dir, "micro")
    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)
    """ df_stat_6 = load_pickle(results_dir, STATS_FNAME, 6)

    masks, labels, labels_pretty = get_t_onset_masks(df_stat_6, 6)

    plot_reliability_split_bar(fig_dir, df_stat, num_regs, "bl", "amp5s", "amprnd")
    save_fig(fig_dir, "reliability_bar") """

    plot_reliability_split_bar(
        fig_dir,
        df_stat,
        num_regs,
        ["filt_amprnd", "filt_amp5s"],
        ["Spontaneous", "Light onset"],
        False,
    )
    save_fig(fig_dir, "reliability_bar_amp_filt")

    """ plot_reliability_split_bar(
        fig_dir, df_stat, num_regs, ["filt_bl"], ["Baseline\nfluorescence"], False
    )
    save_fig(fig_dir, "reliability_bar_bl_filt") """


def ex_micro_generating_code(num_regs, results_dir, fig_dir):
    fig_dir = os.path.join(fig_dir, "micro")

    df_stat_6 = load_pickle(results_dir, STATS_FNAME, 6)
    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)
    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)
    dff_6 = load_np(results_dir, DFF_REGS_FNAME, 6)

    masks, labels, _ = get_t_onset_masks(df_stat_6, 6)

    cell_mask = (df_stat_6["ptz"] == True) & (df_stat_6["roi_number"] == 5)
    cell_mask = (df_stat_6["ptz"] == False) | (df_stat_6["ptz"] == True)

    rng = np.random.default_rng()

    set_fig_size(0.48, 0.7)
    for mask, label in zip(masks, labels):
        print(label)
        c_mask = mask & cell_mask
        num_trials = c_mask.shape[0]
        mask_ex = np.zeros(num_trials, dtype=bool)
        random_inds = rng.choice(np.arange(num_trials), num_trials, replace=False)
        for ind in random_inds:
            if c_mask[ind]:
                mask_ex[ind] = True

                row = df_stat[mask_ex]
                bl = np.squeeze(
                    np.array([row[f"bl_r{reg_num}"] for reg_num in range(num_regs)])
                    * 100
                )

                plot_trace(dff_6, mask_ex, t_max=60)
                save_fig(fig_dir, f"ex_trace_{label}")
                plot_heatmap(dff, mask_ex, t_max=60)
                save_fig(fig_dir, f"ex_heatmap_{label}")
                plot_heatmap(dff, mask_ex, t_max=60, include_baseline=True, bl=bl)
                save_fig(fig_dir, f"ex_heatmap_w_bl_{label}")
                plt.show()

                mask_ex[ind] = False

                cont = input("continue? (y/n)")
                if cont == "n":
                    break


def freq_analysis_generating_code(num_regs, results_dir, fig_dir):
    fig_dir = os.path.join(fig_dir, "freq")

    d2f = load_np(results_dir, SECOND_DERIVATIVE_FNAME, num_regs)

    print(f"d2f.shape: {d2f.shape}")

    fft = np.fft.fft(d2f, axis=1)
    n = fft.shape[1]
    freqs = np.fft.fftfreq(n, 1 / VOLUME_RATE)

    fft = fft[:, : n // 2]
    freqs = freqs[: n // 2]
    reg_colors = get_region_colors(num_regs)

    set_fig_size(0.48, 0.7)

    plt.figure()
    for reg_num, reg_col in zip(range(num_regs), reg_colors):
        if reg_num == 0:
            plt.plot(
                freqs,
                10 * np.log10(np.absolute(np.mean(fft[:, :, reg_num], axis=0))),
                color=reg_col,
                alpha=MARKER_ALPHA,
                label="Distal",
            )
        elif reg_num == num_regs - 1:
            plt.plot(
                freqs,
                10 * np.log10(np.absolute(np.mean(fft[:, :, reg_num], axis=0))),
                color=reg_col,
                alpha=MARKER_ALPHA,
                label="Proximal",
            )
        else:
            plt.plot(
                freqs,
                10 * np.log10(np.absolute(np.mean(fft[:, :, reg_num], axis=0))),
                color=reg_col,
                alpha=MARKER_ALPHA,
            )

    plt.legend()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("10 log(mean(FFT))")

    save_fig(fig_dir, "mean_raw_second_derivative")

    plt.figure()
    for reg_num, reg_col in zip(range(num_regs), reg_colors):

        if reg_num == 0:
            plt.plot(
                freqs,
                10 * np.log10(np.absolute(np.std(fft[:, :, reg_num], axis=0))),
                color=reg_col,
                alpha=MARKER_ALPHA,
                label="Distal",
            )
        elif reg_num == num_regs - 1:
            plt.plot(
                freqs,
                10 * np.log10(np.absolute(np.std(fft[:, :, reg_num], axis=0))),
                color=reg_col,
                alpha=MARKER_ALPHA,
                label="Proximal",
            )
        else:
            plt.plot(
                freqs,
                10 * np.log10(np.absolute(np.std(fft[:, :, reg_num], axis=0))),
                color=reg_col,
                alpha=MARKER_ALPHA,
            )

    plt.legend()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("10 log(std(FFT))")

    save_fig(fig_dir, "std_raw_second_derivative")


def print_sample_size(num_regs, results_dir, fig_dir):

    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)

    ctrl_mask, ptz_mask = ctrl_ptz_mask(df_stat)

    group_labels = get_group_labels()

    for group, g_mask in zip(group_labels, [ctrl_mask, ptz_mask]):
        sample_size = get_sample_size(df_stat, g_mask)
        print(f"\n{group}:\n{sample_size}")

    sample_size = get_sample_size(df_stat, ctrl_mask | ptz_mask)
    print(f"\nAll:\n{sample_size}")


def main():
    results_dir = generate_global_results_dir()
    fig_dir = generate_figures_dir()

    num_reg = 4
    """ print_sample_size(num_reg, results_dir, fig_dir)
    amp_regions_generating_code(num_reg, results_dir, fig_dir)
    plt.close("all")

    t_lag_regions_generating_code(num_reg, results_dir, fig_dir)
    t_onset_regions_generating_code(num_reg, results_dir, fig_dir)
    plt.close("all") """

    amp_t_onset_generating_code(num_reg, results_dir, fig_dir)
    peak_t_onset_generating_code(num_reg, results_dir, fig_dir)
    bl_t_onset_generating_code(num_reg, results_dir, fig_dir)
    plt.close("all")

    """ clustering_generating_code(num_reg, results_dir, fig_dir)
    freq_analysis_generating_code(num_reg, results_dir, fig_dir)
    plt.close("all")

    schematic_calcium_event_detection(num_reg, results_dir, fig_dir)
    schematic_calcium_wave_detection(num_reg, results_dir, fig_dir)
    plot_schematic_time_lag_distal(num_reg, results_dir, fig_dir) """

    num_reg = 110
    # micro_fig1(num_reg, results_dir, fig_dir)
    # micro_fig2(num_reg, results_dir, fig_dir)
    # micro_fig3(num_reg, results_dir, fig_dir)
    # micro_sfig3(num_reg, results_dir, fig_dir)
    # micro_generating_code(num_reg, results_dir, fig_dir)
    # rep_ex_micro_generating_code(num_reg, results_dir, fig_dir)
    # ex_micro_generating_code(num_reg, results_dir, fig_dir)

    # plt.show()


if __name__ == "__main__":
    main()
