import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
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

# ISIS = [300, 120, 60, 30, 15]
ISIS = [300, 120, 60]

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
STIM_DURATION = 10
POST_STIM_T = 5
POST_EVENT_T_PEAK = 40

CTRL_PTZ_COLORS = ("gray", "brown")
AMP_CAT_COLORS = (DARKGRAY, GREEN, BROWN)
AMP_CAT_COLORS_DICT = {
    "negative": DARKGRAY,
    "distal-dominant": GREEN,
    "proximal-dominant": BROWN,
}

PEAK_CAT_COLORS = (DARKGRAY, GREEN, BROWN)
PEAK_CAT_COLORS_DICT = {
    "negative": DARKGRAY,
    "distal-dominant": GREEN,
    "proximal-dominant": BROWN,
}


T_ONSET_CAT_COLORS = (ORANGE, LIGHTBLUE, DARKGRAY)
T_ONSET_CAT_COLORS_DICT = {
    "synchronized": ORANGE,
    "calcium_wave": LIGHTBLUE,
    "undefined": DARKGRAY,
}

""" T_ONSET_CAT_COLORS_DICT = {
    "distal_proximal": ORANGE,
    "proximal_distal": LIGHTBLUE,
    "simultaneous": DARKGRAY,
} """


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

    """ lon_s, lon_e = -0.5, 4
    loff_s, loff_e = 9.5, 14 """
    lon_s, lon_e = 0, 4
    loff_s, loff_e = 10, 14

    mean_t, med_t = 0, 0

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
    """ wave = ~sync & (
        (df["t_onset_lag_mean"] > mean_t)  # & ((df["t_onset_lag_median"] > med_t))
    ) """
    wave = ~sync & (
        (df[f"t_onset_r{distal_reg}"] < 10) & (df["t_onset_lag_median"] > med_t)
    )
    undefined = ~wave & ~sync

    category_masks = [sync, wave, undefined]
    category_labels = ["synchronized", "calcium_wave", "undefined"]
    categor_labels_pretty = ["synchronized", "calcium-wave", "undefined"]

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


def mask_sort_df(df, mask, sort_columns):
    masked_df = df[mask]
    sorted_df = masked_df.sort_values(by=sort_columns)
    return sorted_df


def plot_light_on_off(ax, lw=0.7, ymin=0.05, ymax=0.95):
    ax.axvline(0, ymin, ymax, color="orange", linestyle="--", linewidth=lw, alpha=0.75)
    ax.axvline(
        10, ymin, ymax, color="darkgray", linestyle="--", linewidth=lw, alpha=0.75
    )


def plot_amp_times(ax, x, amp4s_color, amp14s_color, lw=0.7):
    ax.axvline(
        4,
        np.amin(x),
        np.amax(x),
        color=amp4s_color,
        alpha=0.75,
        linewidth=lw,
    )
    ax.axvline(
        14,
        np.amin(x),
        np.amax(x),
        color=amp14s_color,
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

    for ind_cat, cat_mask, cat_name, color in zip(
        range(len(cat_masks)), cat_masks, cat_names, colors
    ):
        df_cat = df[cat_mask]
        if percent:
            scatter_hist(
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
            scatter_hist(
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

    x_ticks = [-100, 0, 100, 200]
    y_ticks = [-100, 0, 100, 200, 300, 400, 500]

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.legend(loc="upper left")
    if labels is None:
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    else:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

    ax_histx.axhline(color="darkgray", alpha=0.7)
    ax_histy.axvline(color="darkgray", alpha=0.7)
    ax_histx.set_yticks([0, 10])
    ax_histy.set_xticks([0, 10])

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])


def plot_scatter_cont_color(
    df,
    x_col,
    y_col,
    color_col,
    cmap,
    size_col=None,
    size_lim=None,
    clim=None,
    axlim=None,
):
    s = get_dot_size(df, size_col, size_lim)

    plt.figure()
    plt.scatter(
        df[x_col], df[y_col], s=s, c=df[color_col], cmap=cmap, alpha=MARKER_ALPHA
    )
    cbar = plt.colorbar()
    cbar.set_label(color_col)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if clim is not None:
        plt.clim(clim[0], clim[1])

    if axlim is not None:
        plt.xlim((axlim[0], axlim[1]))
        plt.ylim((axlim[0], axlim[1]))


def plot_tuning_categories(
    df,
    x_col,
    y_col,
    colors,
    cat_masks,
    cat_names,
    mask_mask=None,
    num_bins=50,
):

    plt.figure()
    for cat_mask, cat_name, color in zip(cat_masks, cat_names, colors):
        if mask_mask is not None:
            cat_mask = copy.deepcopy(cat_mask)
            cat_mask[~mask_mask] = False

        df_cat = df[cat_mask]

        x_s = df_cat[x_col]
        y_s = df_cat[y_col]

        x_min, x_max = x_s.min(), x_s.max()

        ts = np.linspace(x_min, x_max, num_bins)
        x = []
        y = []

        for t_l, t_h in zip(ts[:-1], ts[1:]):
            bin_mask = (x_s > t_l) & (x_s < t_h)
            if np.any(bin_mask):
                x.append((t_l + t_h) / 2)
                y.append(y_s[bin_mask].mean())

        plt.plot(
            x,
            y,
            color=color,
            label=cat_name,
            alpha=0.9,
        )

    plt.legend()
    plt.xlabel(x_col)
    plt.ylabel(y_col)


def plot_split_bar(
    exp_names: pd.Series,
    masks,
    labels,
    split_mask,
    colors,
    test_sig=False,
):
    num_cat = len(labels)
    ns = []
    for s_mask in [split_mask, ~split_mask]:
        s_exp_names = exp_names[s_mask]
        unique_names = s_exp_names.unique()
        n = np.zeros((num_cat, len(unique_names)))

        for i, mask in enumerate(masks):
            for j, name in enumerate(unique_names):
                n[i, j] = np.sum(mask & (exp_names == name)) / np.sum(exp_names == name)

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


def plot_cell_overview(df, cat_column, colors):
    df_sort = df.sort_values(by=["ptz", "exp_name", "roi_number", "evt_num"])
    cat_s = df_sort[cat_column]
    ptz_s = df_sort["ptz"]
    exp_s = df_sort["exp_name"]

    num_evts = 5
    cell_im = np.zeros((num_evts, 3))
    empty_row = np.ones((num_evts, 3))
    cat_im = []

    y_ticks = []
    y_labels = []

    num_cells_exp = 0

    e_ind = 0
    im_ind = 0
    prev_ptz = None
    prev_exp = None

    ptz_count = 0
    ctrl_count = 0

    for cat, ptz, exp in zip(cat_s, ptz_s, exp_s):
        if prev_ptz is None:
            prev_ptz = ptz
            prev_exp = exp

        if exp != prev_exp:
            print("Fish change")
            y_ticks.append(im_ind - int(num_cells_exp / 2))
            if prev_ptz:
                y_labels.append("PTZ " + str(ptz_count))
                ptz_count += 1
            else:
                y_labels.append("Control " + str(ctrl_count))
                ctrl_count += 1
            cat_im.append(empty_row)
            im_ind += 1
            prev_exp = exp
            num_cells_exp = 0

        if ptz != prev_ptz:
            print("Group change")
            cat_im.append(empty_row)
            im_ind += 1
            prev_ptz = ptz

        if e_ind == num_evts:
            e_ind = 0
            cat_im.append(copy.deepcopy(cell_im))
            im_ind += 1
            num_cells_exp += 1

        cell_im[e_ind] = colors[cat]
        e_ind += 1

    y_ticks.append(im_ind - int(num_cells_exp / 2))
    if prev_ptz:
        y_labels.append("PTZ " + str(ptz_count))
        ptz_count += 1
    else:
        y_labels.append("Control " + str(ctrl_count))
        ctrl_count += 1

    cat_im = np.array(cat_im)
    cat_im_rgba = np.zeros((cat_im.shape[0], cat_im.shape[1], 4))
    cat_im_rgba[:, :, :3] = cat_im
    cat_im_rgba[:, :, 3] = MARKER_ALPHA
    background_im = np.ones(cat_im.shape)

    plt.figure()
    plt.imshow(background_im, aspect="auto")
    plt.imshow(cat_im, aspect="auto")
    plt.yticks(y_ticks, y_labels)
    plt.ylabel("Cells each fish")
    plt.xlabel("Event number")
    plt.tight_layout()


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
    # ax.spines[["right", "top"]].set_visible(False)


def plot_heatmap(
    dff, mask, title=None, cmap=cm.inferno, t_max=None, norm=False, light_lw=0.5
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
        vmin=max(-50, np.amin(av_dff)),
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


def plot_split_violin_lag_corr(df, num_regs, interval="whole"):
    ctrl_mask, ptz_mask = ctrl_ptz_mask(df)

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
        ("Control", "PTZ"),
        CTRL_PTZ_COLORS,
        "Lag [s]",
        labels,
        title=group,
    )

    plot_split_violin(
        (corr_reg_ctrl, corr_reg_ptz),
        ("Control", "PTZ"),
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


def plot_fr_regions(df_ts, t_col, size_col, mask, num_regs, title=None):
    reg_colors = get_region_colors(num_regs)
    reg_labels = get_region_labels(num_regs)

    df_t = df_ts[mask]
    num_bins = int(25)
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

    _, ax = plt.subplots()
    for reg_num in range(num_regs):

        if (not (reg_num == 0 or reg_num == (num_regs - 1))) and num_regs > 3:
            ax.plot(
                t,
                y_dc[reg_num] + fr[:, reg_num],
                color=reg_colors[reg_num],
            )
        else:
            ax.plot(
                t,
                y_dc[reg_num] + fr[:, reg_num],
                color=reg_colors[reg_num],
                label=reg_labels[reg_num],
            )

    for y0 in y_dc:
        ax.axhline(y0, np.amin(t), np.amax(t), color="darkgray", alpha=0.7)

    plot_light_on_off(ax)

    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Region/Average event density")

    ax.legend()

    if title is not None:
        ax.set_title(title)


def plot_av_d2xdt2_regions(d2xdt2, mask, title):
    num_regs = d2xdt2.shape[2]
    reg_colors = get_region_colors(num_regs)

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


def calc_relative_diff(series_list, ind_ref):
    rel_diff_series_list = []
    ref_series = series_list[ind_ref]
    for series in series_list:
        rel_diff_series_list.append((series - ref_series) / ref_series * 100)

    return rel_diff_series_list


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


def amp_regions_generating_code(num_regs, results_dir, fig_dir):
    distal_reg = 0
    proximal_reg = num_regs - 1
    fig_dir = os.path.join(fig_dir, "amplitude")

    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)
    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)

    ctrl_mask, ptz_mask = ctrl_ptz_mask(df_stat)

    labels = get_region_labels(num_regs)
    group_labels = ["Control", "PTZ"]

    set_fig_size(0.8, 0.7)
    for group, g_mask in zip(["Control", "PTZ"], [ctrl_mask, ptz_mask]):
        plot_trace(dff, g_mask, plot_std=True, ylim=(-25, 150))
        save_fig(fig_dir, f"trace_av_std_{group}")
        plot_trace(dff, g_mask, plot_sem=True, ylim=(-25, 150))
        save_fig(fig_dir, f"trace_av_sem_{group}")
        plot_trace(dff, g_mask, plot_all=True, ylim=(-100, 300))
        save_fig(fig_dir, f"trace_av_singles_{group}")

    x = df_stat[f"amp_r{distal_reg}"].to_numpy()
    y = df_stat[f"amp_r{proximal_reg}"].to_numpy()
    a = df_stat["ptz"].to_numpy()

    mixreg, _, y_mix = linear_mixed_model(x, a, y)
    p = likelihood_ratio_test(x, a, y)
    print(f"likelihood-ratio-test: {p}")
    rsq_mix = rsquared(y, y_mix)

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

    """ for trial_num in range(5):
        mask = df_stat["evt_num"] == trial_num
        df_trial, dff_trial = df_stat[mask], dff[mask]

        ctrl_mask, ptz_mask = ctrl_ptz_mask(df_trial)

        labels = get_region_labels(num_regs)
        group_labels = ("Control", "PTZ")

        set_fig_size(0.8, 0.7)
        for group, g_mask in zip(group_labels, [ctrl_mask, ptz_mask]):
            plot_trace(dff_trial, g_mask, plot_std=True, fill_std=True, ylim=(-25, 150))
            save_fig(fig_dir, f"trace_av_std_{group}_trial_{trial_num}")

        x = df_trial[f"amp_r{distal_reg}"].to_numpy()
        y = df_trial[f"amp_r{proximal_reg}"].to_numpy()
        a = df_trial["ptz"].to_numpy()

        mixreg, _, y_mix = linear_mixed_model(x, a, y)
        p = likelihood_ratio_test(x, a, y)
        print(f"likelihood-ratio-test: {p}")
        rsq_mix = rsquared(y, y_mix)

        rs_distal = ranksums(x[a], x[np.logical_not(a)])
        rs_proximal = ranksums(y[a], y[np.logical_not(a)])

        print(f"rank-sum distal amp: {rs_distal.pvalue}")
        print(f"rank-sum proximal amp: {rs_proximal.pvalue}")

        set_fig_size(0.8, 1)
        plot_scatter_hist_categories(
            df_trial,
            f"amp_r{distal_reg}",
            f"amp_r{proximal_reg}",
            CTRL_PTZ_COLORS,
            (ctrl_mask, ptz_mask),
            group_labels,
            labels=(
                r"Distal amplitude $\Delta F / F_0$ [%]",
                r"Proximal amplitude $\Delta F / F_0$ [%]",
            ),
            binsize=15,
            percent=True,
            mixreg=mixreg,
        )
        save_fig(
            fig_dir, f"amp_scatter_hist_distal_proximal_ptz_ctrl_trial_{trial_num}"
        ) """


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

    """ plot_scatter_categories(
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
    ) """

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

    """ plot_scatter_categories(
        df_stat,
        f"peak_r{distal_reg}",
        f"peak_r{proximal_reg}",
        PEAK_CAT_COLORS,
        masks,
        labels,
        labels=(
            r"Distal peak $\Delta F / F_0$ [%]",
            r"Proximal peak $\Delta F / F_0$ [%]",
        ),
        percent=True,
    ) """

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

    plot_cell_overview(df_stat, "peak_cat", PEAK_CAT_COLORS_DICT)
    save_fig(fig_dir, "cell_category_overview")


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
        # significant = num_sig == num_regs
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
                ["Control", "PTZ"], [ctrl_mask, ptz_mask]
            ):
                c_label = group_label + "_" + label
                c_mask = mask & group_mask & significant

                plot_trace(dff, c_mask, ylim=(-50, 150), t_max=60)
                save_fig(fig_dir, f"trace_{lag_type}_{c_label}")


def schematic_calcium_event_detection(num_regs, results_dir, fig_dir):
    distal_reg = 0
    proximal_reg = num_regs - 1
    fig_dir = os.path.join(fig_dir, "time_onset")
    reg_colors = get_region_colors(num_regs)

    df_t_onsets = load_pickle(results_dir, T_ONSET_FNAME, num_regs)
    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)
    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)
    d2xdt2 = load_np(results_dir, SECOND_DERIVATIVE_FNAME, num_regs)

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

        set_fig_size(0.48, 0.7)
        plot_trace(
            dff,
            mask,
            t_max=15,
            xlim=(-2.5, 15),
            xticks=[0, 5, 10, 15],
        )
        save_fig(fig_dir, "ex_lr")

        plot_trace(
            d2xdt2,
            mask,
            t_max=15,
            xlim=(-2.5, 15),
            xticks=[0, 5, 10, 15],
            ylabel=r"Second derivative $\Delta F / F_0$",
        )
        save_fig(fig_dir, "ex_d2xdt2_lr")

        set_fig_size(0.9, 0.7)
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
        save_fig(fig_dir, "ex_event_scatter")

        plt.show()


def schematic_calcium_wave_detection(num_regs, results_dir, fig_dir):
    distal_reg = 0
    proximal_reg = num_regs - 1
    fig_dir = os.path.join(fig_dir, "time_onset")
    reg_colors = get_region_colors(num_regs)

    df_t_onsets = load_pickle(results_dir, T_ONSET_FNAME, num_regs)
    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)
    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)
    d2xdt2 = load_np(results_dir, SECOND_DERIVATIVE_FNAME, num_regs)

    s = get_dot_size(df_t_onsets, "diff_dxdt", [1.5, 40])

    masks, _, _ = get_t_onset_masks(df_stat, num_regs)

    sync_mask = masks[0]
    seq_mask = masks[1]

    cli_input = ""

    df_shuffled = df_stat.sample(frac=1)
    for p_mask, pattern in zip([sync_mask, seq_mask], ["sync", "seq"]):
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
    distal_reg = 0
    proximal_reg = num_regs - 1
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
    s_all = np.array(get_dot_size(df_t_onsets, size_col, [0.1, 10], logarithmic=False))

    masks, labels, labels_pretty = get_t_onset_masks(df_stat, num_regs)
    onset_masks = [
        stat_mask_to_t_onset_mask(df_stat, df_t_onsets, mask) for mask in masks
    ]

    df_stat["t_onset_cat"] = ""
    for mask, label in zip(masks, labels_pretty):
        df_stat.loc[mask, "t_onset_cat"] = label

    for ptz, group_mask_stat in zip([False, True], [ctrl_mask, ptz_mask]):
        set_fig_size(0.9, 0.7)
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
            xlim=(-2.5, 15),
            xticks=[0, 5, 10, 15],
        )
        save_fig(fig_dir, f"event_scatter_{group}")

        set_fig_size(0.9, 0.7)
        plot_trace(
            dff,
            group_mask_stat,
            xlim=(-2.5, 15),
            xticks=[0, 5, 10, 15],
            ylim=(-10, 140),
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

            plot_trace(dff, c_mask, ylim=(-20, 200), t_max=60)
            save_fig(fig_dir, f"trace_{group_label}")

            """ plot_trace(dff, c_mask, plot_all=True, ylim=(-20, 200), t_max=60)
            save_fig(fig_dir, f"trace_lines_{group_label}")

            plot_trace(dff, c_mask, plot_sem=True, ylim=(-20, 200), t_max=60)
            save_fig(fig_dir, f"trace_sem_{group_label}") """

            """ plot_trace(dff, c_mask, plot_std=True, ylim=(-20, 200), t_max=60)
            save_fig(fig_dir, f"trace_std_{group_label}") """

    set_fig_size(0.7, 0.8)
    plot_split_bar(
        df_stat["exp_name"],
        masks,
        labels_pretty,
        ctrl_mask,
        CTRL_PTZ_COLORS,
        test_sig=True,
    )
    save_fig(fig_dir, "category_bar")

    """ plot_cell_overview(df_stat, "t_onset_cat", T_ONSET_CAT_COLORS_DICT)
    save_fig(fig_dir, "category_overview") """

    """ ptz_masks = [df_t_onsets["ptz"] == False, df_t_onsets["ptz"] == True]
    proximal_mask = df_t_onsets["region"] == proximal_reg

    plot_tuning_categories(
        df_t_onsets,
        "pre_dff",
        "diff_dxdt",
        CTRL_PTZ_COLORS,
        ptz_masks,
        ["Control", "ptz"],
        mask_mask=proximal_mask,
    )
    save_fig(fig_dir, "peak_evt_size")

    plot_tuning_categories(
        df_t_onsets,
        "pre_bl_sub_dff",
        "diff_dxdt",
        CTRL_PTZ_COLORS,
        ptz_masks,
        ["Control", "ptz"],
        mask_mask=proximal_mask,
    )
    save_fig(fig_dir, "amp_evt_size") """


def amp_t_onset_generating_code(num_regs, results_dir, fig_dir):
    distal_reg = 0
    proximal_reg = num_regs - 1
    fig_dir = os.path.join(fig_dir, "amp_time")

    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)
    df_stat_3 = load_pickle(results_dir, STATS_FNAME, 3)
    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)

    masks, labels, labels_pretty = get_t_onset_masks(df_stat, num_regs)
    xlim = (
        df_stat_3[f"amp_r{distal_reg}"].min() * 100 - 5,
        df_stat_3[f"amp_r{distal_reg}"].max() * 100 + 5,
    )
    ylim = (
        df_stat_3[f"amp_r{2}"].min() * 100 - 5,
        df_stat_3[f"amp_r{2}"].max() * 100 + 5,
    )

    df_stat_3["t_onset_cat"] = ""
    for mask, label in zip(masks, labels):
        df_stat_3.loc[mask, "t_onset_cat"] = label

    set_fig_size(0.7, 1)
    for ptz in [False, True]:
        group = "ptz" if ptz else "Control"
        print(f"\n{group}")

        light_seq_mask = np.logical_and(
            np.logical_or(masks[0], masks[1]), df_stat_3["ptz"] == ptz
        )
        df_stat_light_seq = df_stat_3[light_seq_mask]
        x = df_stat_light_seq[f"amp_r{distal_reg}"].to_numpy()
        y = df_stat_light_seq[f"amp_r{2}"].to_numpy()
        a = (df_stat_light_seq["t_onset_cat"] == "calcium_wave").to_numpy()

        mixreg, _, y_mix = linear_mixed_model(x, a, y)
        p = likelihood_ratio_test(x, a, y)
        print(f"likelihood-ratio-test: {p}")
        rsq_mix = rsquared(y, y_mix)

        rs_distal = ranksums(x[a], x[np.logical_not(a)])
        rs_proximal = ranksums(y[a], y[np.logical_not(a)])

        print(f"rank-sum distal amp: {rs_distal.pvalue}")
        print(f"rank-sum proximal amp: {rs_proximal.pvalue}")

        plot_scatter_categories(
            df_stat_light_seq,
            f"amp_r{distal_reg}",
            f"amp_r{2}",
            (T_ONSET_CAT_COLORS[0], T_ONSET_CAT_COLORS[1]),
            (masks[0], masks[1]),
            (labels_pretty[0], labels_pretty[1]),
            labels=(
                r"Distal amplitude $\Delta F / F_0$ [%]",
                r"Proximal amplitude $\Delta F / F_0$ [%]",
            ),
            percent=True,
            title=f"p-value = {p:.03g}, R^2 = {rsq_mix:.03g}",
            mixreg=mixreg,
            xlim=xlim,
            ylim=ylim,
        )
        save_fig(fig_dir, f"amp_scatter_distal_proximal_sync_wave_{group}")

        plot_scatter_hist_categories(
            df_stat_light_seq,
            f"amp_r{distal_reg}",
            f"amp_r{2}",
            (T_ONSET_CAT_COLORS[0], T_ONSET_CAT_COLORS[1]),
            (masks[0], masks[1]),
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

    """ masks, labels, labels_pretty = get_amp_t_onset_masks(
        df_stat, df_stat_3, num_regs, 3
    )

    plot_split_bar(
        df_stat["exp_name"],
        masks,
        labels_pretty,
        df_stat["ptz"] == False,
        CTRL_PTZ_COLORS,
        test_sig=True,
    )
    save_fig(fig_dir, "category_bar")

    set_fig_size(0.48, 0.7)
    ylims = [(-40, 100), (-20, 250)]
    for ptz, ylim in zip([False, True], ylims):
        group = "ptz" if ptz else "Control"
        group_mask_stat = df_stat["ptz"] == ptz
        for mask, label in zip(masks, labels):
            group_label = group + "_" + label
            c_mask = mask & group_mask_stat

            plot_trace(dff, c_mask, ylim=ylim, t_max=60)
            save_fig(fig_dir, f"trace_{group_label}") """


def peak_t_onset_generating_code(num_regs, results_dir, fig_dir):
    distal_reg = 0
    proximal_reg = num_regs - 1
    fig_dir = os.path.join(fig_dir, "peak_time")

    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)
    df_stat_3 = load_pickle(results_dir, STATS_FNAME, 3)
    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)

    masks, labels, labels_pretty = get_t_onset_masks(df_stat, num_regs)
    xlim = (
        df_stat_3[f"peak_r{distal_reg}"].min() * 100 - 5,
        df_stat_3[f"peak_r{distal_reg}"].max() * 100 + 5,
    )
    ylim = (
        df_stat_3[f"peak_r{2}"].min() * 100 - 5,
        df_stat_3[f"peak_r{2}"].max() * 100 + 5,
    )

    df_stat_3["t_onset_cat"] = ""
    for mask, label in zip(masks, labels):
        df_stat_3.loc[mask, "t_onset_cat"] = label

    set_fig_size(0.7, 1)
    for ptz in [False, True]:
        group = "ptz" if ptz else "Control"
        light_seq_mask = np.logical_and(
            np.logical_or(masks[0], masks[1]), df_stat_3["ptz"] == ptz
        )
        df_stat_light_seq = df_stat_3[light_seq_mask]
        x = df_stat_light_seq[f"peak_r{distal_reg}"].to_numpy()
        y = df_stat_light_seq[f"peak_r{2}"].to_numpy()
        a = (df_stat_light_seq["t_onset_cat"] == "sequential").to_numpy()

        mixreg, _, y_mix = linear_mixed_model(x, a, y)
        p = likelihood_ratio_test(x, a, y)
        rsq_mix = rsquared(y, y_mix)

        plot_scatter_categories(
            df_stat_light_seq,
            f"peak_r{distal_reg}",
            f"peak_r{2}",
            (T_ONSET_CAT_COLORS[0], T_ONSET_CAT_COLORS[1]),
            (masks[0], masks[1]),
            (labels_pretty[0], labels_pretty[1]),
            labels=(
                r"Distal peak $\Delta F / F_0$ [%]",
                r"Proximal peak $\Delta F / F_0$ [%]",
            ),
            percent=True,
            title=f"p-value = {p:.03g}, R^2 = {rsq_mix:.03g}",
            mixreg=mixreg,
            xlim=xlim,
            ylim=ylim,
        )
        save_fig(fig_dir, f"peak_scatter_distal_proximal_light_seq_{group}")

    masks, labels, labels_pretty = get_peak_t_onset_masks(
        df_stat, df_stat_3, num_regs, 3
    )

    plot_split_bar(
        df_stat["exp_name"],
        masks,
        labels_pretty,
        df_stat["ptz"] == False,
        CTRL_PTZ_COLORS,
        test_sig=True,
    )
    save_fig(fig_dir, "category_bar")

    set_fig_size(0.48, 0.7)
    ylims = [(-40, 100), (-20, 250)]
    for ptz, ylim in zip([False, True], ylims):
        group = "ptz" if ptz else "Control"
        group_mask_stat = df_stat["ptz"] == ptz
        for mask, label in zip(masks, labels):
            group_label = group + "_" + label
            c_mask = mask & group_mask_stat

            plot_trace(dff, c_mask, ylim=ylim, t_max=60)
            save_fig(fig_dir, f"trace_{group_label}")


def clustering_generating_code(num_regs, results_dir, fig_dir):
    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)
    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)

    t = np.arange(dff.shape[1]) / VOLUME_RATE - PRE_EVENT_T
    t_mask = t < 20

    dff_2d = np.reshape(dff[:, t_mask], (dff.shape[0], -1))

    fig_dir = os.path.join(fig_dir, "clustering")

    group_labels = get_group_labels()
    group_masks = [df_stat["ptz"] == b for b in [False, True]]

    """ 
    fig, ax = plt.subplots()

    ks = np.arange(10) + 1

    inertia_ctrl = []
    inertia_ptz = []
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        for i, mask in enumerate(group_masks):
            kmeans.fit(dff_2d[mask])
            if i == 0:
                inertia_ctrl.append(kmeans.inertia_)
            else:
                inertia_ptz.append(kmeans.inertia_)

    ax.plot(ks, inertia_ctrl, "x-", color=CTRL_PTZ_COLORS[0], label=group_labels[0])
    ax.plot(ks, inertia_ptz, "x-", color=CTRL_PTZ_COLORS[1], label=group_labels[1])

    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")

    plt.show() """

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


def plot_amp4s_amp14s(
    x,
    amp4s=None,
    amp14s=None,
    amp4s_color=None,
    amp14s_color=None,
    amp4s_single=None,
    amp14s_single=None,
):
    fig, ax = plt.subplots()

    if amp4s is not None:
        ax.plot(x, amp4s, color=amp4s_color, label=r"$\Delta F / F$ 4 sec")
    if amp14s is not None:
        ax.plot(x, amp14s, color=amp14s_color, label=r"$\Delta F / F$ 14 sec")

    if amp4s_single is not None:
        for amp4s_s in amp4s_single:
            ax.plot(x, amp4s_s, color=amp4s_color, alpha=0.3)
    if amp14s_single is not None:
        for amp14s_s in amp14s_single:
            ax.plot(x, amp14s_s, color=amp14s_color, alpha=0.3)

    ax.legend()

    ax.set_xlabel("Distal <-> Proximal [micrometer]")
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


def plot_autocorr_amp4s_amp14s(
    x, amp4s, amp14s, amp4s_color, amp14s_color, amp4s_single=None, amp14s_single=None
):
    fig, ax = plt.subplots()

    space_sr = 1 / (x[1] - x[0])
    lags, a_amp4s, sig_lim = calc_lag_corr_lim(amp4s, amp4s, space_sr)
    _, a_amp14s, _ = calc_lag_corr_lim(amp14s, amp14s, space_sr)

    pos_mask = lags > 0

    alpha_sig = 0.7
    alpha_single = 0.3

    ax.axhline(sig_lim, linestyle="dashed", color="gray", alpha=alpha_sig)
    ax.axhline(-sig_lim, linestyle="dashed", color="gray", alpha=alpha_sig)

    if amp4s_single is not None:
        for amp4s_s in amp4s_single:
            _, a_amp4s_s, _ = calc_lag_corr_lim(amp4s_s, amp4s_s, space_sr)
            ax.plot(
                lags[pos_mask],
                a_amp4s_s[pos_mask],
                color=amp4s_color,
                alpha=alpha_single,
            )

    if amp14s_single is not None:
        for amp14s_s in amp14s_single:
            _, a_amp14s_s, _ = calc_lag_corr_lim(amp14s_s, amp14s_s, space_sr)
            ax.plot(
                lags[pos_mask],
                a_amp14s_s[pos_mask],
                color=amp14s_color,
                alpha=alpha_single,
            )

    ax.plot(
        lags[pos_mask],
        a_amp4s[pos_mask],
        color=amp4s_color,
        label=r"$\Delta F / F$ 4 sec",
    )
    ax.plot(
        lags[pos_mask],
        a_amp14s[pos_mask],
        color=amp14s_color,
        label=r"$\Delta F / F$ 14 sec",
    )

    ax.set_ylabel("Correlation")
    ax.set_xlabel(f"Lag [micrometer]")
    ax.legend()


def calc_t_amp_amp4s(dff, mask):
    av_dff = np.mean(dff[mask], axis=0)
    dff_peak = av_dff[
        int(PRE_EVENT_T * VOLUME_RATE) : int(
            (PRE_EVENT_T + POST_EVENT_T_PEAK) * VOLUME_RATE
        )
    ]
    t_peak = np.argmax(dff_peak, axis=0) / VOLUME_RATE
    amp = np.amax(dff_peak, axis=0) * 100
    amp4s = dff_peak[int(4 * VOLUME_RATE)] * 100

    return t_peak, amp, amp4s


def calc_reliability(xs, ys=None):
    num_trials = len(xs)
    rs = []
    if ys is not None:
        for ind1 in range(num_trials):
            x1 = xs[ind1]
            for ind2 in range(num_trials):
                """if ind1 == ind2:
                continue"""

                x2 = ys[ind2]
                r, _ = stats.pearsonr(x1, x2)
                rs.append(r)

    else:
        for ind1 in range(num_trials):
            x1 = xs[ind1]
            for ind2 in range(ind1):
                x2 = xs[ind2]

                r, _ = stats.pearsonr(x1, x2)
                rs.append(r)

    return np.mean(np.array(rs))


def rep_ex_micro_generating_code(num_regs, results_dir, fig_dir):
    distal_reg = 0
    proximal_reg = num_regs - 1
    fig_dir = os.path.join(fig_dir, "micro")

    amp4s_color = "green"
    amp14s_color = "blue"

    amp_lw = 0.5

    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)
    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)
    pos_regs = load_np(results_dir, POS_REGS_FNAME, num_regs)

    x = np.linspace(0, CELL_LENGTH, dff.shape[2])

    print(df_stat.head())
    df_shuffled = df_stat.sample(frac=1)
    print(df_shuffled["exp_name"].size)
    for ind, row in df_shuffled.iterrows():
        exp_name = row["exp_name"]
        roi_number = row["roi_number"]
        event_number = row["evt_num"]
        print(f"exp_name: {exp_name}\nroi_number: {roi_number}\n\n")

        cell_mask = (df_stat["exp_name"] == exp_name) & (
            df_stat["roi_number"] == roi_number
        )
        trial_mask = cell_mask & (df_stat["evt_num"] == event_number)

        t_peak = np.array([row[f"t_peak_r{reg_num}"] for reg_num in range(num_regs)])
        amp = np.array([row[f"amp_r{reg_num}"] for reg_num in range(num_regs)]) * 100
        amp4s = (
            np.array([row[f"amp4s_r{reg_num}"] for reg_num in range(num_regs)]) * 100
        )
        amp14s = (
            np.array([row[f"amp14s_r{reg_num}"] for reg_num in range(num_regs)]) * 100
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
        _, ax = plot_heatmap(dff, trial_mask, t_max=60, light_lw=amp_lw)
        plot_amp_times(ax, x, amp4s_color, amp14s_color, lw=amp_lw)
        add_scale_bar(
            ax, 0.5, 10, f"10 \u03bcm", loc="lower right", borderpad=1, color="darkgray"
        )
        save_fig(fig_dir, f"ex_heatmap")

        plot_amp4s_amp14s(x, amp4s, amp14s, amp4s_color, amp14s_color)
        save_fig(fig_dir, f"ex_amps")

        plot_autocorr_amp4s_amp14s(x, amp4s, amp14s, amp4s_color, amp14s_color)
        save_fig(fig_dir, f"ex_autocorr_amps")

        t_peak, amp, amp4s = calc_t_amp_amp4s(dff, cell_mask)

        amp4s_singles = []
        amp14s_singles = []
        rows = df_stat[cell_mask]
        for _, row in rows.iterrows():
            amp4s_singles.append(
                np.array([row[f"amp4s_r{reg_num}"] for reg_num in range(num_regs)])
                * 100
            )
            amp14s_singles.append(
                np.array([row[f"amp14s_r{reg_num}"] for reg_num in range(num_regs)])
                * 100
            )

        _, ax = plot_heatmap(dff, trial_mask, t_max=60, light_lw=amp_lw)
        plot_amp_times(ax, x, amp4s_color, amp14s_color, lw=amp_lw)
        add_scale_bar(
            ax, 0.5, 10, f"10 \u03bcm", loc="lower right", borderpad=1, color="darkgray"
        )
        save_fig(fig_dir, f"ex_heatmap_multi")

        reliability = calc_reliability(amp4s_singles)
        plot_amp4s_amp14s(
            x,
            amp4s=amp4s,
            amp4s_color=amp4s_color,
            amp4s_single=amp4s_singles,
        )
        print(f"reliability amp4s: {reliability}")
        save_fig(fig_dir, f"ex_amp4s_multi")

        reliability = calc_reliability(amp14s_singles)
        plot_amp4s_amp14s(
            x,
            amp14s=amp14s,
            amp14s_color=amp14s_color,
            amp14s_single=amp14s_singles,
        )
        print(f"reliability amp14s: {reliability}")
        save_fig(fig_dir, f"ex_amp14s_multi")

        plot_autocorr_amp4s_amp14s(
            x,
            amp4s,
            amp14s,
            amp4s_color,
            amp14s_color,
            amp4s_single=amp4s_singles,
            amp14s_single=amp14s_singles,
        )
        save_fig(fig_dir, f"ex_autocorr_amps_multi")

        plt.show()


def micro_generating_code(num_regs, results_dir, fig_dir):
    distal_reg = 0
    proximal_reg = num_regs - 1
    fig_dir = os.path.join(fig_dir, "micro")

    amp4s_color = "green"
    amp_color = "blue"

    df_stat = load_pickle(results_dir, STATS_FNAME, num_regs)
    ptz = []
    amp4s_reliabilities = []
    amp14s_reliabilities = []
    cross_reliabilities = []

    exp_name_set = df_stat["exp_name"].unique()
    roi_num_set = df_stat["roi_number"].unique()

    for exp_name in exp_name_set:
        for roi_num in roi_num_set:
            cell_mask = (df_stat["exp_name"] == exp_name) & (
                df_stat["roi_number"] == roi_num
            )

            if not np.any(cell_mask):
                continue

            df_cell = df_stat[cell_mask]
            amp4s_singles = []
            amp14s_singles = []
            for _, trial in df_cell.iterrows():
                amp4s_singles.append(
                    np.array(
                        [trial[f"amp4s_r{reg_num}"] for reg_num in range(num_regs)]
                    )
                    * 100
                )
                amp14s_singles.append(
                    np.array(
                        [trial[f"amp14s_r{reg_num}"] for reg_num in range(num_regs)]
                    )
                    * 100
                )

            amp4s_reliabilities.append(calc_reliability(amp4s_singles))
            amp14s_reliabilities.append(calc_reliability(amp14s_singles))
            cross_reliabilities.append(calc_reliability(amp4s_singles, amp14s_singles))
            ptz.append(df_cell["ptz"].unique()[0])

    ptz = np.array(ptz)
    amp4s_reliabilities = np.array(amp4s_reliabilities)
    amp14s_reliabilities = np.array(amp14s_reliabilities)
    cross_reliabilities = np.array(cross_reliabilities)

    for mask, group_label in zip(
        [df_stat["ptz"] == False, df_stat["ptz"] == True], ["control", "ptz"]
    ):
        sample_size = get_sample_size(df_stat, mask)
        print(f"\n{group_label}:\n{sample_size}")

    set_fig_size(0.48, 1)
    plot_split_bar_ptz(
        ptz,
        [amp4s_reliabilities, amp14s_reliabilities, cross_reliabilities],
        ["4 sec", "14 sec", "4-14 sec"],
        "Inter-Trial Correlation (mean)",
    )
    save_fig(fig_dir, "reliability_bar")


def ex_micro_generating_code(num_regs, results_dir, fig_dir):
    distal_reg = 0
    proximal_reg = num_regs - 1
    fig_dir = os.path.join(fig_dir, "micro")

    df_stat_3 = load_pickle(results_dir, STATS_FNAME, 3)
    df_stat_6 = load_pickle(results_dir, STATS_FNAME, 6)
    dff = load_np(results_dir, DFF_REGS_FNAME, num_regs)
    dff_3 = load_np(results_dir, DFF_REGS_FNAME, 3)

    masks, labels, labels_pretty = get_amp_t_onset_masks(df_stat_6, df_stat_3, 6, 3)

    cell_mask = (df_stat_3["ptz"] == True) & (df_stat_3["roi_number"] == 5)
    cell_mask = (df_stat_3["ptz"] == False) | (df_stat_3["ptz"] == True)

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

                plot_trace(dff_3, mask_ex, t_max=60)
                save_fig(fig_dir, f"ex_trace_{label}")
                plot_heatmap(dff, mask_ex, t_max=60)
                save_fig(fig_dir, f"ex_heatmap_{label}")
                """ plot_heatmap(dff, mask_ex, t_max=60, norm=True)
                save_fig(fig_dir, f"ex_heatmap_norm_{label}") """
                plt.show()

                mask_ex[ind] = False

                cont = input("continue? (y/n)")
                if cont == "n":
                    break


def plot_amp_pos(dff):
    plt.figure()

    amp = np.amax(dff[:, :])
    av_amp = amp.mean()
    x = np.linspace(0, CELL_LENGTH, dff.shape[2])


def main():
    results_dir = generate_global_results_dir()
    fig_dir = generate_figures_dir()

    num_reg = 3
    # amp_regions_generating_code(num_reg, results_dir, fig_dir)
    # peak_regions_generating_code(num_reg, results_dir, fig_dir)
    # t_lag_regions_generating_code(num_reg, results_dir, fig_dir)
    # plot_schematic_time_lag_distal(num_reg, results_dir, fig_dir)

    num_reg = 6
    # schematic_calcium_event_detection(num_reg, results_dir, fig_dir)
    # schematic_calcium_wave_detection(num_reg, results_dir, fig_dir)
    # t_onset_regions_generating_code(num_reg, results_dir, fig_dir)
    amp_t_onset_generating_code(num_reg, results_dir, fig_dir)
    # clustering_generating_code(num_reg, results_dir, fig_dir)

    num_reg = 110
    # rep_ex_micro_generating_code(num_reg, results_dir, fig_dir)
    # micro_generating_code(num_reg, results_dir, fig_dir)

    plt.show()


if __name__ == "__main__":
    main()
