import numpy as np
import tifffile
import os
from suite2p.io import BinaryFile
import threading

from bin2tiff_util import load_cfg, generate_exp_dir, generate_denoised_dir

BINARY_REL_PATH = os.path.join("suite2p", "plane0")
ALIGNED_BIN_FNAME = "data.bin"

TIF_FNAME = "denoised.tif"

WORKING_DIR = os.path.join("..", "..", "processed")

EXP_NAMES = [
    "20211112_13_23_43_GFAP_GCamp6s_F2_PTZ",
    "20211112_18_30_27_GFAP_GCamp6s_F5_c2",
    "20211112_19_48_54_GFAP_GCamp6s_F6_c3",
    #"20211117_14_17_58_GFAP_GCamp6s_F2_C",
    "20211117_17_33_00_GFAP_GCamp6s_F4_PTZ",
    "20211117_21_31_08_GFAP_GCamp6s_F6_PTZ",
    "20211119_16_36_20_GFAP_GCamp6s_F4_PTZ",
    "20211119_18_15_06_GFAP_GCamp6s_F5_C",
    "20211119_21_52_35_GFAP_GCamp6s_F7_C",
    "20220211_13_18_56_GFAP_GCamp6s_F2_C",
    #"20220211_15_02_16_GFAP_GCamp6s_F3_PTZ",
    "20220211_16_51_15_GFAP_GCamp6s_F4_PTZ",
    #"20220412_10_43_04_GFAP_GCamp6s_F1_PTZ",
    "20220412_12_32_27_GFAP_GCamp6s_F2_PTZ",
    "20220412_13_59_55_GFAP_GCamp6s_F3_C",
    #"20220412_16_06_54_GFAP_GCamp6s_F4_PTZ",
]
CROP_IDS = ["OpticTectum"]

USE_DENOISED = True
USE_CHAN2 = False

TIMEBIN_FACTOR = 2


def timebin(x, timebin_factor):
    num_frames = x.shape[0]
    remainder = num_frames % timebin_factor
    if remainder > 0:
        x = x[:-remainder]
        num_frames = x.shape[0]

    x_reshaped = np.reshape(
        x, (timebin_factor, num_frames // timebin_factor, x.shape[1], x.shape[2])
    )
    return np.squeeze(np.mean(x_reshaped, axis=0)).astype("<u2")


def bin_to_tif(cfg, exp_dir, timebin_factor):
    binary_path = os.path.join(exp_dir, BINARY_REL_PATH, ALIGNED_BIN_FNAME)

    x_bin = BinaryFile(Ly=cfg.Ly, Lx=cfg.Lx, read_filename=binary_path).data

    if timebin_factor > 1:
        x_bin = timebin(x_bin, timebin_factor)

    out_path = os.path.join(exp_dir, TIF_FNAME)
    tifffile.imwrite(
        out_path,
        x_bin.astype(dtype="<u2"),
        dtype="uint16",
        photometric="minisblack",
    )


if __name__ == "__main__":
    exp_dir_list = []
    cfg_list = []

    for exp_name in EXP_NAMES:
        for crop_id in CROP_IDS:
            exp_dir = generate_exp_dir(WORKING_DIR, exp_name, crop_id)
            cfg = load_cfg(exp_dir)
            if USE_DENOISED:
                exp_dir = generate_denoised_dir(exp_dir, USE_CHAN2)

            exp_dir_list.append(exp_dir)
            cfg_list.append(cfg)

    threads = []
    for cfg, exp_dir in zip(cfg_list, exp_dir_list):
        t = threading.Thread(target=bin_to_tif, args=(cfg, exp_dir, TIMEBIN_FACTOR))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
