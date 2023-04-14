import numpy as np
import os
import threading

from suite2p.run_s2p import run_s2p

from s2p_util import load_cfg, generate_exp_dir, write_cfg_to_ops, create_db_denoised, create_db_normal, process_h5, crop_h5

WORKING_DIR = os.path.join("..", "..", "processed")

USE_DENOISED = True
DENOISED_DIR = "denoised"
DENOISED_FNAME = "denoised.h5"
DENOISED_CHAN2_DIR = "denoised_chan2"
DENOISED_CHAN2_FNAME = "denoised_chan2.h5"

""" EXP_NAMES = [
    "20211112_13_23_43_GFAP_GCamp6s_F2_PTZ",
    "20220211_13_18_56_GFAP_GCamp6s_F2_C",
    "20220211_15_02_16_GFAP_GCamp6s_F3_PTZ",
    "20220412_13_59_55_GFAP_GCamp6s_F3_C",
    "20220531_12_34_15_HuC_GCamp6s_GFAP_jRGECO_F1_C",
    "20220531_14_21_02_HuC_GCamp6s_GFAP_jRGECO_F2_PTZ",
] """
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

OPS_FNAME = "ops_celldetect.npy"


def get_input_dir(exp_dir, channel_number, use_denoised):
    if use_denoised:
        if channel_number == 1:
            input_dir = os.path.join(exp_dir, DENOISED_DIR)
        else:
            input_dir = os.path.join(exp_dir, DENOISED_CHAN2_DIR)
    else:
        input_dir = exp_dir

    return input_dir


def cell_detect(exp_dir, cfg, ops, channel_number, denoised_fname):
    cfg.n_ch = 1
    ops = write_cfg_to_ops(ops, cfg)

    input_dir = get_input_dir(exp_dir, channel_number, USE_DENOISED)

    print(f"input_dir: {input_dir}")

    success = process_h5(input_dir, cfg, denoised_fname)
    if not success:
        return
    
    if USE_DENOISED:
        db = create_db_denoised(input_dir, denoised_fname)

    else:
        db = create_db_normal(input_dir)

    _ = run_s2p(ops, db)  # Returns ops, but currently unused


def main():
    threads = []
    for exp_name in EXP_NAMES:
        for crop_id in CROP_IDS:
            exp_dir = generate_exp_dir(WORKING_DIR, exp_name, crop_id)

            if not os.path.exists(exp_dir):
                print(f"Skipping directory (does not exist): {exp_dir}")
                continue
            else:
                print(f"Loading files from {exp_dir}")

            cfg = load_cfg(exp_dir)
            ops = np.load(OPS_FNAME, allow_pickle=True).item()

            # This seems hacky
            dual_channel = True if cfg.n_ch == 2 else False
            denoised_fnames = [DENOISED_FNAME, DENOISED_CHAN2_FNAME] if dual_channel else [DENOISED_FNAME]
            channel_numbers = [1,2] if dual_channel else [1]

            for denoised_fname, channel_number in list(zip(denoised_fnames, channel_numbers)):
                t = threading.Thread(
                    target=cell_detect,
                    args=(exp_dir, cfg, ops, channel_number, denoised_fname),
                )
                threads.append(t)
                t.start()

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
