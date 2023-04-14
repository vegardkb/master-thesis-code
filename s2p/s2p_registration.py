import numpy as np
import os
import multiprocessing
import threading
import h5py

from suite2p.run_s2p import run_s2p
from suite2p.io import BinaryFile

from s2p_util import load_cfg, generate_exp_dir, write_cfg_to_ops, create_db_denoised, create_db_normal, CIConfig

WORKING_DIR = os.path.join("..", "..", "processed")
BINARY_REL_PATH = os.path.join("suite2p", "plane0")


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

OPS_FNAME = "ops_registration.npy"
ALIGNED_BIN_FNAME = "data.bin"
ALIGNED_CHAN2_BIN_FNAME = "data_chan2.bin"
H5_FNAME = "aligned.h5"
H5_CHAN2_FNAME = "aligned_chan2.h5"
H5_DATASET_NAME = "data"

USE_DENOISED = False


def run_s2p_wrap(ops, db):
    _ = run_s2p(ops, db)


def bin_to_h5(cfg: CIConfig, exp_dir, aligned_bin_fname, h5_fname):
    binary_path = os.path.join(exp_dir, BINARY_REL_PATH, aligned_bin_fname)

    x_bin = BinaryFile(Ly=cfg.Ly, Lx=cfg.Lx, read_filename=binary_path).data

    x_np = x_bin
    #x_np = np.swapaxes(x_np, 1, 2)

    out_path = os.path.join(exp_dir, h5_fname)
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset(H5_DATASET_NAME, data=x_np)


def bin_to_h5_wrap(cfg: CIConfig, exp_dir):
    if cfg.n_ch == 2:
        for aligned_bin_fname, h5_fname in list(zip([ALIGNED_BIN_FNAME, ALIGNED_CHAN2_BIN_FNAME], [H5_FNAME, H5_CHAN2_FNAME])):
            bin_to_h5(cfg, exp_dir, aligned_bin_fname, h5_fname)

    else:
        bin_to_h5(cfg, exp_dir, ALIGNED_BIN_FNAME, H5_FNAME)


def main():
    ops_list = []
    db_list = []
    cfg_list = []
    exp_dir_list = []
    do_reg_list = []

    for exp_name in EXP_NAMES:
        for crop_id in CROP_IDS:
            exp_dir = generate_exp_dir(WORKING_DIR, exp_name, crop_id)

            aligned_binary_path = os.path.join(exp_dir, BINARY_REL_PATH, ALIGNED_BIN_FNAME)

            if not os.path.exists(exp_dir):
                print(f"Skipping directory (does not exist): {exp_dir}")
                continue
            elif os.path.isfile(aligned_binary_path):
                print(f"Skipping: data already aligned: {exp_dir}")
                do_reg_list.append(False)
            else:
                print(f"Loading files from {exp_dir}")
                do_reg_list.append(True)

            exp_dir_list.append(exp_dir)

            cfg = load_cfg(exp_dir)
            cfg_list.append(cfg)

            ops = np.load(OPS_FNAME, allow_pickle=True).item()

            ops = write_cfg_to_ops(ops, cfg)

            input_dir = exp_dir
            if USE_DENOISED:
                raise(Exception("Experiment not denoised yet"))

            db = create_db_normal(input_dir)

            ops_list.append(ops)
            db_list.append(db)

    processes = []
    for ops, db, do_reg in zip(ops_list, db_list, do_reg_list):
        if do_reg:
            p = multiprocessing.Process(
                target=run_s2p_wrap,
                args=(
                    ops,
                    db,
                ),
            )
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

    """
        Save as h5 file
    """

    threads = []
    for cfg, exp_dir in zip(cfg_list, exp_dir_list):
        t = threading.Thread(target=bin_to_h5_wrap, args=(cfg, exp_dir))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
