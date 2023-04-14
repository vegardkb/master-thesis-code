import threading
import os
import shutil
from shutil import SameFileError

"""
To copy:
    - config.ini
    - denoised:
        - rois.npy
        - denoised.tif
        - suite2p:
            - *
        - results:
            - *
"""

EXP_NAMES = [
    "20211112_13_23_43_GFAP_GCamp6s_F2_PTZ",
    "20211112_18_30_27_GFAP_GCamp6s_F5_c2",
    "20211112_19_48_54_GFAP_GCamp6s_F6_c3",
    # "20211117_14_17_58_GFAP_GCamp6s_F2_C",
    # "20211117_17_33_00_GFAP_GCamp6s_F4_PTZ",
    "20211117_21_31_08_GFAP_GCamp6s_F6_PTZ",
    "20211119_16_36_20_GFAP_GCamp6s_F4_PTZ",
    # "20211119_18_15_06_GFAP_GCamp6s_F5_C",
    # "20211119_21_52_35_GFAP_GCamp6s_F7_C",
    "20220211_13_18_56_GFAP_GCamp6s_F2_C",
    # "20220211_15_02_16_GFAP_GCamp6s_F3_PTZ",
    "20220211_16_51_15_GFAP_GCamp6s_F4_PTZ",
    # "20220412_10_43_04_GFAP_GCamp6s_F1_PTZ",
    "20220412_12_32_27_GFAP_GCamp6s_F2_PTZ",
    "20220412_13_59_55_GFAP_GCamp6s_F3_C",
    # "20220412_16_06_54_GFAP_GCamp6s_F4_PTZ",
]

CROP_ID = "OpticTectum"


SOURCE = os.path.join("..", "..", "processing", "processed")
DESTINATION = os.path.join("..", "..", "..", "data")


def generate_exp_dirs(exp_name: str, crop_id: str):
    exp_dir_source = os.path.join(SOURCE, exp_name, crop_id)
    if not os.path.isdir(exp_dir_source):
        os.makedirs(exp_dir_source)

    exp_dir_destination = os.path.join(DESTINATION, exp_name, crop_id)
    if not os.path.isdir(exp_dir_destination):
        os.makedirs(exp_dir_destination)

    return exp_dir_source, exp_dir_destination


def copy_file(dir_src, dir_dst, fname):
    if not os.path.join(dir_src, fname):
        print(f"The file {fname} not found in {dir_src}")
        return

    src_path = os.path.join(dir_src, fname)
    if not os.path.isdir(dir_dst):
        os.makedirs(dir_dst)
    dst_path = os.path.join(dir_dst, fname)

    try:
        shutil.copyfile(src_path, dst_path)
    except SameFileError:
        print(f"{fname} already exists in {dst_path}")
    except IsADirectoryError:
        print(f"The destination ({dst_path}) is a directory")


def copy_all_files_recursive(dir_src, dir_dst):
    for fname in os.listdir(dir_src):
        src_path = os.path.join(dir_src, fname)
        if os.path.isfile(src_path):
            copy_file(dir_src, dir_dst, fname)

        elif os.path.isdir(src_path):
            dst_path = os.path.join(dir_dst, fname)
            print(f"source: {src_path}, dist: {dst_path}")
            copy_all_files_recursive(src_path, dst_path)


def copy_exp(exp_dir_source, exp_dir_destination):
    dir_src = exp_dir_source
    dir_dst = exp_dir_destination
    fname = "config.ini"
    copy_file(dir_src, dir_dst, fname)

    dir_src = os.path.join(dir_src, "denoised")
    dir_dst = os.path.join(dir_dst, "denoised")
    fname = "rois.npy"
    copy_file(dir_src, dir_dst, fname)

    fname = "denoised.tif"
    copy_file(dir_src, dir_dst, fname)

    s2p_dir_src = os.path.join(dir_src, "suite2p")
    s2p_dir_dst = os.path.join(dir_dst, "suite2p")
    copy_all_files_recursive(s2p_dir_src, s2p_dir_dst)

    results_dir_src = os.path.join(dir_src, "results")
    results_dir_dst = os.path.join(dir_dst, "results")
    copy_all_files_recursive(results_dir_src, results_dir_dst)


def main():
    threads = []

    for exp_name in EXP_NAMES:
        exp_dir_source, exp_dir_destination = generate_exp_dirs(exp_name, CROP_ID)
        t = threading.Thread(
            target=copy_exp, args=(exp_dir_source, exp_dir_destination)
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
