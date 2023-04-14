import os
import h5py


from s2p_util import generate_exp_dir

WORKING_DIR = os.path.join("..", "..", "processed")
BINARY_REL_PATH = os.path.join("suite2p", "plane0")


""" EXP_NAMES = [
    "20211112_13_23_43_GFAP_GCamp6s_F2_PTZ",
    "20220211_13_18_56_GFAP_GCamp6s_F2_C",
    "20220211_15_02_16_GFAP_GCamp6s_F3_PTZ",
    "20220412_13_59_55_GFAP_GCamp6s_F3_C",
] """
EXP_NAMES = [
    "20211112_13_23_43_GFAP_GCamp6s_F2_PTZ",
    "20220211_13_18_56_GFAP_GCamp6s_F2_C",
]
CROP_IDS = ["OpticTectum"]

ALIGNED_FNAME = "aligned.h5"
DENOISED_DIR = "denoised"
DENOISED_FNAME = "denoised.h5"
H5PY_KEY = "data"

START_FRAME = 5000
END_FRAME = 6000

DENOISED_OFFSET = 30


def main():
    for exp_name in EXP_NAMES:
        for crop_id in CROP_IDS:
            exp_dir = generate_exp_dir(WORKING_DIR, exp_name, crop_id)

            h5_path = os.path.join(exp_dir, ALIGNED_FNAME)
            crop_h5_path = os.path.join(exp_dir, "small_"+ALIGNED_FNAME) 
            with h5py.File(h5_path, "r") as h5:
                with h5py.File(crop_h5_path, "w") as croph5:
                    croph5[H5PY_KEY] = h5[H5PY_KEY][START_FRAME:END_FRAME]

            denoised_dir = os.path.join(exp_dir, DENOISED_DIR)
            h5_path = os.path.join(denoised_dir, DENOISED_FNAME)
            crop_h5_path = os.path.join(denoised_dir, "small_"+DENOISED_FNAME) 
            with h5py.File(h5_path, "r") as h5:
                with h5py.File(crop_h5_path, "w") as croph5:
                    croph5[H5PY_KEY] = h5[H5PY_KEY][START_FRAME+DENOISED_OFFSET:END_FRAME+DENOISED_OFFSET]


if __name__ == "__main__":
    main()
