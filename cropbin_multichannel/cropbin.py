import numpy as np
import time
import threading
import configparser
import os

from cropbin_util import find_cfg_raw, create_raw_out_path, RawReader

"""
    cropbin.py is a script that performs cropping and timebinning of raw file(s), way faster than fiji.
    Resources are efficiently utilized when files are processed in batch due to multithreading.

    Usage:
        - Specify INPUT_DIR and OUTPUT_DIR
        - Specify which experiments and crops to load by changing EXP_NAMES and CROP_IDS
            - These should be described in cropbin.ini
        - Modify TIME_BIN_SIZE to suit your needs
        - Tested with single-plane, single-channel and single-plane dual-channel.
    
    Caution:
        - Not sure if multiplane recordings are supported.
        - Output raw file uses little endian bit order.

    Possible future features:
        - Convert output to (OME-)tiff for easier transition into suite2p processing.
        - Make parameters less awkward.
"""


"""
    Parameters
"""
INPUT_DIR = "Y:/Sunniva/VR/2022_05_31"
OUTPUT_DIR = os.path.join("..", "..", "processed")

CFG_FNAME = "cropbin.ini"

MAX_MEMORY_ALLOCATE_GB = 1
TIME_BIN_SIZE = 5
PLOT_FIRST_FRAME = False


""" EXP_NAMES = [
    "20211112_13_23_43_GFAP_GCamp6s_F2_PTZ",
    "20220211_13_18_56_GFAP_GCamp6s_F2_C",
    "20220211_15_02_16_GFAP_GCamp6s_F3_PTZ",
    "20220412_13_59_55_GFAP_GCamp6s_F3_C",
    "20220531_12_34_15_HuC_GCamp6s_GFAP_jRGECO_F1_C",
    "20220531_14_21_02_HuC_GCamp6s_GFAP_jRGECO_F2_PTZ",
] """
EXP_NAMES = [
    "20220531_12_34_15_HuC_GCamp6s_GFAP_jRGECO_F1_C",
    "20220531_14_21_02_HuC_GCamp6s_GFAP_jRGECO_F2_PTZ",
]
CROP_IDS = ["OpticTectum"]

CROP_SIZES = {
    "OpticTectum": {
        "lx": 350,
        "ly": 500,
    },
}

def read_config(fname, exp_names, crop_ids):
    cfg = configparser.ConfigParser()
    cfg.read(fname)

    rawr_list = []
    for sec in cfg.sections():
        exp_name = cfg[sec]["exp_name"]
        crop_id = cfg[sec]["crop_id"]
        if exp_name not in exp_names or crop_id not in crop_ids:
            print(f"{cfg[sec]['exp_name']} not in exp_names or {cfg[sec]['crop_id']} not in crop_ids")
            continue

        rawr = RawReader()
        load_parameters(rawr)

        crop_lx, crop_ly = CROP_SIZES[crop_id]["lx"], CROP_SIZES[crop_id]["ly"]

        crop_x_start, crop_x_stop = (
            cfg.getint(sec, "crop_x_start"),
            cfg.getint(sec, "crop_x_stop"),
        )
        crop_y_start, crop_y_stop = (
            cfg.getint(sec, "crop_y_start"),
            cfg.getint(sec, "crop_y_stop"),
        )

        crop_x_mid = int((crop_x_start + crop_x_stop) / 2)
        crop_y_mid = int((crop_y_start + crop_y_stop) / 2)

        crop_x_start, crop_x_stop = crop_x_mid - crop_lx // 2, crop_x_mid + crop_lx // 2
        crop_y_start, crop_y_stop = crop_y_mid - crop_ly // 2, crop_y_mid + crop_ly // 2

        rawr.crop_x_start, rawr.crop_x_stop = (crop_x_start, crop_x_stop)
        rawr.crop_y_start, rawr.crop_y_stop = (crop_y_start, crop_y_stop)

        rawr.exp_name, rawr.crop_id = exp_name, crop_id
        if cfg.has_option(sec, "volume_rate"):
            rawr.volume_rate = cfg.getfloat(sec, "volume_rate")
        rawr_list.append(rawr)

    return rawr_list


def load_parameters(rawr: RawReader):
    rawr.max_memory_allocate_gb = MAX_MEMORY_ALLOCATE_GB
    rawr.tb_size = TIME_BIN_SIZE
    rawr.plot_first_frame = PLOT_FIRST_FRAME


def rw_loop(raw_path_in, raw_path_out, rawr: RawReader):
    with open(raw_path_out, "wb") as raw_out, open(raw_path_in, "rb") as raw_in:
        rawr.t0 = time.time()

        in_data, tmp1, tmp2, out_data = rawr.init_arrays()
        # Could possibly get speed increase by taking mean on entire bin, memory usage would go way up though
        for _ in range(rawr.n_batches):
            for i in range(rawr.tb_per_batch):
                in_data = rawr.read_time_bin(raw_in, in_data)

                tmp1 = np.reshape(
                    in_data,
                    (rawr.cfg_in.Ly, rawr.cfg_in.Lx, rawr.tb_size, rawr.n_planes, rawr.n_ch),
                )
                tmp2[:, :, i] = np.mean(
                    tmp1[
                        rawr.crop_y_start : rawr.crop_y_stop,
                        rawr.crop_x_start : rawr.crop_x_stop,
                    ],
                    axis=2,
                )

                """ if PLOT_FIRST_FRAME and rawr.frame_counter == rawr.starting_frame:
                    # Not plotting when PLOT_FIRST_FRAME==True
                    plot_each_plane(tmp2, rawr.n_planes, i)
                    plt.show() """

            out_data = np.reshape(tmp2, (-1, rawr.tb_per_batch * rawr.n_planes * rawr.n_ch))
            raw_out.write(out_data.tobytes(order="F"))

        # Lots of duplicate code here
        num_time_bins = rawr.n_rest // (rawr.tb_size * rawr.n_planes * rawr.n_ch)
        for i in range(num_time_bins):
            in_data = rawr.read_time_bin(raw_in, in_data)

            tmp1 = np.reshape(
                in_data,
                (rawr.cfg_in.Ly, rawr.cfg_in.Lx, rawr.tb_size, rawr.n_planes, rawr.n_ch),
            )
            tmp2[:, :, i] = np.mean(
                tmp1[
                    rawr.crop_y_start : rawr.crop_y_stop,
                    rawr.crop_x_start : rawr.crop_x_stop,
                ],
                axis=2,
            )

        out_data = np.reshape(
            tmp2[:, :, : num_time_bins],
            (-1, num_time_bins * rawr.n_planes * rawr.n_ch),
        )
        raw_out.write(out_data.tobytes(order="F"))

        print("")


def cropbin(rawr: RawReader):
    cfg_path_in, raw_path_in = find_cfg_raw(rawr.exp_name, INPUT_DIR)

    rawr.parse_cfg(cfg_path_in)

    rawr.get_data_type()

    rawr.check_dims()

    rawr.calc_cfg_out_and_derived()

    cfg_path_out, raw_path_out = create_raw_out_path(
        rawr.exp_name, rawr.crop_id, OUTPUT_DIR
    )

    rawr.save_cfg(cfg_path_out)

    print(f"\nbatch_size = {rawr.batch_size}")
    print(f"num batches = {rawr.n_batches}")
    print(f"num frames remaining = {rawr.n_rest}")
    print(
        f"Output shape: (Lx, Ly, Lt) = ({rawr.cfg_out.Lx}, {rawr.cfg_out.Ly}, {rawr.cfg_out.n_frames})\n"
    )
    rw_loop(raw_path_in, raw_path_out, rawr)


def main():
    rawr_list = read_config(CFG_FNAME, EXP_NAMES, CROP_IDS)

    threads = []

    for rawr in rawr_list:
        print(rawr.exp_name)
        t = threading.Thread(target=cropbin, args=(rawr,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
