import configparser
import os
from typing import Tuple, List, Any
import numpy as np
from pickle import UnpicklingError
import h5py
import suite2p

"""
    File I/O helper functions
"""

OUT_NAME_CFG = "config.ini"

WORKING_DIR = os.path.join("..", "..", "..", "data")
DENOISED_DIR = "denoised"
DENOISED_CHAN2_DIR = "denoised_chan2"
RESULTS_DIR = "results"
PROTOCOL_DIR = os.path.join("..", "..", "..", "raw_data", "protocols")
ROIS_DIR = "ROI"
EVTS_DIR = "events"
DFF_DIR = "dff"
S2P_DIR = os.path.join("suite2p", "plane0")
FIGURES_DIR = "figures"

CFG_FORMAT = {
    "raw": {
        "Lx": "x.pixels",
        "Ly": "y.pixels",
        "pixel_size": "x.pixel.sz",
        "n_frames": "no..of.frames.to.acquire",
        "volume_rate": "volume.rate.(in.Hz)",
    },
    "processed": {
        "Lx": "lx",
        "Ly": "ly",
        "pixel_size": "pixel_size",
        "n_frames": "n_frames",
        "volume_rate": "volume_rate",
    },
}


class CIConfig:
    def __init__(self) -> None:
        self.Lx = 0
        self.Ly = 0
        self.n_pix = 0
        self.pixel_size = 0
        self.n_frames = 0
        self.volume_rate = 0
        self.dtype = ""

    def parse_cfg(self, cfg_path, format, verbose=False):
        """
        Set values from config file (.ini)
        """
        cfg_file = configparser.ConfigParser()
        cfg_file.read(cfg_path)
        hdr = "_"

        fmt_dict = CFG_FORMAT[format]

        self.Lx = int(cfg_file.getfloat(hdr, fmt_dict["Lx"]))
        self.Ly = int(cfg_file.getfloat(hdr, fmt_dict["Ly"]))
        self.n_pix = self.Lx * self.Ly
        self.pixel_size = cfg_file.getfloat(hdr, fmt_dict["pixel_size"])
        self.n_frames = int(cfg_file.getfloat(hdr, fmt_dict["n_frames"]))
        self.volume_rate = cfg_file.getfloat(hdr, fmt_dict["volume_rate"])

        if verbose:
            print(
                f"\nInput shape: (Lx, Ly, Lt) = ({self.Lx}, {self.Ly}, {self.n_frames})"
            )
            print(f"volume_rate: {self.volume_rate} Hz\n")

    def parse_proc_cfg(self, cfg_path, verbose=False):
        self.parse_cfg(cfg_path, "processed", verbose)


def generate_exp_dir(exp_name: str, crop_id: str):
    output_dir = os.path.join(WORKING_DIR, exp_name, crop_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    return output_dir


def generate_denoised_dir(exp_dir, use_chan2):
    denoised_partial_dir = DENOISED_CHAN2_DIR if use_chan2 else DENOISED_DIR
    denoised_dir = os.path.join(
        exp_dir,
        denoised_partial_dir,
    )
    if not os.path.isdir(denoised_dir):
        raise Exception(f"denoised directory does not exist:\n{denoised_dir}")

    return denoised_dir


def generate_results_dir(exp_dir):
    results_dir = os.path.join(exp_dir, RESULTS_DIR)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    return results_dir


def generate_results_roi_dir(exp_dir, roi_number: str):
    results_dir = generate_results_dir(exp_dir)
    rois_dir = os.path.join(results_dir, ROIS_DIR + roi_number)
    if not os.path.isdir(rois_dir):
        os.makedirs(rois_dir)

    return rois_dir


def generate_roi_dff_dir(exp_dir, roi_number: str, roi_fname):
    rois_dir = generate_results_roi_dir(exp_dir, roi_number)
    dff_dir = os.path.join(rois_dir, DFF_DIR, roi_fname)
    if not os.path.isdir(dff_dir):
        os.makedirs(dff_dir)

    return dff_dir


def generate_processed_cfg_path(exp_dir):
    cfg_path = os.path.join(exp_dir, OUT_NAME_CFG)
    return cfg_path


def generate_s2p_dir(exp_dir):
    s2p_dir = os.path.join(
        exp_dir,
        S2P_DIR,
    )
    if not os.path.isdir(s2p_dir):
        raise Exception(f"suite2p directory does not exist:\n{s2p_dir}")

    return s2p_dir


def load_s2p_rois(s2p_df) -> List[Any]:
    iscell = s2p_df["iscell"]
    mask_shape = (s2p_df["x"].shape[1], s2p_df["x"].shape[2])
    rois = []

    n_roi = iscell.shape[0]

    c_mask = np.zeros(mask_shape, dtype=bool)

    for i_roi in range(n_roi):
        c_iscell = iscell[i_roi]
        if c_iscell[0] > 0.5:
            c_stats = s2p_df["stats"][i_roi]
            for (i, j) in list(zip(c_stats["xpix"], c_stats["ypix"])):
                c_mask[j, i] = True

            rois.append(c_mask)

            c_mask = np.zeros(mask_shape, dtype=bool)

    print("Rois loaded from iscell.npy")

    return rois


def load_s2p_bonus_stuff(s2p_df, s2p_dir):
    s2p_df["F"] = np.load(os.path.join(s2p_dir, "F.npy"), allow_pickle=True)
    s2p_df["Fneu"] = np.load(os.path.join(s2p_dir, "Fneu.npy"), allow_pickle=True)
    s2p_df["spks"] = np.load(os.path.join(s2p_dir, "spks.npy"), allow_pickle=True)
    s2p_df["stats"] = np.load(os.path.join(s2p_dir, "stat.npy"), allow_pickle=True)
    ops = np.load(os.path.join(s2p_dir, "ops.npy"), allow_pickle=True)
    s2p_df["ops"] = ops.item()
    s2p_df["iscell"] = np.load(os.path.join(s2p_dir, "iscell.npy"), allow_pickle=True)

    return s2p_df


def load_s2p_df(exp_dir, Ly: int, Lx: int):
    s2p_dir = generate_s2p_dir(exp_dir)

    bin_file_names = ["data.bin", "data_tempfilt.bin"]
    loaded = False

    s2p_df = {}
    for bin_fname in bin_file_names:
        try:
            f_path = os.path.join(s2p_dir, bin_fname)
            s2p_df["x"] = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, read_filename=f_path).data
            s2p_df = load_s2p_bonus_stuff(s2p_df, s2p_dir)

            print(f"Loaded data from:\n{f_path}")
            loaded = True
            break

        except FileNotFoundError as e:
            print(e)
            continue

    if not loaded:
        raise Exception(f"Binary file not found in {s2p_dir}")

    print(f"shape (nframes, Ly, Lx): {s2p_df['x'].shape}")

    return s2p_df


def load_rois(s2p_df, exp_dir, rois_fname) -> List[Any]:
    # print("Attempting to retrieve custom rois...")
    custom_rois = load_custom_rois(exp_dir, rois_fname)
    if custom_rois is not None:
        return custom_rois

    # print(f"Reading rois from suite2p output...")
    s2p_rois = load_s2p_rois(s2p_df)
    return s2p_rois


def load_s2p_df_rois(exp_dir, cfg: CIConfig, rois_fname) -> Tuple[Any, Any]:
    s2p_df = load_s2p_df(exp_dir, cfg.Ly, cfg.Lx)
    rois = load_rois(s2p_df, exp_dir, rois_fname)
    return s2p_df, rois


def load_cfg(exp_dir, verbose=False) -> CIConfig:
    cfg_path = generate_processed_cfg_path(exp_dir)
    cfg = CIConfig()
    cfg.parse_proc_cfg(cfg_path, verbose)
    return cfg


def load_protocol(protocol_name):
    protocol = {}
    in_fname = os.path.join(PROTOCOL_DIR, protocol_name, protocol_name + ".mat")
    prot_mat = h5py.File(in_fname, "r")

    protocol["t_onset_light"] = np.squeeze(prot_mat["PTZresults"]["timeONSET"])

    return protocol


def load_custom_rois(exp_dir, rois_fname) -> List[Any]:
    roi_path = gen_npy_fname(exp_dir, rois_fname)

    try:
        rois = []
        rois_np = np.load(roi_path, allow_pickle=True)
        for roi in rois_np:
            rois.append(roi)

        # print(f"Rois loaded from {roi_path}")
        return rois

    except OSError:
        print(f"Roi file does not exist or cannot be read:\n{roi_path}")
    except UnpicklingError:
        print(f"File cannot be loaded as pickle:\n{roi_path}")
    except ValueError:
        print(f"File contains object array, but allow_pickle=False")

    return None


def gen_npy_fname(output_dir, fname):
    return os.path.join(output_dir, fname + ".npy")
