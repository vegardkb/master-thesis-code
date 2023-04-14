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
NAME_ROIS = "rois"

WORKING_DIR = os.path.join("..", "..", "..", "data")
S2P_DIR = os.path.join("suite2p", "plane0")
DENOISED_DIR = "denoised"
DENOISED_CHAN2_DIR = "denoised_chan2"

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

    def parse_cfg(self, cfg_path, format):
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

        print(f"\nInput shape: (Lx, Ly, Lt) = ({self.Lx}, {self.Ly}, {self.n_frames})")
        print(f"volume_rate: {self.volume_rate} Hz\n")

    def parse_proc_cfg(self, cfg_path):
        self.parse_cfg(cfg_path, "processed")


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

    bin_file_names = ["data_tempfilt.bin", "data.bin"]
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


def load_rois(s2p_df, exp_dir) -> List[Any]:
    print("Attempting to retrieve custom rois...")
    custom_rois = load_custom_rois(exp_dir)
    if custom_rois is not None:
        return custom_rois

    print(f"Reading rois from suite2p output...")
    s2p_rois = load_s2p_rois(s2p_df)
    return s2p_rois


def load_s2p_df_rois(exp_dir, cfg: CIConfig) -> Tuple[Any, Any]:
    s2p_df = load_s2p_df(exp_dir, cfg.Ly, cfg.Lx)
    rois = load_rois(s2p_df, exp_dir)
    return s2p_df, rois


def load_cfg(exp_dir) -> CIConfig:
    cfg_path = os.path.join(exp_dir, OUT_NAME_CFG)
    cfg = CIConfig()
    cfg.parse_proc_cfg(cfg_path)
    return cfg


def generate_roi_path(exp_dir):
    roi_path = os.path.join(
        exp_dir,
        NAME_ROIS + ".npy",
    )
    return roi_path


def load_custom_rois(exp_dir) -> List[Any]:
    roi_path = generate_roi_path(exp_dir)

    try:
        rois = []
        rois_np = np.load(roi_path, allow_pickle=True)
        for roi in rois_np:
            rois.append(roi)

        print(f"Rois loaded from {roi_path}")
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


def get_t(fs, n_frames, n_t_start_discard=0):
    n_frames = n_frames - n_t_start_discard
    return np.linspace(0, (n_frames - 1) / fs, n_frames)


def save_rois(rois, exp_dir, name_rois):
    roi_path = gen_npy_fname(exp_dir, name_rois)
    np.save(roi_path, rois, allow_pickle=True)
