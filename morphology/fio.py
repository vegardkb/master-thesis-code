import configparser
import os
from typing import List, Any
import numpy as np
from pickle import UnpicklingError

"""
    File I/O helper functions
"""

OUT_NAME_CFG = "config.ini"
NAME_ROIS = "rois.npy"

WORKING_DIR = os.path.join("..", "..", "..", "data")
DENOISED_DIR = "denoised"
DENOISED_CHAN2_DIR = "denoised_chan2"
RESULTS_DIR = "results"
PROTOCOL_DIR = os.path.join("..", "..", "..", "raw_data", "protocols")
ROIS_DIR = "ROI"
EVTS_DIR = "events"
DFF_DIR = "dff"
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


def generate_global_results_dir():
    results_dir = os.path.join(WORKING_DIR, RESULTS_DIR)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    return results_dir


def generate_figures_dir():
    figures_dir = os.path.join(WORKING_DIR, FIGURES_DIR)
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir)

    return figures_dir


def generate_processed_cfg_path(exp_dir):
    cfg_path = os.path.join(exp_dir, OUT_NAME_CFG)
    return cfg_path


def load_cfg(exp_dir, verbose=False) -> CIConfig:
    cfg_path = generate_processed_cfg_path(exp_dir)
    cfg = CIConfig()
    cfg.parse_proc_cfg(cfg_path, verbose)
    return cfg


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


def gen_pickle_fname(output_dir, fname):
    return os.path.join(output_dir, fname + ".pkl")


def gen_npy_fname(output_dir, fname):
    return os.path.join(output_dir, fname + ".npy")
