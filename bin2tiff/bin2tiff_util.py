import os
import configparser

CFG_FNAME = "config.ini"
OUT_NAME_RAW = "timebinned.raw"

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
        "n_ch": "n_ch",
        "n_planes": "n_planes",
        "volume_rate": "volume_rate",
    },
}


def write_cfg_to_ops(ops, cfg):
    ops["nplanes"] = cfg.n_planes
    ops["nchannels"] = cfg.n_ch
    ops["fs"] = cfg.volume_rate
    return ops


def create_db(input_dir, output_dir):
    db = {}
    db["data_path"] = [input_dir]
    db["save_folder"] = output_dir
    return db


def get_s2p_dir(exp_dir):
    return os.path.join(exp_dir, "suite2p")


class CIConfig:
    def __init__(self) -> None:
        self.Lx = 0
        self.Ly = 0
        self.n_pix = 0
        self.pixel_size = 0
        self.n_frames = 0
        self.n_ch = 0
        self.n_planes = 0
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
        self.n_ch = int(cfg_file.getfloat(hdr, fmt_dict["n_ch"]))
        self.n_planes = int(cfg_file.getfloat(hdr, fmt_dict["n_planes"]))
        self.volume_rate = cfg_file.getfloat(hdr, fmt_dict["volume_rate"])

        print(f"\nInput shape: (Lx, Ly, Lt) = ({self.Lx}, {self.Ly}, {self.n_frames})")
        print(f"volume_rate: {self.volume_rate} Hz\n")

    def parse_proc_cfg(self, cfg_path):
        self.parse_cfg(cfg_path, "processed")


def generate_exp_dir(input_dir, exp_name, crop_id):
    exp_dir = os.path.join(input_dir, exp_name, crop_id)
    return exp_dir


def generate_denoised_dir(exp_dir, use_chan2):
    denoised_partial_dir = DENOISED_CHAN2_DIR if use_chan2 else DENOISED_DIR
    denoised_dir = os.path.join(
        exp_dir,
        denoised_partial_dir,
    )
    if not os.path.isdir(denoised_dir):
        raise Exception(f"denoised directory does not exist:\n{denoised_dir}")

    return denoised_dir


def load_cfg(exp_dir) -> CIConfig:
    cfg_path = os.path.join(exp_dir, CFG_FNAME)
    cfg = CIConfig()
    cfg.parse_proc_cfg(cfg_path)
    return cfg
