import os
import configparser
import h5py
from suite2p.io.binary import BinaryRWFile

CFG_FNAME = "config.ini"
OUT_NAME_RAW = "timebinned.raw"

H5PY_KEY = "data"

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


def create_db_denoised(input_dir, denoised_fname):
    db = {}
    db["h5py"] = os.path.join(input_dir, denoised_fname)
    db["h5py_key"] = H5PY_KEY
    db["data_path"] = ""
    return db

def create_db_normal(input_dir):
    db = {}
    db["data_path"] = [input_dir]
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


def load_cfg(exp_dir) -> CIConfig:
    cfg_path = os.path.join(exp_dir, CFG_FNAME)
    cfg = CIConfig()
    cfg.parse_proc_cfg(cfg_path)
    return cfg

def crop_h5(input_dir, cfg: CIConfig, denoised_fname: str):
    h5_path = os.path.join(input_dir, denoised_fname)
    crop_h5_path = os.path.join(input_dir, "cropped_"+denoised_fname)
    did_crop = False
    with h5py.File(h5_path, "r") as h5:
        if not (h5[H5PY_KEY].shape[2] < cfg.Lx or h5[H5PY_KEY].shape[1] < cfg.Ly):
            if h5[H5PY_KEY].shape[2] > cfg.Lx or h5[H5PY_KEY].shape[1] > cfg.Ly:
                with h5py.File(crop_h5_path, "w") as croph5:
                    croph5[H5PY_KEY] = h5[H5PY_KEY][:, :cfg.Ly, :cfg.Lx]
                    print(f"Cropped {h5_path} to shape (x,y) = ({cfg.Lx}, {cfg.Ly})")
                    did_crop = True
    
    if did_crop:
        os.remove(h5_path)
        os.rename(crop_h5_path, h5_path)


def h5_to_bin(input_dir, cfg: CIConfig, denoised_fname: str):
    h5_path = os.path.join(input_dir, denoised_fname)
    bin_dir = os.path.join(input_dir, "suite2p", "plane0")
    os.makedirs(bin_dir)
    bin_path = os.path.join(bin_dir, "data.bin")

    if cfg.n_planes > 1:
        print("Multiplane .h5 files not supported")

    if cfg.n_ch > 1:
        print("Multichannel .h5 files not supported")

    binary_rw = BinaryRWFile(cfg.Ly, cfg.Lx, bin_path)
    if binary_rw.n_frames < 20:
        with h5py.File(h5_path, "r") as h5:
            binary_rw.write(h5[H5PY_KEY])
            print(f"Wrote {h5_path} to {bin_path}")


def process_h5(input_dir, cfg, denoised_fname: str):
    bin_dir = os.path.join(input_dir, "suite2p", "plane0")
    if os.path.isfile(os.path.join(bin_dir, "data.bin")):
        print(f"Skipping process_h5: data.bin already exists")
        return True

    if not os.path.isfile(os.path.join(input_dir, denoised_fname)):
        print(f"Skipping experiment: {denoised_fname} does not exist")
        return False

    crop_h5(input_dir, cfg, denoised_fname)
    h5_to_bin(input_dir, cfg, denoised_fname)
    return True