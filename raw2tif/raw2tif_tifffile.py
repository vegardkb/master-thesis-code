import os
import numpy as np
import tifffile
import zarr
from tqdm import tqdm
import threading

from raw2tif_util import RawConfig

"""
    Converts .raw files to .ome.tiff files, to be run after cropbin.py
    Usage:
        - Set CFG_FNAME and RAW_FNAME
"""

"""
    Parameters
"""
WORKING_DIR = os.path.join("..", "..", "processed")

CFG_FNAME = "config.ini"
RAW_FNAME = "timebinned.raw"
OMETIFF_FNAME = "timebinned.ome.tiff"

DIM_ORDER = "TZCYX"
DTYPE_RAW = "<u2"

""" EXP_NAMES = [
    "20211112_13_23_43_GFAP_GCamp6s_F2_PTZ",
    "20220211_13_18_56_GFAP_GCamp6s_F2_C",
    "20220211_15_02_16_GFAP_GCamp6s_F3_PTZ",
    "20220412_13_59_55_GFAP_GCamp6s_F3_C",
] """
EXP_NAMES = [
    "20211112_13_23_43_GFAP_GCamp6s_F2_PTZ"
]

CROP_IDS = ["OpticTectum"]


def raw2tif(exp_name, crop_id):
    exp_dir = os.path.join(WORKING_DIR, exp_name, crop_id)
    cfg_path = os.path.join(exp_dir, CFG_FNAME)
    raw_path = os.path.join(exp_dir, RAW_FNAME)
    ometiff_path = os.path.join(exp_dir, OMETIFF_FNAME)

    cfg = RawConfig()
    cfg.parse_cfg(cfg_path)

    nt = cfg.n_frames
    nz = cfg.n_planes
    nc = cfg.n_ch
    ny = cfg.Ly
    nx = cfg.Lx
    data = np.zeros((nt, nz, nc, ny, nx))
    metadata_dict = {}


    """
        Possibly nice to add pixel size in x-y-z directions
    """
    unit = "um"
    metadata_dict["PhysicalSizeX"] = cfg.pixel_size
    metadata_dict["PhysicalSizeXUnit"] = unit
    metadata_dict["PhysicalSizeY"] = cfg.pixel_size
    metadata_dict["PhysicalSizeYUnit"] = unit

    metadata_dict["axes"] = DIM_ORDER

    n_pix = nx * ny

    with open(raw_path, "r") as f_raw:
        for t in tqdm(range(nt), desc="frame", leave=False):
            for z in range(nz):
                for c in range(nc):
                    tmp = np.fromfile(
                        f_raw, dtype=DTYPE_RAW, count=n_pix
                    )
                    data[t,z,c,:,:] = np.reshape(tmp, newshape=(ny,nx))




    tifffile.imwrite(
        ometiff_path,
        shape=data.shape,
        dtype="uint16",
        photometric='minisblack',
        metadata=metadata_dict,
        ome=True
    )

    store = tifffile.imread(ometiff_path, mode="r+", aszarr=True)
    z = zarr.open(store, mode="r+")
    z = data
    store.close()


def main():
    threads = []

    for exp_name in tqdm(EXP_NAMES, desc="experiment"):
        for crop_id in tqdm(CROP_IDS, desc="crop_id", leave=False):
            t = threading.Thread(target=raw2tif, args=(exp_name, crop_id,))
            threads.append(t)
            t.start()

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()