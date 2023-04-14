import os
import numpy as np
from pyometiff import OMETIFFWriter, OMETIFFReader
from tqdm import tqdm
import threading

from raw2tif_util import RawConfig

"""
    Converts .raw files to .ome.tiff files, to be run after cropbin.py
    Usage:
        - Set CFG_FNAME and RAW_FNAME
        - Set CHANNELS_DICT

    Todo:
        - cropbin should get the channel dict info somehow and save in config.

"""

"""
    Parameters
"""
WORKING_DIR = os.path.join("..", "..", "processed")

CFG_FNAME = "config.ini"
RAW_FNAME = "timebinned.raw"
OMETIFF_FNAME = "timebinned.ome.tiff"

DIM_ORDER = "ZTCYX"
DTYPE_RAW = "<u2"


EXP_NAMES = [
    "20211112_13_23_43_GFAP_GCamp6s_F2_PTZ",
    "20211112_18_30_27_GFAP_GCamp6s_F5_c2",
    "20211112_19_48_54_GFAP_GCamp6s_F6_c3",
    #"20211117_14_17_58_GFAP_GCamp6s_F2_C",
    #"20211117_17_33_00_GFAP_GCamp6s_F4_PTZ",
    "20211117_21_31_08_GFAP_GCamp6s_F6_PTZ",
    "20211119_16_36_20_GFAP_GCamp6s_F4_PTZ",
    #"20211119_18_15_06_GFAP_GCamp6s_F5_C",
    #"20211119_21_52_35_GFAP_GCamp6s_F7_C",
    "20220211_13_18_56_GFAP_GCamp6s_F2_C",
    #"20220211_15_02_16_GFAP_GCamp6s_F3_PTZ",
    "20220211_16_51_15_GFAP_GCamp6s_F4_PTZ",
    #"20220412_10_43_04_GFAP_GCamp6s_F1_PTZ",
    "20220412_12_32_27_GFAP_GCamp6s_F2_PTZ",
    "20220412_13_59_55_GFAP_GCamp6s_F3_C",
    #"20220412_16_06_54_GFAP_GCamp6s_F4_PTZ",
]

CROP_IDS = ["OpticTectum"]

""" CHANNELS_DICT = {
    "x" : {
        "Name" : "neuron_xnm",
        "SamplesPerPixel": 1,
        "ExcitationWavelength": 400.,
        "ExcitationWavelengthUnit": "nm"
    },
    "y" : {
        "Name" : "glia_ynm",
        "SamplesPerPixel": 1,
        "ExcitationWavelength": 800.,
        "ExcitationWavelengthUnit": "nm"
    },
} """

CHANNELS_DICT = {
    "y" : {
        "Name" : "glia_ynm",
        "SamplesPerPixel": 1,
        "ExcitationWavelength": 800.,
        "ExcitationWavelengthUnit": "nm"
    },
}


def raw2tif(exp_name, crop_id):
    exp_dir = os.path.join(WORKING_DIR, exp_name, crop_id)
    cfg_path = os.path.join(exp_dir, CFG_FNAME)
    raw_path = os.path.join(exp_dir, RAW_FNAME)
    ometiff_path = os.path.join(exp_dir, OMETIFF_FNAME)

    cfg = RawConfig()
    cfg.parse_cfg(cfg_path)

    nz = cfg.n_planes
    nt = cfg.n_frames // (cfg.n_ch * cfg.n_planes)
    nc = cfg.n_ch
    ny = cfg.Ly
    nx = cfg.Lx
    data = np.zeros((nz, nt, nc, ny, nx), dtype=np.uint16)

    metadata_dict = OMETIFFReader._get_metadata_template()

    # Adding channel_dict to avoid bug
    # Fix this somehow
    metadata_dict["Channels"] = CHANNELS_DICT

    """
        Possibly nice to add pixel size in x-y-z directions
    """
    unit = "um"
    metadata_dict["PhysicalSizeX"] = cfg.pixel_size
    metadata_dict["PhysicalSizeXUnit"] = unit
    metadata_dict["PhysicalSizeY"] = cfg.pixel_size
    metadata_dict["PhysicalSizeYUnit"] = unit

    n_pix = nx * ny

    ometiff_path = os.path.join(exp_dir, OMETIFF_FNAME)

    with open(raw_path, "r") as f_raw:
        for t in tqdm(range(nt), desc="frame", leave=False):
            for z in range(nz):
                for c in range(nc):
                    tmp = np.fromfile(
                        f_raw, dtype=DTYPE_RAW, count=n_pix
                    )
                    data[z,t,c,:,:] = np.reshape(tmp, newshape=(ny,nx))



    writer = OMETIFFWriter(
        fpath=ometiff_path,
        dimension_order=DIM_ORDER,
        array=data,
        metadata=metadata_dict,
    )
    writer.write()


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