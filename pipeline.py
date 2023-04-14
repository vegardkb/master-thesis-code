from cropbin.cropbin import main as cropbin
from raw2tif.raw2tif import main as raw2tif
from s2p.s2p_registration import main as s2p_reg
from s2p.s2p_celldetect import main as s2p_celldetect


"""
    Clearly, this is not the way
"""

def main():
    cropbin()
    raw2tif()
    s2p_reg()


if __name__ == "__main__":
    main()