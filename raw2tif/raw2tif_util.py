import configparser


RAW_DTYPE = "<u2"


class RawConfig:
    def __init__(self) -> None:
        self.Lx = 0
        self.Ly = 0
        self.n_pix = 0
        self.pixel_size = 0
        self.n_frames = 0
        self.n_ch = 0
        self.n_planes = 0
        self.volume_rate = 0
        self.dtype = RAW_DTYPE

    def parse_cfg(self, cfg_path):
        """
        Set values from config file (.ini)
        """
        cfg_file = configparser.ConfigParser()
        cfg_file.read(cfg_path)
        hdr = "_"

        self.Lx = int(cfg_file.getfloat(hdr, "lx"))
        self.Ly = int(cfg_file.getfloat(hdr, "ly"))
        self.n_pix = self.Lx * self.Ly
        self.pixel_size = cfg_file.getfloat(hdr, "pixel_size")
        self.n_frames = int(cfg_file.getfloat(hdr, "n_frames"))
        self.n_ch = int(cfg_file.getfloat(hdr, "n_ch"))
        self.n_planes = int(cfg_file.getfloat(hdr, "n_planes"))
        self.volume_rate = cfg_file.getfloat(hdr, "volume_rate")
        _ = int(cfg_file.getfloat(hdr, "f_format"))

        print(f"\nInput shape: (Lx, Ly, Lt) = ({self.Lx}, {self.Ly}, {self.n_frames})")
        print(f"volume_rate: {self.volume_rate} Hz\n")
