import configparser
import os
import numpy as np
from time import time

SEC_PER_MIN = 60

BITS_PER_BYTE = 8
BITS_PER_GB = BITS_PER_BYTE * 1000 * 1000 * 1000

RAW_FILE_SUFFIX = "_XYT.raw"
CFG_FILE_SUFFIX = "_XYT.ini"

OUT_NAME_CFG = "config.ini"
OUT_NAME_RAW = "timebinned.raw"

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


def incr_frame_counter(counter, n_frames, starting_frame, t0):
    te = int(np.round(time() - t0))
    te_min = te // SEC_PER_MIN
    te_sec = te % SEC_PER_MIN

    frames_read = counter - starting_frame

    tr = (
        int(np.round(te * (n_frames - frames_read) / frames_read))
        if frames_read != 0
        else int(np.round(te * 2 * n_frames))
    )
    tr_min = tr // SEC_PER_MIN
    tr_sec = tr % SEC_PER_MIN

    will_print = False
    """ if counter + 1 == n_frames:
        end_char = "\n"
        will_print = True """

    if counter % 500 == 0:
        end_char = "\r"
        will_print = True

    if will_print:
        print(
            f"Loading frame {frames_read} / {n_frames}, time elapsed: {te_min}:{te_sec}, time remaining: {tr_min}:{tr_sec}",
            end=end_char,
        )

    return counter + 1


def find_cfg_raw(exp_name: str, input_dir):
    try:
        parent_dir = os.path.join(
            input_dir,
            exp_name,
        )

        cfg_fname = exp_name + CFG_FILE_SUFFIX
        raw_fname = exp_name + RAW_FILE_SUFFIX

        cfg_path = os.path.join(
            parent_dir,
            cfg_fname,
        )
        raw_path = os.path.join(
            parent_dir,
            raw_fname,
        )

        return cfg_path, raw_path

    except OSError:
        raise Exception(
            f"Failed to find config or raw file in directory:\n{parent_dir}"
        )


def create_raw_out_path(exp_name: str, crop_id: str, output_dir):
    output_dir = os.path.join(output_dir, exp_name, crop_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    cfg_path = os.path.join(
        output_dir,
        OUT_NAME_CFG,
    )

    raw_path = os.path.join(
        output_dir,
        OUT_NAME_RAW,
    )

    return cfg_path, raw_path


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


class RawReader:
    def __init__(self) -> None:
        """
        Input config
        """
        self.cfg_in = CIConfig()

        """
            Parameters
        """
        self.max_memory_allocate_gb = 0
        self.tb_size = 0
        self.crop_x_start, self.crop_x_stop = 0, 0
        self.crop_y_start, self.crop_y_stop = 0, 0
        self.plot_first_frame = False
        self.exp_name = ""
        self.crop_id = ""
        self.volume_rate = 0 

        """
            Derived parameters
        """
        self.n_batches = 0
        self.n_rest = 0
        self.n_planes = 0
        self.n_ch = 0
        self.f_format = 0
        self.tb_per_batch = 0
        self.batch_size = 0
        self.starting_frame = 0
        self.frame_counter = 0
        self.t0 = 0
        self.dt = 0
        self.bits_per_pix = 0

        """
            Output config
        """
        self.cfg_out = CIConfig()

    def parse_cfg(self, cfg_path):
        """
        Set values from config file (.ini)
        """
        cfg_file = configparser.ConfigParser()
        cfg_file.read(cfg_path)
        hdr = "_"

        self.cfg_in.Lx = int(cfg_file.getfloat(hdr, "x.pixels"))
        self.cfg_in.Ly = int(cfg_file.getfloat(hdr, "y.pixels"))
        self.cfg_in.n_pix = self.cfg_in.Lx * self.cfg_in.Ly
        self.cfg_in.pixel_size = cfg_file.getfloat(hdr, "x.pixel.sz")
        self.cfg_in.n_frames = int(cfg_file.getfloat(hdr, "no..of.frames.to.acquire"))
        self.n_planes = int(cfg_file.getfloat(hdr, "frames.per.z.cycle"))

        self.n_ch = 0
        base_key = "save.ch."
        for i in range(10):
            key = base_key + str(i)
            try:
                value = cfg_file.getboolean(hdr, key)
                if value:
                    self.n_ch = self.n_ch + 1

            except configparser.NoOptionError:
                pass

        self.f_format = int(cfg_file.getfloat(hdr, "file.format"))
        self.cfg_in.volume_rate = cfg_file.getfloat(hdr, "volume.rate.(in.Hz)") if self.volume_rate == 0 else self.volume_rate

        print(
            f"\nInput shape: (Lx, Ly, Lt) = ({self.cfg_in.Lx}, {self.cfg_in.Ly}, {self.cfg_in.n_frames})"
        )
        print(f"n_ch: {self.n_ch}")
        print(f"f_format: {self.f_format}")
        print(f"n_planes: {self.n_planes}")
        print(f"volume_rate: {self.cfg_in.volume_rate} Hz\n")

    def calc_cfg_out_and_derived(self, starting_frame, end_frame):
        self.cfg_out.Lx = self.crop_x_stop - self.crop_x_start
        self.cfg_out.Ly = self.crop_y_stop - self.crop_y_start
        self.cfg_out.n_pix = self.cfg_out.Lx * self.cfg_out.Ly
        self.cfg_out.pixel_size = self.cfg_in.pixel_size
        self.cfg_out.volume_rate = self.cfg_in.volume_rate / self.tb_size

        max_memory_alloc_bits = self.max_memory_allocate_gb * BITS_PER_GB
        const_bits = (
            2 * self.bits_per_pix * self.cfg_out.n_pix * self.tb_size * self.n_planes
        )
        bits_remaining = max_memory_alloc_bits - const_bits
        self.tb_per_batch = bits_remaining // (
            2 * self.bits_per_pix * self.cfg_out.n_pix * self.n_planes
        )
        self.batch_size = self.tb_per_batch * self.tb_size * self.n_planes

        if starting_frame is None:
            if end_frame is None:
                self.starting_frame = self.cfg_in.n_frames % (self.tb_size * self.n_planes)
                n_frames_read = self.cfg_in.n_frames

            else:
                n_frames_read = end_frame
                self.starting_frame = n_frames_read % (self.tb_size * self.n_planes)
        
        else:
            if end_frame is None:
                n_frames_read = self.cfg_in.n_frames - starting_frame
                self.starting_frame = starting_frame + n_frames_read % (self.tb_size * self.n_planes)

            else:
                n_frames_read = end_frame - starting_frame
                self.starting_frame = starting_frame + n_frames_read % (self.tb_size * self.n_planes)


        self.cfg_out.n_frames = n_frames_read // self.tb_size

        self.n_batches = n_frames_read // self.batch_size
        self.n_rest = n_frames_read % self.batch_size

    def get_data_type(self):
        """
        Returns data types used by different file formats.

        NOTE: Hardcoded values should be replaced by enum or const dict declared at top of code file.
        """

        if self.f_format == 1:  # 32 bit raw file
            self.cfg_in.dtype = ">f4"
            self.cfg_out.dtype = "<u2"
            self.bits_per_pix = 32
            raise Exception(f"Format ({self.f_format}) not supported")

        elif self.f_format == 0:  # 16 bit raw file
            self.cfg_in.dtype = ">u2"
            self.cfg_out.dtype = "<u2"
            self.bits_per_pix = 16

        else:
            raise Exception(
                f"{self.exp_name} has unrecognizable file format: {self.f_format}"
            )

    def check_dims(self):
        """
        Checks that the supplied crop parameters are compatible with recording.
        """
        if not (
            self.crop_x_stop < self.cfg_in.Lx and self.crop_y_stop < self.cfg_in.Ly
        ):
            raise Exception(
                f"Invalid CROP_X_STOP ({self.crop_x_stop}) or CROP_Y_STOP ({self.crop_y_stop}) for image stack with (Lx, Ly)=({self.Lx}, {self.Ly})"
            )

        if not (self.crop_x_start >= 0 and self.crop_y_start >= 0):
            raise Exception(
                f"Invalid CROP_X_START ({self.crop_x_start}) or CROP_Y_START ({self.crop_y_start}): Start pixels must be greater than or equal to 0."
            )

        if self.n_ch != 1:
            raise Exception(
                f"Only 1 channel recordings supported. Provided n_ch: {self.n_ch}"
            )

        if not (self.crop_x_start < self.crop_x_stop):
            raise Exception(
                f"Invalid CROP_X_START ({self.crop_x_start}) and CROP_X_STOP ({self.crop_x_stop}): CROP_X_START must be smaller than CROP_X_STOP."
            )

        if not (self.crop_y_start < self.crop_y_stop):
            raise Exception(
                f"Invalid CROP_Y_START ({self.crop_y_start}) and CROP_Y_STOP ({self.crop_y_stop}): CROP_Y_START must be smaller than CROP_Y_STOP."
            )

    def init_arrays(self):
        in_data = np.zeros(
            (self.cfg_in.n_pix, self.tb_size, self.n_planes), dtype=self.cfg_in.dtype
        )
        tmp1 = np.zeros(
            (self.cfg_in.Ly, self.cfg_in.Lx, self.tb_size, self.n_planes),
            dtype=self.cfg_in.dtype,
        )
        tmp2 = np.zeros(
            (self.cfg_out.Ly, self.cfg_out.Lx, self.tb_per_batch, self.n_planes),
            dtype=self.cfg_out.dtype,
        )
        out_data = np.zeros(
            (self.cfg_out.n_pix, self.tb_per_batch * self.n_planes),
            dtype=self.cfg_out.dtype,
        )

        return in_data, tmp1, tmp2, out_data

    def save_cfg(self, cfg_path):
        """
        Saves config variables in .ini file.
        """
        cfg = configparser.ConfigParser()
        hdr = "_"

        cfg[hdr] = {
            "Lx": self.cfg_out.Lx,
            "Ly": self.cfg_out.Ly,
            "pixel_size": self.cfg_out.pixel_size,
            "n_frames": self.cfg_out.n_frames,
            "n_ch": self.n_ch,
            "n_planes": self.n_planes,
            "f_format": self.f_format,
            "volume_rate": self.cfg_out.volume_rate,
            "starting_frame": self.starting_frame // self.tb_size,
        }
        with open(cfg_path, "w") as f:
            cfg.write(f)

    def read_skip_frames(self, raw_in):
        while self.frame_counter < self.starting_frame:
            _ = np.fromfile(
                raw_in, dtype=self.cfg_in.dtype, count=self.cfg_in.n_pix
            )
            self.frame_counter += 1

    def read_time_bin(self, raw_in, in_data: np.ndarray):
        for j in range(self.tb_size):
            for k in range(self.n_planes):
                in_data[:, j, k] = np.fromfile(
                    raw_in, dtype=self.cfg_in.dtype, count=self.cfg_in.n_pix
                )
                self.frame_counter = incr_frame_counter(
                    self.frame_counter, self.cfg_out.n_frames * self.tb_size, self.starting_frame, self.t0
                )

        return in_data
