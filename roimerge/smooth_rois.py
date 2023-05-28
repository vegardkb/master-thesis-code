import numpy as np
from scipy.ndimage import gaussian_filter
from threading import Thread

from fio import (
    load_custom_rois,
    generate_exp_dir,
    generate_denoised_dir,
    save_rois,
)

EXP_NAMES = [
    "20211112_13_23_43_GFAP_GCamp6s_F2_PTZ",
    "20211112_18_30_27_GFAP_GCamp6s_F5_c2",
    "20211112_19_48_54_GFAP_GCamp6s_F6_c3",
    # "20211117_14_17_58_GFAP_GCamp6s_F2_C",
    # "20211117_17_33_00_GFAP_GCamp6s_F4_PTZ",
    # "20211117_21_31_08_GFAP_GCamp6s_F6_PTZ",
    "20211119_16_36_20_GFAP_GCamp6s_F4_PTZ",
    # "20211119_18_15_06_GFAP_GCamp6s_F5_C",
    # "20211119_21_52_35_GFAP_GCamp6s_F7_C",
    "20220211_13_18_56_GFAP_GCamp6s_F2_C",
    # "20220211_15_02_16_GFAP_GCamp6s_F3_PTZ",
    "20220211_16_51_15_GFAP_GCamp6s_F4_PTZ",
    # "20220412_10_43_04_GFAP_GCamp6s_F1_PTZ",
    "20220412_12_32_27_GFAP_GCamp6s_F2_PTZ",
    "20220412_13_59_55_GFAP_GCamp6s_F3_C",
    # "20220412_16_06_54_GFAP_GCamp6s_F4_PTZ",
]
CROP_IDS = [
    "OpticTectum",
]

USE_DENOISED = True
USE_CHAN_2 = False
NAME_ROIS = "rois_smooth"

SIGMA = 0.9
THRESHOLD = 0.05


def get_adjacent_pixels(pixel: tuple, dims: tuple, include_self: bool):
    """
    Finds adjacent pixels to pixel.

    Input:
        - pixel: tuple (y, x)
        - dims: tuple, shape of rect movie (num_frames, num_y, num_x)

    Returns:
        - adjacent_pixels: list of pixels
    """

    adjacent_pixels = []
    if include_self:
        adjacent_pixels.append(pixel)

    padding_y_low = pixel[0] > 0
    padding_y_high = pixel[0] < dims[0] - 1
    padding_x_low = pixel[1] > 0
    padding_x_high = pixel[1] < dims[1] - 1

    if padding_y_low:
        adjacent_pixels.append((pixel[0] - 1, pixel[1]))

    if padding_y_high:
        adjacent_pixels.append((pixel[0] + 1, pixel[1]))

    if padding_x_low:
        adjacent_pixels.append((pixel[0], pixel[1] - 1))

    if padding_x_high:
        adjacent_pixels.append((pixel[0], pixel[1] + 1))

    if padding_y_low and padding_x_low:
        adjacent_pixels.append((pixel[0] - 1, pixel[1] - 1))

    if padding_y_low and padding_x_high:
        adjacent_pixels.append((pixel[0] - 1, pixel[1] + 1))

    if padding_y_high and padding_x_low:
        adjacent_pixels.append((pixel[0] + 1, pixel[1] - 1))

    if padding_y_high and padding_x_high:
        adjacent_pixels.append((pixel[0] + 1, pixel[1] + 1))

    return adjacent_pixels


def get_neighbours(pixel: tuple, roi):
    active_neighbours = []
    dims = roi.shape
    neighbours = get_adjacent_pixels(pixel, dims, False)

    for neighbour in neighbours:
        if roi[neighbour]:
            active_neighbours.append(neighbour)

    return active_neighbours


def build_group(pixels_to_add: set, roi, group: set):
    """
    Builds group of connected active pixels from a seed pixel.

    Input:
        - pixel: set of pixels to add to group, set(tuple (y, x))
        - roi: boolean ndarray (n_y, n_x) that satisfy:
        - group: set in which the pixels belonging to the group is stored

    Returns:
        - group: set of pixels that form the group defined by adjacency to seed pixel(s).
        - active_mask: active_mask with member pixels set to False
    """

    if not pixels_to_add:
        return group, roi

    for pixel in list(pixels_to_add):
        group.add(pixel)
        pixels_to_add.remove(pixel)
        roi[pixel] = False
        active_neighbours = get_neighbours(pixel, roi)
        if len(active_neighbours) == 0:
            continue

        for neighbour in active_neighbours:
            pixels_to_add.add(neighbour)

    return build_group(pixels_to_add, roi, group)


def smooth_rois(rois, sigma, threshold):
    new_rois = []
    for roi in rois:
        smooth_roi = gaussian_filter(roi.astype(float), sigma)
        tmp_roi = smooth_roi > threshold
        sub_rois = []
        num_remaining_pixels = np.sum(tmp_roi)
        y_idx, x_idx = np.nonzero(tmp_roi)
        for i in range(y_idx.shape[0]):
            pixel = (y_idx[i], x_idx[i])
            if not tmp_roi[pixel]:
                continue

            sub_roi, tmp_roi = build_group({pixel}, tmp_roi, set())
            sub_rois.append(sub_roi)

        sub_sizes = np.zeros(len(sub_rois))
        for i, sub_roi in enumerate(sub_rois):
            sub_sizes[i] = len(sub_roi)

        largest_roi = sub_rois[np.argmax(sub_sizes)]
        new_roi = np.zeros(tmp_roi.shape, dtype=bool)
        for pixel in largest_roi:
            new_roi[pixel] = True

        new_rois.append(new_roi)

        """ plt.figure()
        plt.imshow(roi, cmap="gray", origin="lower")
        plt.figure()
        plt.imshow(smooth_roi, cmap="gray", origin="lower")
        plt.colorbar()
        plt.figure()
        plt.imshow(new_roi, cmap="gray", origin="lower")
        plt.show() """

    return new_rois


def process_exp(exp_name, crop_id):
    exp_dir = generate_exp_dir(exp_name, crop_id)

    if USE_DENOISED:
        exp_dir = generate_denoised_dir(exp_dir, USE_CHAN_2)

    rois = load_custom_rois(exp_dir)

    new_rois = smooth_rois(rois, SIGMA, THRESHOLD)

    save_rois(new_rois, exp_dir, NAME_ROIS)


def main():
    ts = []
    for exp_name in EXP_NAMES:
        for crop_id in CROP_IDS:
            t = Thread(
                target=process_exp,
                args=(
                    exp_name,
                    crop_id,
                ),
            )
            t.start()
            ts.append(t)

    for t in ts:
        t.join()


if __name__ == "__main__":
    main()
