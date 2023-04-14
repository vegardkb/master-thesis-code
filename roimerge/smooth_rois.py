import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from fio import (
    load_cfg,
    load_s2p_df_rois,
    get_t,
    generate_exp_dir,
    generate_denoised_dir,
    save_rois,
)

from plotters import plot_frame, plot_rois

from sigproc import (
    get_mean_img_rgb,
    get_pos,
    spatial_distance,
    activity_distance,
)

from uio import merge_suggest_gui


EXP_NAMES = [
    "20211112_13_23_43_GFAP_GCamp6s_F2_PTZ",
    "20220211_13_18_56_GFAP_GCamp6s_F2_C",
]
CROP_IDS = [
    "OpticTectum",
]

USE_DENOISED = True
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


def main():
    for exp_name in EXP_NAMES:
        for crop_id in CROP_IDS:
            exp_dir = generate_exp_dir(exp_name, crop_id)
            cfg = load_cfg(exp_dir)

            if USE_DENOISED:
                exp_dir = generate_denoised_dir(exp_dir)

            s2p_df, rois = load_s2p_df_rois(exp_dir, cfg)

            new_rois = smooth_rois(rois, SIGMA, THRESHOLD)

            save_rois(new_rois, exp_dir, NAME_ROIS)


if __name__ == "__main__":
    main()
