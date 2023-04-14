import numpy as np
from scipy import signal


def get_mean_img_rgb(f):
    n_ch = 3
    mean_im = np.zeros((f.shape[1], f.shape[2], n_ch))

    mu_im = np.mean(f, axis=0)
    mu_im = mu_im / np.amax(mu_im)  # Normalize
    for i in range(n_ch):
        mean_im[:, :, i] = mu_im

    return mean_im


def get_pos(Ly, Lx):
    y = np.linspace(0, Ly - 1, Ly)
    x = np.linspace(0, Lx - 1, Lx)
    yv, xv = np.meshgrid(y, x, indexing="ij")
    pos = np.stack([yv, xv], axis=0)
    return pos


def euclid_dist(x, y):
    return np.sqrt(np.sum(np.power(x - y, 2)))


def spatial_distance(roi1, roi2, pos):
    """
    Returns:
    Minumum of distance between roi centers and
    distance between pixels that are furthest in the direction of the other roi's center.
    Arguments:
        - roi1/2: boolean ndarray(Ly,Lx)
        - pos: ndarray(2,Ly,Lx)
    """
    if np.any(np.logical_and(roi1, roi2)):
        # Overlap
        return 0

    pos1 = pos[:, roi1]
    pos2 = pos[:, roi2]
    mu1 = np.mean(pos1, axis=1)
    mu2 = np.mean(pos2, axis=1)

    v_norm = np.linalg.norm(mu2 - mu1)
    v21 = (mu2 - mu1) / v_norm
    v12 = (mu1 - mu2) / v_norm

    proj_v21_pos1 = pos1.T @ v21
    proj_v12_pos2 = pos2.T @ v12

    closest_pos1_ind = np.argmax(proj_v21_pos1)
    closest_pos2_ind = np.argmax(proj_v12_pos2)

    return min(
        euclid_dist(pos1[:, closest_pos1_ind], pos2[:, closest_pos2_ind]),
        euclid_dist(mu1, mu2),
    )


def activity_distance(roi1, roi2, f, pos, t, pixel_size):
    min_speed = 1  # um/sec

    pos1 = pos[:, roi1]
    pos2 = pos[:, roi2]
    mu1 = np.mean(pos1, axis=1)
    mu2 = np.mean(pos2, axis=1)

    dist = euclid_dist(mu1, mu2) * pixel_size

    max_delay = dist / min_speed

    f1, f2 = f[:, roi1], f[:, roi2]
    mu_fs = [np.mean(f1, axis=1), np.mean(f2, axis=1)]

    mu_fs_norm = []
    for mu_f in mu_fs:
        mu_fs_norm.append((mu_f - np.mean(mu_f, axis=0)) / np.std(mu_f, axis=0))

    xcorr = signal.correlate(mu_fs_norm[0], mu_fs_norm[1], mode="same")
    xcorr = xcorr / (np.linalg.norm(mu_fs_norm[0]) * np.linalg.norm(mu_fs_norm[1]))
    delay = np.linspace(-np.amax(t) / 2, np.amax(t) / 2, t.shape[0])
    plausible_mask = np.absolute(delay) < max_delay
    if not np.any(plausible_mask):
        peak = xcorr[np.argmin(np.absolute(delay))]
    else:
        peak = np.amax(xcorr[plausible_mask])
    return 1 - peak
