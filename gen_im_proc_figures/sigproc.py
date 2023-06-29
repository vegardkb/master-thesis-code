import numpy as np
from sklearn.decomposition import PCA


def get_pos(Ly, Lx):
    y = np.linspace(0, Ly - 1, Ly, dtype=int)
    x = np.linspace(0, Lx - 1, Lx, dtype=int)
    yv, xv = np.meshgrid(y, x, indexing="ij")
    pos = np.stack([yv, xv], axis=0)
    return pos


def calc_c_pos(roi, pos, Ly):
    """
    Returns:
        - c_pos: ndarray: (2, n_pix). c_pos[0] = y-position, c_pos[1] = x-position.
    """
    c_pos = np.reshape(pos[:, roi], (2, -1))
    c_pos[0, :] = Ly - c_pos[0, :]
    return c_pos


def calc_pos_pca0(c_pos, cfg, micron_per_meter):
    pca = PCA(n_components=2).fit(c_pos.T)
    center = np.mean(c_pos, axis=1)
    comp0 = pca.components_[0]
    comp0y = comp0[0]
    comp0x = comp0[1]
    if center[0] - cfg.Ly / 2 > 0:
        """
        center below midline
        """
        if comp0y > 0:
            comp0[0] = -np.absolute(comp0y)
            comp0[1] = -comp0x

    else:
        """
        cell above midline
        """
        if comp0y < 0:
            comp0[0] = np.absolute(comp0y)
            comp0[1] = -comp0x

    c_pos_demean = (c_pos.T - center).T
    comp0_norm = np.linalg.norm(comp0)

    pos_pca0 = comp0.T @ c_pos_demean / comp0_norm
    pos_pca0 = pos_pca0 * cfg.pixel_size * micron_per_meter
    return pos_pca0


def calc_pc1(c_pos, cfg, micron_per_meter):
    pca = PCA(n_components=2).fit(c_pos.T)
    center = np.mean(c_pos, axis=1)
    comp0 = pca.components_[0]
    comp0y = comp0[0]
    comp0x = comp0[1]
    if center[0] - cfg.Ly / 2 > 0:
        """
        center below midline
        """
        if comp0y > 0:
            comp0[0] = -np.absolute(comp0y)
            comp0[1] = -comp0x

    else:
        """
        cell above midline
        """
        if comp0y < 0:
            comp0[0] = np.absolute(comp0y)
            comp0[1] = -comp0x

    return comp0


def get_reg_mask(pos_pca0, n_regions):
    """
    Divide into regions by position along proximal-distal axis, equal length of regions
    """

    thr = np.linspace(
        np.amin(pos_pca0) - 1e-10, np.amax(pos_pca0) + 1e-10, n_regions + 1
    )

    mask = []
    for i in range(n_regions):
        mask.append(np.logical_and(pos_pca0 > thr[i], pos_pca0 <= thr[i + 1]))

    return np.array(mask)
