import numpy as np
import matplotlib.pyplot as plt


def plot_frame(x, frame_number=0, title=""):
    x_frame = np.squeeze(x[frame_number])

    plt.figure()
    plt.imshow(x_frame, "gray")
    plt.title(f"Frame {frame_number} {title}")


def plot_rois(rois, title=""):
    plt.figure()
    for roi in rois:
        mu_im_w_rois[roi] = (mu_im[roi] + color) / 2

    plt.imshow(mu_im_w_rois)
    plt.axis("off")
