import numpy as np
from time import time
import matplotlib.pyplot as plt
from threading import Thread
from queue import Queue
from scipy import signal
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.widgets import Button

SEC_PER_MIN = 60


def incr_frame_counter(counter, n_frames, t0):
    te = int(np.round(time() - t0))
    te_min = te // SEC_PER_MIN
    te_sec = te % SEC_PER_MIN

    tr = (
        int(np.round(te * (n_frames - counter) / counter))
        if counter != 0
        else int(np.round(te * 2 * n_frames))
    )
    tr_min = tr // SEC_PER_MIN
    tr_sec = tr % SEC_PER_MIN

    will_print = False
    if counter + 1 == n_frames:
        end_char = "\n"
        will_print = True

    elif counter % 500 == 0:
        end_char = "\r"
        will_print = True

    if will_print:
        print(
            f"Loading frame {counter} / {n_frames}, time elapsed: {te_min}:{te_sec}, time remaining: {tr_min}:{tr_sec}",
            end=end_char,
        )

    return counter + 1


def timer_func(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.3f}s")
        return result

    return wrap_func


def plot_two_rois(mean_im, f, rois, k, l, t):
    """
    Functionality that would be nice to have:
        - Plot ROIs
    Bug fixes and improvements:
        - Faster toggle roi on/off
        - Duplicate ROI at start of each iteration

    This is way too big and has non-informative name
    """
    n_ch = 3
    roi1, roi2 = rois[k], rois[l]
    f1, f2 = f[:, roi1], f[:, roi2]
    mu_fs = [np.mean(f1, axis=1), np.mean(f2, axis=1)]
    rois = [roi1, roi2]
    roi_names = ["roi1", "roi2"]
    colors = ["green", "red"]
    opacity_t = 0.8

    im_shape = (roi1.shape[0], roi1.shape[1], n_ch)

    roi_im = np.zeros(im_shape)

    for i, roi_m in enumerate(rois):
        roi_im[roi_m] = roi_im[roi_m] + mcolors.to_rgb(colors[i])

    mu_fs_norm = []
    for mu_f in mu_fs:
        mu_fs_norm.append((mu_f - np.mean(mu_f, axis=0)) / np.std(mu_f, axis=0))

    xcorr = signal.correlate(mu_fs_norm[0], mu_fs_norm[1], mode="same")
    xcorr = xcorr / (np.linalg.norm(mu_fs_norm[0]) * np.linalg.norm(mu_fs_norm[1]))
    t_max = np.amax(t)
    delay = np.linspace(-t_max / 2, t_max / 2, t.shape[0])
    delay_max = delay[np.argmax(xcorr)]

    fig = plt.figure()
    fig.set_figwidth(16)
    fig.set_figheight(10)
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_title("Mean img with roi masks")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Mean across rois")
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title(f"xcorr(roi1, roi2), peak at {np.round(delay_max, 2)} sec")

    im_w_roi = ax1.imshow((mean_im + roi_im) / np.amax(mean_im + roi_im))
    im_wo_roi = ax1.imshow(mean_im / np.amax(mean_im))
    labels = ["roi1", "roi2"]
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(2)]
    ax1.legend(handles=patches)
    im_wo_roi.set_visible(False)

    for i in range(2):
        ax2.plot(t, mu_fs_norm[i], color=colors[i], label=roi_names[i], alpha=opacity_t)

    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("z-score")

    ax3.plot(delay, xcorr)
    ax3.set_xlabel("Delay [s]")

    class UI:
        user_input = "n"
        show_roi = True

        def close_fig(self):
            plt.close()

        def merge(self, _):
            self.user_input = "y"
            self.close_fig()

        def not_merge(self, _):
            self.user_input = "n"
            self.close_fig()

        def skip_iteration(self, _):
            self.user_input = "skip"
            self.close_fig()

        def save_quit(self, _):
            self.user_input = "q"
            self.close_fig()

        def remove_roi1(self, _):
            self.user_input = "remove_1"
            self.close_fig()

        def remove_roi2(self, _):
            self.user_input = "remove_2"
            self.close_fig()

        def toggle_roi(self, _):
            self.show_roi = not self.show_roi
            if self.show_roi:
                im_w_roi.set_visible(True)
                im_wo_roi.set_visible(False)
            else:
                im_w_roi.set_visible(False)
                im_wo_roi.set_visible(True)

    ui = UI()

    btn_y = 0.95
    btn_x0 = 0.2
    btn_w = 0.065
    btn_h = 0.025
    btn_spacex = 0.01

    cbs_roi = [ui.remove_roi1, ui.remove_roi2, ui.toggle_roi]
    txt = ["remove roi1", "remove roi2", "toggle_roi"]
    btns_roi = []
    for i, cb in enumerate(cbs_roi):
        btn_ax = plt.axes([btn_x0 + i * (btn_spacex + btn_w), btn_y, btn_w, btn_h])
        btn = Button(btn_ax, txt[i])
        btn.on_clicked(cb)
        btns_roi.append(btn)

    btn_x0 = 0.6
    cbs = [ui.merge, ui.not_merge, ui.skip_iteration, ui.save_quit]
    txt = ["merge", "no merge", "skip it", "save quit"]
    btns_state = []
    for i, cb in enumerate(cbs):
        btn_ax = plt.axes([btn_x0 + i * (btn_spacex + btn_w), btn_y, btn_w, btn_h])
        btn = Button(btn_ax, txt[i])
        btn.on_clicked(cb)
        btns_state.append(btn)

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

    plt.show()
    return ui.user_input


def plot_one_roi(mean_im, f, rois, k, t):
    n_ch = 3
    roi = rois[k]
    f1 = f[:, roi]
    mu_f = np.mean(f1, axis=1)
    label = "roi"
    c = "green"
    opacity_t = 0.8

    im_shape = (roi.shape[0], roi.shape[1], n_ch)

    roi_im = np.zeros(im_shape)

    roi_im[roi] = roi_im[roi] + mcolors.to_rgb(c)

    mu_f_norm = (mu_f - np.mean(mu_f, axis=0)) / np.std(mu_f, axis=0)

    fig = plt.figure()
    fig.set_figwidth(16)
    fig.set_figheight(10)
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title("Mean img with roi masks")
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_title("Mean across rois")

    im_w_roi = ax1.imshow((mean_im + roi_im) / np.amax(mean_im + roi_im))
    im_wo_roi = ax1.imshow(mean_im / np.amax(mean_im))
    patches = [mpatches.Patch(color=c, label=label)]
    ax1.legend(handles=patches)
    im_wo_roi.set_visible(False)

    ax2.plot(t, mu_f_norm, color=c, label=label, alpha=opacity_t)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("z-score")

    class UI:
        user_input = "keep"
        show_roi = True

        def close_fig(self):
            plt.close()

        def skip_iteration(self, _):
            self.user_input = "skip"
            self.close_fig()

        def save_quit(self, _):
            self.user_input = "q"
            self.close_fig()

        def remove_roi(self, _):
            self.user_input = "remove"
            self.close_fig()

        def keep_roi(self, _):
            self.user_input = "keep"
            self.close_fig()

        def toggle_roi(self, _):
            self.show_roi = not self.show_roi
            if self.show_roi:
                im_w_roi.set_visible(True)
                im_wo_roi.set_visible(False)
            else:
                im_w_roi.set_visible(False)
                im_wo_roi.set_visible(True)

    ui = UI()

    btn_y = 0.95
    btn_x0 = 0.2
    btn_w = 0.065
    btn_h = 0.025
    btn_spacex = 0.01

    cbs_roi = [ui.remove_roi, ui.keep_roi, ui.toggle_roi]
    txt = ["remove roi", "keep_roi", "toggle_roi"]
    btns_roi = []
    for i, cb in enumerate(cbs_roi):
        btn_ax = plt.axes([btn_x0 + i * (btn_spacex + btn_w), btn_y, btn_w, btn_h])
        btn = Button(btn_ax, txt[i])
        btn.on_clicked(cb)
        btns_roi.append(btn)

    btn_x0 = 0.6
    cbs = [ui.skip_iteration, ui.save_quit]
    txt = ["skip it", "save quit"]
    btns_state = []
    for i, cb in enumerate(cbs):
        btn_ax = plt.axes([btn_x0 + i * (btn_spacex + btn_w), btn_y, btn_w, btn_h])
        btn = Button(btn_ax, txt[i])
        btn.on_clicked(cb)
        btns_state.append(btn)

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

    plt.show()
    return ui.user_input


def get_user_input(prompt, q, valid):
    user_input = input(prompt)

    while user_input not in valid:
        print("Invalid input")
        user_input = input(prompt)

    q.put(user_input)
    plt.close()


def merge_suggest_cli_gui(mean_im, f, rois, k, l, t):
    """
    Start CLI thread
    """
    q = Queue()
    valid = ["y", "n", "q"]
    t_cli = Thread(target=get_user_input, args=("Merge ROIs? (y/n/q)", q, valid))
    t_cli.start()

    """
        Display merge suggestion
        TODO:
            - Make this less crappy.
    """
    plot_two_rois(mean_im, f, rois, k, l, t)
    plt.show()

    """
        Wait for CLI thread to join
    """
    t_cli.join()
    plt.close()
    return q.get()


def merge_suggest_gui(mean_im, f, rois, k, l, t):
    return plot_two_rois(mean_im, f, rois, k, l, t)
