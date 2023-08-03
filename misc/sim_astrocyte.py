import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA

N_NODES = 10
N_SYNAPSES = 2000
N_NEURONS = 100
N_C = 400
NUM_FRAMES = 2000


def decay(x, l):
    return x * l


def a_func(x):
    return np.minimum(np.maximum(np.zeros(x.shape), x), np.ones(x.shape))


def amplification(x, c, t_c, a, mu, sigma):
    exp_k = np.exp(-5e-3 * np.arange(N_C))
    exp_k = np.flip(exp_k / np.sum(exp_k) * N_C)

    zero = np.zeros(x.shape)
    t = np.maximum(t_c - np.sum((c.T * exp_k).T, axis=0), zero)

    amp = np.power(
        a / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.power((x - mu) / sigma, 2)),
        2,
    )

    amp = np.minimum(np.maximum(amp, zero), t)
    return amp


def synapse_astro(s, W):
    return W @ s * N_NODES / N_SYNAPSES


def diffusion(x, D):
    return D @ x


def update(i, x, dxdt, s, n, c, W, V, U, D, sn, sn0, ns, t_c, a, mu, sigma, l, rng):
    c_t = amplification(x, c, t_c, a, mu, sigma)
    c_old = c[1:]

    dec = decay(x, l)
    diff = diffusion(x, D)
    sa = synapse_astro(s, W)

    dxdt = (
        np.multiply(dxdt, np.linspace(0.2, 0.5, N_NODES))
        + sa
        + diff
        + (3 * c_t + 2 * c_old[-1] + c_old[-2]) / 6
    )
    x = x + dxdt - dec
    n = a_func(sn @ s)

    """ if i % 300 < 2:
        n = np.linspace(0, 1, N_NEURONS)
    if i % 500 < 2:
        n = np.flip(np.linspace(0, 1, N_NEURONS)) * 2 """

    s = ns @ n + U @ x

    asn = V @ x
    sn_update = np.absolute(sn) > 0.01
    sn_update = np.multiply(asn, sn_update)
    # print(f"sn_update:\n{sn_update}")

    sn = 0.2 * sn + 0.8 * sn0 + sn_update

    c = np.concatenate([c_old, c_t.reshape((1, -1))], axis=0)

    return x, dxdt, s, n, c, sn


def run_sim():
    # Initial values
    rng = np.random.default_rng(seed=24)
    x = np.zeros(N_NODES)
    dxdt = np.zeros(N_NODES)
    s = rng.random(N_SYNAPSES) - 0.5
    n = rng.random(N_NEURONS)
    c = rng.random((N_C, N_NODES))

    W = np.zeros((N_NODES, N_SYNAPSES))
    V = np.zeros((N_SYNAPSES, N_NODES))
    U = np.zeros((N_SYNAPSES, N_NODES))
    D = np.zeros((N_NODES, N_NODES))
    sn = np.zeros((N_NEURONS, N_SYNAPSES))
    ns = np.zeros((N_SYNAPSES, N_NEURONS))
    p_node = -np.power(np.arange(N_NODES), 1.5)
    p_node = p_node - np.amin(p_node) + 1
    p_node = p_node / np.sum(p_node)
    for synapse in range(N_SYNAPSES):
        node = rng.choice(N_NODES, p=p_node)
        W[node, synapse] = rng.random() * N_NODES / N_SYNAPSES * 50
        V[synapse, node] = (rng.random() - 0.6) / N_NODES * 1
        U[synapse, node] = (rng.random() - 0.6) / N_NODES * 1e-30

        post_n = rng.choice(N_NEURONS)
        pre_n = post_n
        while pre_n == post_n:
            pre_n = rng.choice(N_NEURONS)

        p_ex = 0.55
        sn[post_n, synapse] = rng.choice([-0.5, 0.5], p=[1 - p_ex, p_ex])
        ns[synapse, pre_n] = 1

    sn0 = copy.deepcopy(sn)

    d = 0.05
    for i in range(N_NODES):
        for j in range(N_NODES):
            if i - j == 1 or j - i == 1:
                D[i, j] = d
                D[i, i] = D[i, i] - d

    t_c = np.linspace(0.5, 3, N_NODES)
    a = 20
    mu = 4.5
    sigma = np.linspace(0.4, 0.75, N_NODES)
    min_l, max_l = 0.015, 0.2
    l = max_l - np.linspace(min_l, max_l, N_NODES) + min_l

    x_full = np.zeros((N_NODES, NUM_FRAMES))
    s_full = np.zeros((N_SYNAPSES, NUM_FRAMES))
    n_full = np.zeros((N_NEURONS, NUM_FRAMES))
    c_full = np.zeros((N_NODES, NUM_FRAMES))
    for i in range(NUM_FRAMES):
        x, dxdt, s, n, c, sn = update(
            i, x, dxdt, s, n, c, W, V, U, D, sn, sn0, ns, t_c, a, mu, sigma, l, rng
        )
        x_full[:, i] = x
        s_full[:, i] = s
        n_full[:, i] = n
        c_full[:, i] = c[-1]

    n_skip = 200
    x_full = x_full[:, n_skip:]
    s_full = s_full[:, n_skip:]
    n_full = n_full[:, n_skip:]
    c_full = c_full[:, n_skip:]
    x_mu = np.mean(x_full, axis=1)
    x_full = (x_full.T - x_mu).T
    print(x_mu)

    colors = cm.viridis_r(np.linspace(0.2, 1, N_NODES))

    av_x = np.mean(x_full, axis=0)
    delta_y = np.amax(x_full) - np.amin(x_full)

    fig, ax = plt.subplots()
    im = ax.imshow(s_full, aspect="auto")
    ax.set_title("Synapses")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    fig, ax = plt.subplots()
    im = ax.imshow(n_full, aspect="auto", interpolation="none", vmin=0, vmax=1)
    ax.set_title("Neurons")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    _, ax = plt.subplots()
    for node, x_node in enumerate(x_full):
        if node == 0:
            ax.plot(av_x, color="darkgray", alpha=0.7, label="Average")
        else:
            ax.plot(av_x - delta_y * node, color="darkgray", alpha=0.7)

        ax.plot(
            x_node - delta_y * node,
            color=colors[node],
            label=f"Node {node+1}",
            alpha=0.8,
        )
    ax.set_title("Astrocyte nodes")

    av_c = np.mean(c_full, axis=0)
    delta_y = np.amax(c_full) - np.amin(c_full)

    _, ax = plt.subplots()
    for node, c_node in enumerate(c_full):
        if node == 0:
            ax.plot(av_c, color="darkgray", alpha=0.7, label="Average")
        else:
            ax.plot(av_c - delta_y * node, color="darkgray", alpha=0.7)

        ax.plot(
            c_node - delta_y * node,
            color=colors[node],
            label=f"Node {node+1}",
            alpha=0.8,
        )
    ax.set_title("Astrocyte amplification")

    pca = PCA(n_components=2)
    n_pca = pca.fit_transform(n_full.T)

    c = cm.viridis(np.linspace(0, 1, n_pca.shape[0]))
    _, ax = plt.subplots()
    ax.scatter(n_pca[:, 0], n_pca[:, 1], c=c, s=10)
    ax.set_title("Neurons scatter")

    plt.show()


def main():
    run_sim()


if __name__ == "__main__":
    main()
