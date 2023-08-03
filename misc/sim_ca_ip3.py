import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA

N_NODES = 10
N_SYNAPSES = 1000
N_NEURONS = 50
N_C = 1000
NUM_FRAMES = 3000


def decay(x, l):
    return x * l


def a_func(x):
    return np.minimum(np.maximum(np.zeros(x.shape), x), np.ones(x.shape))


def open_prob_ca(ca, mu, sigma):
    return np.power(
        1
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * np.power((ca - mu) / sigma, 2)),
        2,
    )


def open_prob_ip3(ip3, ip3_0, k):
    return 1 / (1 + np.exp(-k * (ip3 - ip3_0)))


def plot_open_prob(mu, sigma, ip3_0, k):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    ip3 = np.arange(-1, 10, 0.05)
    ca = np.arange(-1, 10, 0.05)
    ip3, ca = np.meshgrid(ip3, ca)
    Z = open_prob_ip3(ip3, ip3_0, k) * open_prob_ca(ca, mu, sigma)

    # Plot the surface.
    surf = ax.plot_surface(ip3, ca, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_xlabel("ip3")
    ax.set_ylabel("ca2+")

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter("{x:.02f}")

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)


def amplification(ca, mu, sigma, ip3, ip3_0, k, c, t_c, a):

    # maximum releaseable calcium from stores
    zero = np.zeros(ca.shape)
    exp_k = np.exp(-1e-4 * np.arange(N_C))
    exp_k = np.flip(exp_k / np.sum(exp_k) * N_C)
    t = np.maximum(t_c - np.sum((c.T * exp_k).T, axis=0), zero)

    # bell-shaped dependence on calcium (mu, sigma)
    ca_bell = open_prob_ca(ca, mu, sigma)

    # sigmoid-shaped dependence on ip3
    ip3_sigmoid = open_prob_ip3(ip3, ip3_0, k)

    amp = np.minimum(np.maximum(ca_bell * ip3_sigmoid * a, zero), t)
    return amp


def synapse_astro(s, W):
    return W @ s * N_NODES / N_SYNAPSES


def diffusion(x, D):
    return D @ x


def update(
    i,
    ca,
    dcadt,
    ip3,
    dip3dt,
    s,
    n,
    c,
    W_ca,
    W_ip3,
    V,
    U,
    D,
    sn,
    sn0,
    ns,
    t_c,
    a,
    mu,
    sigma,
    ip3_0,
    k,
    l,
    rng,
):
    c_t = amplification(ca, mu, sigma, ip3, ip3_0, k, c, t_c, a)
    c_old = c[1:]

    dec = decay(ca, l)
    diff = diffusion(ca, D)
    sa = synapse_astro(s, W_ca)

    dcadt = (
        np.multiply(dcadt, np.linspace(0.4, 0.6, N_NODES))
        + sa
        + diff
        + (3 * c_t + 2 * c_old[-1] + c_old[-2]) / 6
    )
    ca = ca + dcadt - dec

    dec = decay(ip3, l)
    diff = diffusion(ip3, D)
    sa = synapse_astro(s, W_ip3)

    dip3dt = np.multiply(dip3dt, np.linspace(0.2, 0.5, N_NODES)) + sa + diff
    ip3 = ip3 + dip3dt - dec

    n = a_func(sn @ s)

    """ if i % 300 < 2:
        n = np.linspace(0, 1, N_NEURONS)
    if i % 500 < 2:
        n = np.flip(np.linspace(0, 1, N_NEURONS)) * 2 """

    s = ns @ n + U @ ca

    asn = V @ ca
    sn_update = np.absolute(sn) > 0.01
    sn_update = np.multiply(asn, sn_update)

    sn = 0.2 * sn + 0.8 * sn0 + sn_update

    c = np.concatenate([c_old, c_t.reshape((1, -1))], axis=0)

    return ca, dcadt, ip3, dip3dt, s, n, c, sn


def run_sim():
    # Initial values
    rng = np.random.default_rng(seed=3)
    ca = np.zeros(N_NODES)
    ip3 = np.zeros(N_NODES)
    dcadt = np.zeros(N_NODES)
    dip3dt = np.zeros(N_NODES)
    s = rng.random(N_SYNAPSES) - 0.5
    n = rng.random(N_NEURONS)
    c = rng.random((N_C, N_NODES))

    W_ca = np.zeros((N_NODES, N_SYNAPSES))
    W_ip3 = np.zeros((N_NODES, N_SYNAPSES))
    V = np.zeros((N_SYNAPSES, N_NODES))
    U = np.zeros((N_SYNAPSES, N_NODES))
    D = np.zeros((N_NODES, N_NODES))
    sn = np.zeros((N_NEURONS, N_SYNAPSES))
    ns = np.zeros((N_SYNAPSES, N_NEURONS))

    p_node = -np.power(np.arange(N_NODES), 2)
    p_node = p_node - np.amin(p_node) + 1
    p_node = p_node / np.sum(p_node)

    p_node = -np.power(np.arange(N_NODES), 2)
    p_node = p_node - np.amin(p_node) + 1
    p_node = p_node / np.sum(p_node)
    for synapse in range(N_SYNAPSES):
        node = rng.choice(N_NODES, p=p_node)
        W_ca[node, synapse] = rng.random() * N_NODES / N_SYNAPSES * 50
        p_ip3 = 0.1
        W_ip3[node, synapse] = (
            rng.choice([0, 1], p=[1 - p_ip3, p_ip3]) * N_NODES / N_SYNAPSES * 50
        )
        V[synapse, node] = (rng.random() - 0.6) / N_NODES * 1e-30
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

    t_c = 1
    a = 1
    mu = 4
    sigma = 0.5
    ip3_0 = 3
    k = 2
    min_l, max_l = 0.01, 0.2
    l = max_l - np.linspace(min_l, max_l, N_NODES) + min_l

    ca_full = np.zeros((N_NODES, NUM_FRAMES))
    ip3_full = np.zeros((N_NODES, NUM_FRAMES))
    s_full = np.zeros((N_SYNAPSES, NUM_FRAMES))
    n_full = np.zeros((N_NEURONS, NUM_FRAMES))
    c_full = np.zeros((N_NODES, NUM_FRAMES))
    for i in range(NUM_FRAMES):
        ca, dcadt, ip3, dip3dt, s, n, c, sn = update(
            i,
            ca,
            dcadt,
            ip3,
            dip3dt,
            s,
            n,
            c,
            W_ca,
            W_ip3,
            V,
            U,
            D,
            sn,
            sn0,
            ns,
            t_c,
            a,
            mu,
            sigma,
            ip3_0,
            k,
            l,
            rng,
        )
        ca_full[:, i] = ca
        ip3_full[:, i] = ip3
        s_full[:, i] = s
        n_full[:, i] = n
        c_full[:, i] = c[-1]

    n_skip = 200
    ca_full = ca_full[:, n_skip:]
    ip3_full = ip3_full[:, n_skip:]
    s_full = s_full[:, n_skip:]
    n_full = n_full[:, n_skip:]
    c_full = c_full[:, n_skip:]
    ca_mu = np.mean(ca_full, axis=1)
    ca_full = (ca_full.T - ca_mu).T
    ip3_mu = np.mean(ip3_full, axis=1)
    ip3_full = (ip3_full.T - ip3_mu).T
    print(ca_mu)
    print(ip3_mu)

    plot_open_prob(mu, sigma, ip3_0, k)

    colors = cm.viridis_r(np.linspace(0.2, 1, N_NODES))

    av_ca = np.mean(ca_full, axis=0)
    delta_ca = np.amax(ca_full) - np.amin(ca_full)

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
    for node, ca_node in enumerate(ca_full):
        if node == 0:
            ax.plot(av_ca, color="darkgray", alpha=0.7, label="Average")
        else:
            ax.plot(av_ca - delta_ca * node, color="darkgray", alpha=0.7)

        ax.plot(
            ca_node - delta_ca * node,
            color=colors[node],
            label=f"Node {node+1}",
            alpha=0.8,
        )
    ax.set_title("Astrocyte ca")

    av_ip3 = np.mean(ip3_full, axis=0)
    delta_y = np.amax(ip3_full) - np.amin(ip3_full)

    _, ax = plt.subplots()
    for node, ip3_node in enumerate(ip3_full):
        if node == 0:
            ax.plot(av_ip3, color="darkgray", alpha=0.7, label="Average")
        else:
            ax.plot(av_ip3 - delta_y * node, color="darkgray", alpha=0.7)

        ax.plot(
            ip3_node - delta_y * node,
            color=colors[node],
            label=f"Node {node+1}",
            alpha=0.8,
        )
    ax.set_title("Astrocyte ip3")

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
