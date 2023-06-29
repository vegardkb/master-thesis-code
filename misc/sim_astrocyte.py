import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

N_NODES = 6
N_SYNAPSES = 200
N_C = 100
NUM_FRAMES = 1000


def decay(x, l):
    return x * l


def amplification(x, c, t_c, a, mu, sigma):
    exp_k = np.exp(-1e-1 * np.arange(N_C))
    exp_k = np.flip(exp_k / np.sum(exp_k) * N_C)

    t = t_c - np.sum((c.T * exp_k).T, axis=0)

    # t = t_c - np.sum(c, axis=0)
    zero = np.zeros(x.shape)

    amp = (
        a / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.power((x - mu) / sigma, 2))
    )

    amp = np.minimum(np.maximum(amp, zero), t)
    return amp


def neuron_glia(y, W):
    """y_relu = np.maximum(np.zeros(y.shape), y) * 200 / N_SYNAPSES
    return W @ np.power(y_relu, 2)"""

    return W @ y


def diffusion(x, D):
    return D @ x


def update(x, y, c, W, D, t_c, a, mu, sigma, l, rng):
    c_t = amplification(x, c, t_c, a, mu, sigma)
    c_old = c[1:]

    dec = decay(x, l)
    diff = diffusion(x, D)
    na = neuron_glia(y, W)

    x = x + na + diff + c_t - dec
    y = 0.1 * y + rng.random(N_SYNAPSES) - 0.5
    c = np.concatenate([c_old, c_t.reshape((1, -1))], axis=0)

    return x, y, c


def run_sim():
    # Initial values
    rng = np.random.default_rng(seed=3)
    x = rng.random(N_NODES)
    y = rng.random(N_SYNAPSES) - 0.5
    c = rng.random((N_C, N_NODES))

    W = np.zeros((N_NODES, N_SYNAPSES))
    p_node = -np.power(np.arange(N_NODES), 1.5)
    p_node = p_node - np.amin(p_node) + 1
    p_node = p_node / np.sum(p_node)
    for synapse in range(N_SYNAPSES):
        node = rng.choice(N_NODES, p=p_node)
        W[node, synapse] = rng.random()

    D = np.zeros((N_NODES, N_NODES))
    d = 0.2
    for i in range(N_NODES):
        for j in range(N_NODES):
            if i - j == 1 or j - i == 1:
                D[i, j] = d
                D[i, i] = D[i, i] - d

    t_c = 30
    a = 3
    mu = 4
    sigma = 1
    min_l, max_l = 0.02, 0.3
    l = max_l - np.linspace(min_l, max_l, N_NODES) + min_l

    x_full = np.zeros((N_NODES, NUM_FRAMES))
    y_full = np.zeros((N_SYNAPSES, NUM_FRAMES))
    for i in range(NUM_FRAMES):
        x, y, c = update(x, y, c, W, D, t_c, a, mu, sigma, l, rng)
        x_full[:, i] = x
        y_full[:, i] = y

    x_full = x_full[:, 100:]
    x_mu = np.mean(x_full, axis=1)
    x_full = (x_full.T - x_mu).T
    print(x_mu)

    colors = cm.viridis_r(np.linspace(0.2, 1, N_NODES))

    av_x = np.mean(x_full, axis=0)
    delta_y = np.amax(x_full) - np.amin(x_full)

    _, ax = plt.subplots()
    ax.imshow(y_full)

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

    plt.show()


def main():
    run_sim()


if __name__ == "__main__":
    main()
