"""
Simulator of graphons

Reference:
Chan, Stanley, and Edoardo Airoldi.
"A consistent histogram estimator for exchangeable graph models."
In International Conference on Machine Learning, pp. 208-216. 2014.
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import ot

from typing import List


def synthesize_graphon(r: int = 1000, type_idx: int = 0) -> np.ndarray:
    """
    Synthesize graphons
    :param r: the resolution of discretized graphon
    :param type_idx: the type of graphon
    :return:
        w: (r, r) float array, whose element is in the range [0, 1]
    """
    u = ((np.arange(0, r) + 1) / r).reshape(-1, 1)  # (r, 1)
    v = ((np.arange(0, r) + 1) / r).reshape(1, -1)  # (1, r)

    if type_idx == 0:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = u @ v
    elif type_idx == 1:
        w = np.exp(-(u ** 0.7 + v ** 0.7))
    elif type_idx == 2:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 0.25 * (u ** 2 + v ** 2 + u ** 0.5 + v ** 0.5)
    elif type_idx == 3:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 0.5 * (u + v)
    elif type_idx == 4:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 1 / (1 + np.exp(-2 * (u ** 2 + v ** 2)))
    elif type_idx == 5:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 1 / (1 + np.exp(-(np.maximum(u, v) ** 2 + np.minimum(u, v) ** 4)))
    elif type_idx == 6:
        w = np.exp(-np.maximum(u, v) ** 0.75)
    elif type_idx == 7:
        w = np.exp(-0.5 * (np.minimum(u, v) + u ** 0.5 + v ** 0.5))
    elif type_idx == 8:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = np.log(1 + 0.5 * np.maximum(u, v))
    elif type_idx == 9:
        w = np.abs(u - v)
    elif type_idx == 10:
        w = 1 - np.abs(u - v)
    elif type_idx == 11:
        r2 = int(r / 2)
        w = np.kron(np.eye(2, dtype=int), 0.8 * np.ones((r2, r2)))
    elif type_idx == 12:
        r2 = int(r / 2)
        w = np.kron(np.eye(2, dtype=int), np.ones((r2, r2)))
        w = 0.8 * (1 - w)
    else:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = u @ v

    return w


def simulate_graphs(w: np.ndarray, num_graphs: int = 10,
                    num_nodes: int = 200, graph_size: str = 'fixed') -> List[np.ndarray]:
    """
    Simulate graphs based on a graphon
    :param w: a (r, r) discretized graphon
    :param num_graphs: the number of simulated graphs
    :param num_nodes: the number of nodes per graph
    :param graph_size: fix each graph size as num_nodes or sample the size randomly as num_nodes * (0.5 + uniform)
    :return:
        graphs: a list of binary adjacency matrices
    """
    graphs = []
    r = w.shape[0]
    if graph_size == 'fixed':
        numbers = [num_nodes for _ in range(num_graphs)]
    elif graph_size == 'random':
        numbers = [int(num_nodes * (0.5 + np.random.rand())) for _ in range(num_graphs)]
    else:
        numbers = [num_nodes for _ in range(num_graphs)]
    print(numbers)

    for n in range(num_graphs):
        node_locs = (r * np.random.rand(numbers[n])).astype('int')
        graph = w[node_locs, :]
        graph = graph[:, node_locs]
        noise = np.random.rand(graph.shape[0], graph.shape[1])
        graph -= noise
        graphs.append((graph > 0).astype('float'))

    return graphs


def gw_distance(graphon: np.ndarray, estimation: np.ndarray) -> float:
    p = np.ones((graphon.shape[0],)) / graphon.shape[0]
    q = np.ones((estimation.shape[0],)) / estimation.shape[0]
    loss_fun = 'square_loss'
    dw2 = ot.gromov.gromov_wasserstein2(graphon, estimation, p, q, loss_fun, log=False, armijo=False)
    return np.sqrt(dw2)


def mean_square_error(graphon: np.ndarray, estimation: np.ndarray) -> float:
    return np.linalg.norm(graphon - estimation)


def relative_error(graphon: np.ndarray, estimation: np.ndarray) -> float:
    return np.linalg.norm(graphon - estimation) / np.linalg.norm(graphon)


def visualize_graphon(graphon: np.ndarray, save_path: str, title: str = None, with_bar: bool = False):
    fig, ax = plt.subplots()
    plt.imshow(graphon, cmap='plasma', vmin=0.0, vmax=1.0)
    if with_bar:
        plt.colorbar()
    if title is not None:
        ax.set_title(title, fontsize=36)
    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')


def visualize_weighted_graph(adj_mat: np.ndarray, save_path: str, title: str):
    plt.imshow(adj_mat, cmap='Greys')
    plt.title(title)
    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')


def visualize_unweighted_graph(adj_mat: np.ndarray, save_path: str, title: str):
    fig, ax = plt.subplots()
    plt.imshow(adj_mat, cmap='binary')
    ax.set_title(title, fontsize=36)
    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')


def loglikelihood(graph: np.ndarray, graphon: np.ndarray):
    r = graph.shape[0]
    graphon_r = cv2.resize(graphon, dsize=(r, r), interpolation=cv2.INTER_LINEAR)
    graphon_r[graphon_r < 1e-16] = 1e-16
    graphon_r[graphon_r > 1 - 1e-16] = 1 - 1e-16
    plog_p = graph * np.log(graphon_r) + (1 - graph) * np.log(1 - graphon_r)
    loglike = np.mean(plog_p)
    return loglike
