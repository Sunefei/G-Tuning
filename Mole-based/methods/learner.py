import copy
import os

import cv2
import numpy as np
import torch

from skimage.restoration import denoise_tv_chambolle
from typing import List, Tuple


def graph_numpy2tensor(graphs: List[np.ndarray]) -> torch.Tensor:
    """
    Convert a list of np arrays to a pytorch tensor
    :param graphs: [K (N, N) adjacency matrices]
    :return:
        graph_tensor: [K, N, N] tensor
    """
    graph_tensor = np.array(graphs)
    return torch.from_numpy(graph_tensor).float()


def align_graphs(graphs: List[np.ndarray],
                 padding: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """
    Align multiple graphs by sorting their nodes by descending node degrees

    :param graphs: a list of binary adjacency matrices
    :param padding: whether padding graphs to the same size or not
    :return:
        aligned_graphs: a list of aligned adjacency matrices
        normalized_node_degrees: a list of sorted normalized node degrees (as node distributions)
    """
    num_nodes = [graphs[i].shape[0] for i in range(len(graphs))]
    max_num = max(num_nodes)
    min_num = min(num_nodes)

    aligned_graphs = []
    normalized_node_degrees = []
    for i in range(len(graphs)):
        num_i = graphs[i].shape[0]

        node_degree = 0.5 * np.sum(graphs[i], axis=0) + 0.5 * np.sum(graphs[i], axis=1)
        node_degree /= np.sum(node_degree)
        idx = np.argsort(node_degree)  # ascending
        idx = idx[::-1]  # descending

        sorted_node_degree = node_degree[idx]
        sorted_node_degree = sorted_node_degree.reshape(-1, 1)

        sorted_graph = copy.deepcopy(graphs[i])
        sorted_graph = sorted_graph[idx, :]
        sorted_graph = sorted_graph[:, idx]

        if padding:
            # normalized_node_degree = np.ones((max_num, 1)) / max_num
            normalized_node_degree = np.zeros((max_num, 1))
            normalized_node_degree[:num_i, :] = sorted_node_degree
            aligned_graph = np.zeros((max_num, max_num))
            aligned_graph[:num_i, :num_i] = sorted_graph
            normalized_node_degrees.append(normalized_node_degree)
            aligned_graphs.append(aligned_graph)
        else:
            # normalized_node_degree = np.ones(sorted_node_degree.shape) / sorted_node_degree.shape[0]
            # normalized_node_degrees.append(normalized_node_degree)
            normalized_node_degrees.append(sorted_node_degree)
            aligned_graphs.append(sorted_graph)

    return aligned_graphs, normalized_node_degrees, max_num, min_num


def estimate_target_distribution(probs: List[np.ndarray], dim_t: int = None) -> np.ndarray:
    """
    Estimate target distribution via the average of sorted source probabilities
    Args:
        probs: a list of node distributions [(n_s, 1) the distribution of source nodes]
        dim_t: the dimension of target distribution
    Returns:
        p_t: (dim_t, 1) vector representing a distribution
    """
    if dim_t is None:
        dim_t = min([probs[i].shape[0] for i in range(len(probs))])

    p_t = np.zeros((dim_t, 1))
    x_t = np.linspace(0, 1, p_t.shape[0])
    for i in range(len(probs)):
        p_s = probs[i][:, 0]
        p_s = np.sort(p_s)[::-1]
        x_s = np.linspace(0, 1, p_s.shape[0]) 
        if len(x_s) == 0 or len(p_s) == 0:      #TODO:
            print("Error: array of sample points is empty")
            p_s=probs[(i+1)%len(probs)][:, 0]
            x_s = np.linspace(0, 1, p_s.shape[0])
        p_t_i = np.interp(x_t, x_s, p_s) + 1e-3
        p_t[:, 0] += p_t_i

    p_t /= np.sum(p_t)
    return p_t


def estimate_graphon(graphs: List[np.ndarray], method, args, resolution) -> Tuple[np.ndarray, np.ndarray]:
    if method == 'GWB' or method == 'SGWB' or method == 'FGWB' or method == 'SFGWB':
        aligned_graphs, normalized_node_degrees, max_num, min_num = align_graphs(graphs, padding=False)
    else:
        aligned_graphs, normalized_node_degrees, max_num, min_num = align_graphs(graphs, padding=True)

    block_size = int(np.log2(max_num) + 1)
    num_blocks = int(max_num / block_size)
    p_b = estimate_target_distribution(normalized_node_degrees, dim_t=num_blocks)

    if method == 'SAS':
        stepfunc = sorted_smooth(aligned_graphs, h=block_size)
    elif method == 'LG':
        stepfunc = largest_gap(aligned_graphs, k=num_blocks)
    elif method == 'MC':
        stepfunc = matrix_completion(aligned_graphs)
    elif method == 'USVT':
        stepfunc = universal_svd(aligned_graphs, threshold=args.threshold_usvt)
    elif method == 'SBA':
        stepfunc = estimate_blocks_directed(aligned_graphs, threshold=args.threshold_sba)
    elif method == 'SFGWB':
        # print(p_b, np.sum(p_b))
        ws = np.ones((len(aligned_graphs),)) / len(aligned_graphs)
        stepfunc = smoothed_fgw_barycenter(aligned_graphs,
                                           aligned_ps=normalized_node_degrees,
                                           p_b=p_b,
                                           ws=ws,
                                           alpha=args.alpha_est,
                                           inner_iters=args.inner_iters,
                                           outer_iters=args.outer_iters,
                                           beta=args.beta,
                                           gamma=args.gamma)
    elif method == 'SGWB':
        ws = np.ones((len(aligned_graphs),)) / len(aligned_graphs)
        stepfunc = smoothed_fgw_barycenter(aligned_graphs,
                                           aligned_ps=normalized_node_degrees,
                                           p_b=p_b,
                                           ws=ws,
                                           alpha=args.alpha_est,
                                           inner_iters=args.inner_iters,
                                           outer_iters=args.outer_iters,
                                           beta=args.beta,
                                           gamma=1)
    elif method == 'FGWB':
        ws = np.ones((len(aligned_graphs),)) / len(aligned_graphs)
        stepfunc = fgw_barycenter(aligned_graphs,
                                  aligned_ps=normalized_node_degrees,
                                  p_b=p_b,
                                  ws=ws,
                                  inner_iters=args.inner_iters,
                                  outer_iters=args.outer_iters,
                                  beta=args.beta,
                                  gamma=args.gamma)
    elif method == 'GWB':
        ws = np.ones((len(aligned_graphs),)) / len(aligned_graphs)
        stepfunc = fgw_barycenter(aligned_graphs,
                                  aligned_ps=normalized_node_degrees,
                                  p_b=p_b,
                                  ws=ws,
                                  inner_iters=args.inner_iters,
                                  outer_iters=args.outer_iters,
                                  beta=args.beta,
                                  gamma=1)
    else:
        stepfunc = sorted_smooth(aligned_graphs, h=block_size)

    # graphon = cv2.resize(stepfunc, dsize=(args.r, args.r), interpolation=cv2.INTER_LINEAR)
    graphon = cv2.resize(stepfunc, dsize=(resolution, resolution), interpolation=cv2.INTER_LINEAR)
    return stepfunc, graphon


def sorted_smooth(aligned_graphs: List[np.ndarray], h: int) -> np.ndarray:
    """
    Estimate a graphon by a sorting and smoothing method

    Reference:
    S. H. Chan and E. M. Airoldi,
    "A Consistent Histogram Estimator for Exchangeable Graph Models",
    Proceedings of International Conference on Machine Learning, 2014.

    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param h: the block size
    :return: a (k, k) step function and  a (r, r) estimation of graphon
    """
    aligned_graphs = graph_numpy2tensor(aligned_graphs)
    num_graphs = aligned_graphs.size(0)

    if num_graphs > 1:
        sum_graph = torch.mean(aligned_graphs, dim=0, keepdim=True).unsqueeze(0)
    else:
        sum_graph = aligned_graphs.unsqueeze(0)  # (1, 1, N, N)

    # histogram of graph
    kernel = torch.ones(1, 1, h, h) / (h ** 2)
    # print(sum_graph.size(), kernel.size())
    graphon = torch.nn.functional.conv2d(sum_graph, kernel, padding=0, stride=h, bias=None)
    graphon = graphon[0, 0, :, :].numpy()
    # total variation denoising
    graphon = denoise_tv_chambolle(graphon, weight=h)
    return graphon


def largest_gap(aligned_graphs: List[np.ndarray], k: int) -> np.ndarray:
    """
    Estimate a graphon by a stochastic block model based n empirical degrees

    Reference:
    Channarond, Antoine, Jean-Jacques Daudin, and Stéphane Robin.
    "Classification and estimation in the Stochastic Blockmodel based on the empirical degrees."
    Electronic Journal of Statistics 6 (2012): 2574-2601.

    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param k: the number of blocks
    :return: a (r, r) estimation of graphon
    """
    aligned_graphs = graph_numpy2tensor(aligned_graphs)
    num_graphs = aligned_graphs.size(0)

    if num_graphs > 1:
        sum_graph = torch.mean(aligned_graphs, dim=0)
    else:
        sum_graph = aligned_graphs[0, :, :]  # (N, N)

    num_nodes = sum_graph.size(0)

    # sort node degrees
    degree = torch.sum(sum_graph, dim=1)
    sorted_degree = degree / (num_nodes - 1)
    idx = torch.arange(0, num_nodes)

    # find num_blocks-1 largest gap of the node degrees
    diff_degree = sorted_degree[1:] - sorted_degree[:-1]
    _, index = torch.topk(diff_degree, k=k - 1)
    sorted_index, _ = torch.sort(index + 1, descending=False)
    blocks = {}
    for b in range(k):
        if b == 0:
            blocks[b] = idx[0:sorted_index[b]]
        elif b == k - 1:
            blocks[b] = idx[sorted_index[b - 1]:num_nodes]
        else:
            blocks[b] = idx[sorted_index[b - 1]:sorted_index[b]]

    # derive the graphon by stochastic block model
    probability = torch.zeros(k, k)
    graphon = torch.zeros(num_nodes, num_nodes)
    for i in range(k):
        for j in range(k):
            rows = blocks[i]
            cols = blocks[j]
            tmp = sum_graph[rows, :]
            tmp = tmp[:, cols]
            probability[i, j] = torch.sum(tmp) / (rows.size(0) * cols.size(0))
            for r in range(rows.size(0)):
                for c in range(cols.size(0)):
                    graphon[rows[r], cols[c]] = probability[i, j]
    graphon = graphon.numpy()
    return graphon


def universal_svd(aligned_graphs: List[np.ndarray], threshold: float = 2.02) -> np.ndarray:
    """
    Estimate a graphon by universal singular value thresholding.

    Reference:
    Chatterjee, Sourav.
    "Matrix estimation by universal singular value thresholding."
    The Annals of Statistics 43.1 (2015): 177-214.

    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param threshold: the threshold for singular values
    :return: graphon: the estimated (r, r) graphon model
    """
    aligned_graphs = graph_numpy2tensor(aligned_graphs)
    num_graphs = aligned_graphs.size(0)

    if num_graphs > 1:
        sum_graph = torch.mean(aligned_graphs, dim=0)
    else:
        sum_graph = aligned_graphs[0, :, :]  # (N, N)

    num_nodes = sum_graph.size(0)

    u, s, v = torch.svd(sum_graph)
    singular_threshold = threshold * (num_nodes ** 0.5)
    binary_s = torch.lt(s, singular_threshold)
    s[binary_s] = 0
    graphon = u @ torch.diag(s) @ torch.t(v)
    graphon[graphon > 1] = 1
    graphon[graphon < 0] = 0
    graphon = graphon.numpy()
    return graphon


def matrix_completion(aligned_graphs: List[np.ndarray], rank: int = None) -> np.ndarray:
    """
    Estimate the graphon by matrix completion

    Reference:
    Keshavan, Raghunandan H., Andrea Montanari, and Sewoong Oh.
    "Matrix completion from a few entries."
    IEEE transactions on information theory 56.6 (2010): 2980-2998.

    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param rank: the rank of adjacency matrix
    :return: graphon: the estimated graphon model
    """
    aligned_graphs = graph_numpy2tensor(aligned_graphs)
    num_graphs = aligned_graphs.size(0)

    if num_graphs > 1:
        average_graph = torch.mean(aligned_graphs, dim=0)
    else:
        average_graph = aligned_graphs[0, :, :]  # (N, N)

    num_nodes = average_graph.size(0)

    # low-rank approximation via svd
    average_graph = 2 * (average_graph - 0.5)
    if rank is None:
        rank = int(num_nodes / int(np.log(num_nodes)))
        # rank = guess_rank(average_graph)
        # print(rank)
    u, s, v = torch.svd(average_graph)
    graphon = (u[:, :rank] @ torch.diag(s[:rank]) @ torch.t(v[:, :rank]) + 1) / 2
    graphon[graphon > 1] = 1
    graphon[graphon < 0] = 0
    graphon = graphon.numpy()
    return graphon


def guess_rank(matrix: torch.Tensor) -> int:
    """
    A function to guess the rank of a matrix
    :param matrix: a torch.Tensor matrix
    :return:
    """
    n = matrix.size(0)
    m = matrix.size(1)
    epsilon = torch.sum(matrix != 0) / ((n * m) ** 0.5)

    u, s, v = torch.svd(matrix, compute_uv=False)
    max_num = min([100, s.size(0)])
    s = s[:max_num]
    s, _ = torch.sort(s, descending=True)
    diff_s1 = s[:-1] - s[1:]
    diff_s1 = diff_s1 / torch.mean(diff_s1[-10:])
    r1 = torch.zeros(1)
    gamma = 0.05
    while r1.item() <= 0:
        cost = torch.zeros(diff_s1.size(0))
        for i in range(diff_s1.size(0)):
            cost[i] = gamma * torch.max(diff_s1[i:]) + i + 1

        idx = torch.argmin(cost)
        r1 = torch.argmax(idx)
        gamma += 0.05

    cost = torch.zeros(diff_s1.size(0))
    for i in range(diff_s1.size(0)):
        cost[i] = s[i + 1] + ((i + 1) * epsilon ** 0.5) * s[0] / epsilon

    idx = torch.argmin(cost)
    r2 = torch.max(idx)
    return max([r1.item(), r2.item()])


def estimate_blocks_directed(aligned_graphs: List[np.ndarray], threshold: float) -> np.ndarray:
    """
    Estimate a graphon by stochastic block approximation.

    Reference:
    E. M. Airoldi, T. B. Costa, and S. H. Chan,
    "Stochastic blockmodel approximation of a graphon: Theory and consistent estimation",
    Advances in Neural Information Processing Systems 2013.

    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param threshold: the threshold for singular values
    :return: graphon: the estimated (r, r) graphon model
    """
    aligned_graphs = graph_numpy2tensor(aligned_graphs)
    num_graphs = aligned_graphs.size(0)
    num_nodes = aligned_graphs.size(1)
    aligned_graphs = aligned_graphs.permute(1, 2, 0)

    num_half_graphs = int(num_graphs / 2)
    w = 1 / ((num_half_graphs + 1) * (num_nodes - num_half_graphs))
    if num_graphs > 1:
        sum_graph = torch.sum(aligned_graphs, dim=2)
        if num_half_graphs > 1:
            sum_half_graph1 = torch.sum(aligned_graphs[:, :, :num_half_graphs], dim=2)
        else:
            sum_half_graph1 = aligned_graphs[:, :, 0]

        if num_graphs - num_half_graphs > 1:
            sum_half_graph2 = torch.sum(aligned_graphs[:, :, num_half_graphs:], dim=2)
        else:
            sum_half_graph2 = aligned_graphs[:, :, -1]
    else:
        sum_half_graph1 = aligned_graphs[:, :, 0]
        sum_half_graph2 = aligned_graphs[:, :, 0]
        sum_graph = aligned_graphs[:, :, 0]

    pivot_idx = [0]
    blocks = dict()
    blocks[0] = [pivot_idx[0]]
    not_assigned_idx = list(range(num_nodes))
    not_assigned_idx.remove(pivot_idx[0])
    while len(not_assigned_idx) > 0:
        if len(not_assigned_idx) == 1:
            i = not_assigned_idx[0]
        else:
            idx = np.random.permutation(len(not_assigned_idx))
            i = not_assigned_idx[idx[0]]
        not_assigned_idx.remove(i)

        dhat = torch.zeros(len(pivot_idx))
        for j in range(len(pivot_idx)):
            bj = pivot_idx[j]

            set_idx = list(range(num_nodes))
            set_idx.remove(i)
            set_idx.remove(bj)

            term1 = torch.sum(w * sum_half_graph1[i, set_idx] * sum_half_graph2[i, set_idx])
            term2 = torch.sum(w * sum_half_graph1[bj, set_idx] * sum_half_graph2[bj, set_idx])
            term3 = torch.sum(w * sum_half_graph1[i, set_idx] * sum_half_graph2[bj, set_idx])
            term4 = torch.sum(w * sum_half_graph1[bj, set_idx] * sum_half_graph2[i, set_idx])

            term5 = torch.sum(w * sum_half_graph1[set_idx, i] * sum_half_graph2[set_idx, i])
            term6 = torch.sum(w * sum_half_graph1[set_idx, bj] * sum_half_graph2[set_idx, bj])
            term7 = torch.sum(w * sum_half_graph1[set_idx, i] * sum_half_graph2[set_idx, bj])
            term8 = torch.sum(w * sum_half_graph1[set_idx, bj] * sum_half_graph2[set_idx, i])

            dhat[j] = (0.5 * (torch.abs(term1 + term2 - term3 - term4) +
                              torch.abs(term5 + term6 - term7 - term8)) / len(set_idx)) ** 0.5

        if dhat.size(0) == 1:
            value = dhat[0]
            idx = 0
        else:
            value, idx = torch.min(dhat, dim=0)
            idx = idx.item()

        if value < threshold:
            blocks[idx].append(i)
        else:
            blocks[len(pivot_idx)] = [i]
            pivot_idx.append(i)

    num_blocks = len(blocks)
    for key in blocks.keys():
        tmp = np.array(blocks[key])
        blocks[key] = torch.from_numpy(tmp).long()
    print(num_blocks)

    # # derive the graphon by stochastic block model
    probability = torch.zeros(num_blocks, num_blocks)
    graphon = torch.zeros(num_nodes, num_nodes)
    for i in range(num_blocks):
        for j in range(num_blocks):
            rows = blocks[i]
            cols = blocks[j]
            tmp = sum_graph[rows, :]
            tmp = tmp[:, cols]
            probability[i, j] = torch.sum(tmp) / (num_graphs * rows.size(0) * cols.size(0))
            for r in range(rows.size(0)):
                for c in range(cols.size(0)):
                    graphon[rows[r], cols[c]] = probability[i, j]

    graphon = graphon.numpy()
    return graphon


def smoothed_fgw_barycenter(aligned_graphs: List[np.ndarray],
                            aligned_ps: List[np.ndarray],
                            p_b: np.ndarray,
                            ws: np.ndarray,
                            alpha: float,
                            inner_iters: int,
                            outer_iters: int,
                            beta: float,
                            gamma: float) -> np.ndarray:
    """
    Calculate smoothed Gromov-Wasserstein barycenter

    :param aligned_graphs: a list of (Ni, Ni) adjacency matrices
    :param aligned_ps: a list of (Ni, 1) distributions
    :param p_b: (Nb, 1) distribution
    :param ws: (K, ) weights
    :param alpha: the weight of smoothness regularizer
    :param inner_iters: the number of sinkhorn iterations
    :param outer_iters: the number of barycenter iterations
    :param beta: the weight of proximal term
    :param gamma: the weight of gw term
    :return:
    """
    nb = p_b.shape[0]
    try:
        dmat = 2 * np.concatenate((np.zeros((nb - 2, 1)), np.eye(nb - 2), np.zeros((nb - 2, 1))), axis=1) - \
               np.concatenate((np.zeros((nb - 2, 2)), np.eye(nb - 2)), axis=1) - \
               np.concatenate((np.eye(nb - 2), np.zeros((nb - 2, 2))), axis=1)  # (nb - 2, nb)
    # assert p_b.shape[0] >= 2
    except ValueError:
        print('p_b', p_b.shape[0], p_b)
        print('p_b', p_b.shape[0], p_b)

    lmat = alpha * (dmat.T @ dmat)
    us, ss, _ = np.linalg.svd(lmat)

    numerator = ss ** 2 + p_b[:, 0] ** 2  # + 1e-16
    lmat2 = us @ np.diag(ss / numerator) @ us.T
    pmat2 = np.diag(p_b[:, 0] / numerator)

    cost_ps = []
    trans = []
    priors = []
    for p in aligned_ps:
        cost_p = p_b ** 2 + (p ** 2).T - 2 * (p_b @ p.T)
        cost_p /= np.max(cost_p)
        cost_ps.append(cost_p)
        tran = proximal_ot(cost_p, p_b, p, iters=inner_iters, beta=beta)
        trans.append(tran)
        priors.append(tran + 1e-16)

    barycenter = None
    for o in range(outer_iters):
        # update smoothed barycenter
        averaged_graph = averaging_graphs(aligned_graphs, trans, ws)
        barycenter = lmat2 @ averaged_graph @ lmat2.T + pmat2 @ averaged_graph @ pmat2.T
        # update optimal transports
        for i in range(len(aligned_graphs)):
            cost_i = gw_cost(barycenter, aligned_graphs[i], trans[i], p_b, aligned_ps[i])
            cost_i = gamma * cost_i + (1 - gamma) * cost_ps[i]
            cost_i /= np.max(cost_i)
            trans[i] = proximal_ot(cost_i, p_b, aligned_ps[i], iters=inner_iters, beta=beta, prior=priors[i])

    barycenter[barycenter > 1] = 1
    barycenter[barycenter < 0] = 0
    return barycenter


def fgw_barycenter(aligned_graphs: List[np.ndarray],
                   aligned_ps: List[np.ndarray],
                   p_b: np.ndarray,
                   ws: np.ndarray,
                   inner_iters: int,
                   outer_iters: int,
                   beta: float,
                   gamma: float) -> np.ndarray:
    """
    Calculate smoothed Gromov-Wasserstein barycenter

    :param aligned_graphs: a list of (Ni, Ni) adjacency matrices
    :param aligned_ps: a list of (Ni, 1) distributions
    :param p_b: (Nb, 1) distribution
    :param ws: (K, ) weights
    :param inner_iters: the number of sinkhorn iterations
    :param outer_iters: the number of barycenter iterations
    :param beta: the weight of proximal term
    :param gamma: the weight of gw term
    :return:
    """
    cost_ps = []
    trans = []
    priors = []
    for p in aligned_ps:
        cost_p = p_b ** 2 + (p ** 2).T - 2 * (p_b @ p.T)
        cost_p /= np.max(cost_p)
        cost_ps.append(cost_p)
        tran = proximal_ot(cost_p, p_b, p, iters=inner_iters, beta=beta)
        trans.append(tran)
        priors.append(tran + 1e-16)

    barycenter = None
    for o in range(outer_iters):
        # update smoothed barycenter
        averaged_graph = averaging_graphs(aligned_graphs, trans, ws)
        barycenter = averaged_graph / (p_b @ p_b.T)

        # update optimal transports
        for i in range(len(aligned_graphs)):
            cost_i = gw_cost(barycenter, aligned_graphs[i], trans[i], p_b, aligned_ps[i])
            cost_i = gamma * cost_i + (1 - gamma) * cost_ps[i]
            cost_i /= np.max(cost_i)
            trans[i] = proximal_ot(cost_i, p_b, aligned_ps[i], iters=inner_iters, beta=beta, prior=priors[i])

    barycenter[barycenter > 1] = 1
    barycenter[barycenter < 0] = 0
    return barycenter


def averaging_graphs(aligned_graphs: List[np.ndarray], trans: List[np.ndarray], ws: np.ndarray) -> np.ndarray:
    """
    sum_k w_k * (Tk @ Gk @ Tk')
    :param aligned_graphs: a list of (Ni, Ni) adjacency matrices
    :param trans: a list of (Nb, Ni) transport matrices
    :param ws: (K, ) weights
    :return: averaged_graph: a (Nb, Nb) adjacency matrix
    """
    averaged_graph = 0
    for k in range(ws.shape[0]):
        averaged_graph += ws[k] * (trans[k] @ aligned_graphs[k] @ trans[k].T)
    return averaged_graph


def proximal_ot(cost: np.ndarray,
                p1: np.ndarray,
                p2: np.ndarray,
                iters: int,
                beta: float,
                error_bound: float = 1e-10,
                prior: np.ndarray = None) -> np.ndarray:
    """
    min_{T in Pi(p1, p2)} <cost, T> + beta * KL(T | prior)

    :param cost: (n1, n2) cost matrix
    :param p1: (n1, 1) source distribution
    :param p2: (n2, 1) target distribution
    :param iters: the number of Sinkhorn iterations
    :param beta: the weight of proximal term
    :param error_bound: the relative error bound
    :param prior: the prior of optimal transport matrix T, if it is None, the proximal term degrades to Entropy term
    :return:
        trans: a (n1, n2) optimal transport matrix
    """
    if prior is not None:
        kernel = np.exp(-cost / beta) * prior
    else:
        kernel = np.exp(-cost / beta)

    relative_error = np.inf
    a = np.ones(p1.shape) / p1.shape[0]
    b = []
    i = 0

    while relative_error > error_bound and i < iters:
        b = p2 / (np.matmul(kernel.T, a))
        a_new = p1 / np.matmul(kernel, b)
        relative_error = np.sum(np.abs(a_new - a)) / np.sum(np.abs(a))
        a = copy.deepcopy(a_new)
        i += 1
    trans = np.matmul(a, b.T) * kernel
    return trans


def node_cost_st(cost_s: np.ndarray, cost_t: np.ndarray, p_s: np.ndarray, p_t: np.ndarray) -> np.ndarray:
    """
    Calculate invariant cost between the nodes in different graphs based on learned optimal transport
    Args:
        cost_s: (n_s, n_s) array, the cost matrix of source graph
        cost_t: (n_t, n_t) array, the cost matrix of target graph
        p_s: (n_s, 1) array, the distribution of source nodes
        p_t: (n_t, 1) array, the distribution of target nodes
    Returns:
        cost_st: (n_s, n_t) array, the estimated invariant cost between the nodes in two graphs
    """
    n_s = cost_s.shape[0]
    n_t = cost_t.shape[0]
    f1_st = np.repeat((cost_s ** 2) @ p_s, n_t, axis=1)
    f2_st = np.repeat(((cost_t ** 2) @ p_t).T, n_s, axis=0)
    cost_st = f1_st + f2_st
    return cost_st


def gw_cost(cost_s: np.ndarray, cost_t: np.ndarray, trans: np.ndarray, p_s: np.ndarray, p_t: np.ndarray) -> np.ndarray:
    """
    Calculate the cost between the nodes in different graphs based on learned optimal transport
    Args:
        cost_s: (n_s, n_s) array, the cost matrix of source graph
        cost_t: (n_t, n_t) array, the cost matrix of target graph
        trans: (n_s, n_t) array, the learned optimal transport between two graphs
        p_s: (n_s, 1) array, the distribution of source nodes
        p_t: (n_t, 1) array, the distribution of target nodes
    Returns:
        cost: (n_s, n_t) array, the estimated cost between the nodes in two graphs
    """
    cost_st = node_cost_st(cost_s, cost_t, p_s, p_t)
    return cost_st - 2 * (cost_s @ trans @ cost_t.T)
