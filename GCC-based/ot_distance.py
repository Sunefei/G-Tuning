import torch


def distance_tensor(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :param p:
    :return: [R, C, D] distance matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.abs(x_col - y_row) ** p
    return distance


def distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :param p:
    :return: [R, C] distance matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.sum((torch.abs(x_col - y_row)) ** p, 2)
    return distance


def sum_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor):
    """
    Returns the matrix of "x_i + y_j".
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :return: [R, C, D] sum matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    return x_col + y_row


def cost_mat(cost_s: torch.Tensor, cost_t: torch.Tensor, tran: torch.Tensor) -> torch.Tensor:
    """
    Implement cost_mat for Gromov-Wasserstein discrepancy (GWD)
    Suppose the loss function in GWD is |a-b|^2 = a^2 - 2ab + b^2. We have:
    f1(a) = a^2,
    f2(b) = b^2,
    h1(a) = a,
    h2(b) = 2b
    When the loss function can be represented in the following format: loss(a, b) = f1(a) + f2(b) - h1(a)h2(b), we have
    cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
    cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T
    Args:
        cost_s: (ns, ns) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        cost_t: (nt, nt) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        tran: (ns, nt) matrix (torch tensor), representing the optimal transport from source to target domain.
    Returns:
        cost: (ns, nt) matrix (torch tensor), representing the cost matrix conditioned on current optimal transport
    """
    f1_st = torch.sum(cost_s ** 2, dim=1, keepdim=True) / cost_s.shape[0]
    f2_st = torch.sum(cost_t ** 2, dim=1, keepdim=True) / cost_t.shape[0]
    tmp = torch.sum(sum_matrix(f1_st, f2_st), dim=2)
    cost = tmp - 2 * cost_s @ tran @ torch.t(cost_t)
    return cost


def fgw_distance(cost_posterior, cost_prior, args):
    # cost_pp = distance_matrix(attr1, attr2)
    ns = cost_posterior.size(0)
    nt = cost_prior.size(0)
    p_s = torch.ones(ns, 1, dtype=torch.double) / ns
    p_t = torch.ones(nt, 1, dtype=torch.double) / nt
    tran = torch.ones(ns, nt, dtype=torch.double) / (ns * nt)
    p_s = p_s.to(cost_posterior.device)
    p_t = p_t.to(cost_posterior.device)
    tran = tran.to(cost_posterior.device)
    dual = (torch.ones(ns, 1, dtype=torch.double) / ns).to(cost_posterior.device)
    for m in range(args.n_iter):
        # cost = args.beta_gw * cost_mat(cost_posterior, cost_prior, tran) + (1 - args.beta_gw) * cost_pp
        cost = args.beta_gw * cost_mat(cost_posterior, cost_prior, tran) + (1 - args.beta_gw)
        # print(cost.shape)
        kernel = torch.exp(-cost / torch.max(torch.abs(cost))) * tran
        b = p_t / (torch.t(kernel) @ dual)
        # dual = p_s / (kernel @ b)
        for i in range(args.n_sinkhorn):
            dual = p_s / (kernel @ b)
            b = p_t / (torch.t(kernel) @ dual)
        tran = (dual @ torch.t(b)) * kernel
    if torch.isnan(tran).sum() > 0:
        tran = (torch.ones(ns, nt) / (ns * nt)).to(cost_posterior.device)
    # cost = args.beta_gw * cost_mat(cost_posterior, cost_prior, tran.detach().data) + (1 - args.beta_gw) * cost_pp
    cost = args.beta_gw * cost_mat(cost_posterior, cost_prior, tran.detach().data) + (1 - args.beta_gw)
    d_fgw = (cost * tran.detach().data).sum()
    return d_fgw


def sliced_fgw_distance(posterior_samples, prior_samples, num_projections=50, p=2, beta=0.1):
    # derive latent space dimension size from random samples drawn from latent prior distribution
    embedding_dim = prior_samples.size(1)
    # generate random projections in latent space
    projections = torch.randn(size=(embedding_dim, num_projections)).to(posterior_samples.device)
    projections /= (projections ** 2).sum(0).sqrt().unsqueeze(0)
    # calculate projections through the encoded samples
    posterior_projections = posterior_samples.matmul(projections)  # batch size x #projections
    prior_projections = prior_samples.matmul(projections)  # batch size x #projections
    posterior_projections = torch.sort(posterior_projections, dim=0)[0]
    prior_projections1 = torch.sort(prior_projections, dim=0)[0]
    prior_projections2 = torch.sort(prior_projections, dim=0, descending=True)[0]
    posterior_diff = distance_tensor(posterior_projections, posterior_projections, p=p)
    prior_diff1 = distance_tensor(prior_projections1, prior_projections1, p=p)
    prior_diff2 = distance_tensor(prior_projections2, prior_projections2, p=p)
    # print(posterior_projections.size(), prior_projections1.size())
    # print(posterior_diff.size(), prior_diff1.size())
    w1 = torch.sum((posterior_projections - prior_projections1) ** p, dim=0)
    w2 = torch.sum((posterior_projections - prior_projections2) ** p, dim=0)
    # print(w1.size(), torch.sum(w1))
    gw1 = torch.mean(torch.mean((posterior_diff - prior_diff1) ** p, dim=0), dim=0)
    gw2 = torch.mean(torch.mean((posterior_diff - prior_diff2) ** p, dim=0), dim=0)
    # print(gw1.size(), torch.sum(gw1))
    fgw1 = (1 - beta) * w1 + beta * gw1
    fgw2 = (1 - beta) * w2 + beta * gw2
    return torch.sum(torch.min(fgw1, fgw2))
