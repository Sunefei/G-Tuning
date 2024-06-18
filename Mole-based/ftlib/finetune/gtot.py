import numpy as np
import torch
import torch.nn as nn


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
# Adapted from https://github.com/gpeyre/SinkhornAutoDiff/blob/master/sinkhorn_pointcloud.py
class GTOT(nn.Module):
    r"""
        GTOT implementation.
    """

    def __init__(self, eps=0.1, thresh=0.1, max_iter=100, reduction='none'):
        super(GTOT, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.thresh = thresh
        self.mask_matrix = None

    def marginal_prob_unform(self, N_s=None, N_t=None, mask=None, ):
        if mask is not None:
            mask = mask.float()
            # uniform distribution
            mask_mean = (1 / mask.sum(1)).unsqueeze(1)
            mu = mask * mask_mean  # 1/n
            # mu = mu.unsqueeze(2)
        else:
            mu = torch.ones(self.bs, N_s) / N_s
        nu = mu.clone().detach()
        return mu, nu

    def forward(self, x, y, C=None, A=None, mask=None):
        # The Sinkhorn algorithm takes as input three variables :
        if C is None:
            C = self._cost_matrix(x, y)  # Wasserstein cost function
            C = C / C.max()
        if A is not None:
            if A.type().startswith('torch.cuda.sparse'):
                self.sparse = True
                C = A.to_dense() * C
            else:
                self.sparse = False
                C = A * C
        N_s = x.shape[-2]
        N_t = y.shape[-2]
        if x.dim() == 2:
            self.bs = 1
        else:
            self.bs = x.shape[0]

        # both marginals are fixed with equal weights
        if mask is None:
            mu = torch.empty(self.bs, N_s, dtype=torch.float, device=C.device,
                             requires_grad=False).fill_(1.0 / N_s).squeeze()
            nu = torch.empty(self.bs, N_t, dtype=torch.float, device=C.device,
                             requires_grad=False).fill_(1.0 / N_t).squeeze()
        else:
            mu, nu = self.marginal_prob_unform(N_s=N_s, N_t=N_t, mask=mask)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = self.thresh

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update

            if mask is None:
                u = self.eps * (torch.log(mu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A), dim=-1)) + u
                v = self.eps * (
                        torch.log(nu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A).transpose(-2, -1), dim=-1)) + v
            else:
                u = self.eps * (torch.log(mu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A), dim=-1)) + u
                u = mask * u
                v = self.eps * (
                        torch.log(nu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A).transpose(-2, -1), dim=-1)) + v
                v = mask * v

            # err = (u - u1).abs().sum(-1).mean()
            err = (u - u1).abs().sum(-1).max()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v

        pi = self.exp_M(C, U, V, A=A)

        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()
        if torch.isnan(cost.sum()):
            print(pi)
            raise
        return cost, pi, C

    def M(self, C, u, v, A=None):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"


        S = (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps
        return S

    def exp_M(self, C, u, v, A=None):
        if A is not None:
            if self.sparse:
                a = A.to_dense()
                S = torch.exp(self.M(C, u, v)).masked_fill(mask = (1-a).to(torch.bool),value=0)
            else:
                S = torch.exp(self.M(C, u, v)).masked_fill(mask = (1-A).to(torch.bool),value=0)

            return S
        elif self.mask_matrix is not None:
            return self.mask_matrix * torch.exp(self.M(C, u, v))
        else:
            return torch.exp(self.M(C, u, v))

    def log_sum(self, input_tensor, dim=-1, mask=None):
        s = torch.sum(input_tensor, dim=dim)
        out = torch.log(1e-8 + s)
        if torch.isnan(out.sum()):
            raise
        if mask is not None:
            out = mask * out
        return out

    def cost_matrix_batch_torch(self, x, y, mask=None):
        "Returns the cosine distance batchwise"
        # x is the source feature: bs * d * m
        # y is the target feature: bs * d * m
        # return: bs * n * m
        # print(x.size())
        bs = list(x.size())[0]
        D = x.size(1)
        assert (x.size(1) == y.size(1))
        x = x.contiguous().view(bs, D, -1)  # bs * d * m
        x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
        cos_dis = torch.bmm(torch.transpose(x, 1, 2), y)  # .transpose(1,2)
        cos_dis = 1 - cos_dis  # to minimize this value
        # cos_dis = - cos_dis
        if mask is not None:
            mask0 = mask.unsqueeze(2).clone().float()
            self.mask_matrix = torch.bmm(mask0, (mask0.transpose(2, 1)))  # torch.ones_like(C)
            cos_dis = cos_dis * self.mask_matrix
        if torch.isnan(cos_dis.sum()):
            raise
        return cos_dis.transpose(2, 1)

    def cost_matrix_torch(self, x, y):
        "Returns the cosine distance"
        # x is the image embedding
        # y is the text embedding
        D = x.size(0)
        x = x.view(D, -1)
        assert (x.size(0) == y.size(0))
        x = x.div(torch.norm(x, p=2, dim=0, keepdim=True) + 1e-12)
        y = y.div(torch.norm(y, p=2, dim=0, keepdim=True) + 1e-12)
        cos_dis = torch.mm(torch.transpose(y, 0, 1), x)  # .t()
        cos_dis = 1 - cos_dis  # to minimize this value
        return cos_dis

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1



if __name__ == '__main__':
    def random_A(n, dense_rate=0.5):
        d = n
        rand_mat = torch.rand(n, d)
        k = round(dense_rate * d)  # For the general case change 0.25 to the percentage you need
        k_th_quant = torch.topk(rand_mat, k, largest=False)[0][:, -1:]
        bool_tensor = rand_mat <= k_th_quant
        desired_tensor = torch.where(bool_tensor, torch.tensor(1), torch.tensor(0))
        return desired_tensor


    n = 5
    batch_size = 2
    a = np.array([[[i, 0] for i in range(n)] for b in range(batch_size)])
    b = np.array([[[i, b + 1] for i in range(n)] for b in range(batch_size)])
    # Wrap with torch tensors
    x = torch.tensor(a, dtype=torch.float)
    y = torch.tensor(b, dtype=torch.float)
    x = x.cuda()
    y = y.cuda()
    for i in np.array(range(2, 11)) * 0.1:
        dense_rate = i

        print('Adjacent matrix dense_rate', dense_rate, end=' ')
        A = random_A(n, dense_rate=dense_rate)
        A[range(A.shape[0]), range(A.shape[0])] = 1
        # A  = torch.eye(n)
        print(A)
        A = A.repeat(batch_size, 1, 1)
        A = A.cuda().to_sparse()
        # A=None
        sinkhorn = GTOT(eps=0.1, max_iter=100, reduction=None)
        dist, P, C = sinkhorn(x, y, A=A)
        print("Sinkhorn distances: ", dist)
