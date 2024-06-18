import torch
import torch.nn as nn
import torch_geometric.utils as PyG_utils
from ftlib.finetune.gtot import GTOT



class GTOTRegularization(nn.Module):
    r"""
       GTOT regularization for finetuning
    Shape:
        - Output: scalar.

    """

    def __init__(self, order=1, args=None):
        super(GTOTRegularization, self).__init__()
        self.Gtot = GTOT(eps=0.1, thresh=0.1, max_iter=100, reduction=None)

        self.args = args
        self.order = order
        self.M = 0.05

    def sensible_normalize(self, C, mask=None):
        d_max = torch.max(C.abs().view(C.shape[0], -1), -1)[0]
        d_max = d_max.unsqueeze(1).unsqueeze(2)
        d_max[d_max == 0] = 1e9
        C = (C / d_max)
        if torch.isnan(C.sum()):
            raise
        return C

    def got_dist(self, f_s, f_t, A=None, mask=None):
        ## cosine distance
        '''if there is batch graph, the mask should be added to the cos_distance to make the dist to be zeros when the vector is padding.'''
        cos_distance = self.Gtot.cost_matrix_batch_torch(f_s.transpose(2, 1), f_t.transpose(2, 1), mask=mask)
        cos_distance = cos_distance.transpose(1, 2)

        # normalize cost matrix

        ## D= max(D_{cos},threshold)
        threshod = False
        penalty = 50
        if threshod:
            beta = 0.1
            min_score = cos_distance.min()
            max_score = cos_distance.max()
            threshold = min_score + beta * (max_score - min_score)
            cos_dist = torch.nn.functional.relu(cos_distance - threshold)
        else:
            cos_dist = cos_distance

            self.sensible_normalize(cos_dist, mask=mask)


        # use different A^{order} as mask matrix
        if self.order == 0:
            A = torch.stack([torch.diag(mask_i.type_as(A)) for mask_i in mask])
        elif self.order == 1:
            A = A

        elif self.order >= 9:
            A = self.Gtot.mask_matrix
        elif self.order > 0:  # order =1
            A0 = A
            for i in range(self.order - 1):
                A = A.bmm(A0)
            A = torch.sign(A)
        else:
            raise

        if A is not None:
            A = self.Gtot.mask_matrix * A
            ## find the isolated Points in A , which will make the row (and symmetrical colum )of the cost matrix
            # become zero. This causes numerical overflow
            row, col = torch.nonzero((A.sum(-1) + (~mask)) == 0, as_tuple=True)
            mask[row, col] = False
        ## Masked OT with A^{order} as mask matrix
        wd, P, C = self.Gtot(x=f_s, y=f_t, A=A, C=cos_dist, mask=mask)
        twd = .5 * torch.mean(wd)

        return twd

    def forward(self, layer_outputs_source, layer_outputs_target, *argv):
        '''
        Args:
            layer_outputs_source:
            layer_outputs_target:
            batch: batch is a column vector which maps each node to its respective graph in the batch

        Returns:

        '''

        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")
        output = 0.0

        for i, (fm_src, fm_tgt) in enumerate(zip(layer_outputs_source.values(), layer_outputs_target.values())):


            b_nodes_fea_s, b_mask_s = PyG_utils.to_dense_batch(x=fm_src.detach(), batch=batch)
            b_nodes_fea_t, b_mask_t = PyG_utils.to_dense_batch(x=fm_tgt, batch=batch)

            edge_index, edge_weight = PyG_utils.add_remaining_self_loops(edge_index, num_nodes=fm_tgt.size(0))
            b_A = PyG_utils.to_dense_adj(edge_index, batch=batch)
            ##  GTOT distance
            distance = self.got_dist(f_s=b_nodes_fea_s.detach(), f_t=b_nodes_fea_t, A=b_A, mask=b_mask_t)


            output = output + torch.sum(distance)

        return output
