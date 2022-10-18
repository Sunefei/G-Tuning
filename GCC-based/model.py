import random

from ot_distance import sliced_fgw_distance, fgw_distance
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax, subgraph
import torch_geometric.utils as PyG_utils
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
import torch.nn as nn
# from torch_geometric.nn.conv import GATConv
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

from torch_geometric.datasets import TUDataset
from abc import ABC
from gcc.models.gin import UnsupervisedGIN

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        try:  # PyG 1.6.
            return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)
        except:  # PyG 1.0.3
            return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        # edge_index, edge_weight = add_self_loops(edge_index, num_nodes=x.size(0)) pyg 1.6
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        try:
            norm = self.norm(edge_index, x.size(0), x.dtype)
        except:
            norm = self.norm(edge_index[0], x.size(0), x.dtype)

        x = self.linear(x)
        try:  # PyG 1.6.
            return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)
        except:  # PyG 1.0.3
            return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings, norm=norm)
        # return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)
        # return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_ind = add_self_loops(edge_index, num_nodes=x.size(0))
        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        edge_ind = edge_ind[0]
        print("edge_index", edge_ind.dtype, edge_ind.shape)
        return self.propagate(self.aggr, edge_ind, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.linear(x)

        try:  # PyG 1.6.
            return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)
        except:  # PyG 1.0.3
            return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.gnn_type = gnn_type
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim, emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        batch = None
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        elif len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):

            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self, num_layer, emb_dim, num_tasks, JK="last", drop_ratio=0, graph_pooling="mean", gnn_type="gin",
                 backbone=None, args=None):
        '''
        backbone is gnn default
        '''
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.emb_f = None
        self.gnn_type = gnn_type
        self.args = args

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        if backbone is None:
            self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)
        else:
            self.gnn = backbone
        # self.backbone = self.gnn

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        # self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        model = torch.load(model_file, map_location='cpu')
        self.gnn.load_state_dict(model)  # self.args.device))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)
        self.emb_f = self.pool(node_representation, batch)
        return self.graph_pred_linear(self.emb_f)

    def get_bottleneck(self):
        return self.emb_f


class GraphonEncoder(torch.nn.Module):
    def __init__(self, feature_length, hidden_size, out_size):
        super(GraphonEncoder, self).__init__()
        self.feature_length, self.hidden_size, self.out_size = feature_length, hidden_size, out_size
        self.fc1 = torch.nn.Linear(feature_length, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = x.view(-1, self.feature_length)
        # print(x, x.shape)
        # print(edge_index, edge_index.shape)
        x = F.dropout(x, p=0.9, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.9, training=self.training)
        x = self.fc2(x)
        return x


def sampling_gaussian(mu, logvar, num_sample):
    std = torch.exp(0.5 * logvar)
    samples = None
    for i in range(num_sample):
        eps = torch.randn_like(std)
        if i == 0:
            samples = mu + eps * std
        else:
            samples = torch.cat((samples, mu + eps * std), dim=0)
    return samples


def sampling_gmm(mu, logvar, num_sample):
    std = torch.exp(0.5 * logvar)
    n = int(num_sample / mu.size(0)) + 1
    samples = None
    for i in range(n):
        eps = torch.randn_like(std)
        if i == 0:
            samples = mu + eps * std
        else:
            samples = torch.cat((samples, mu + eps * std), dim=0)
    return samples[:num_sample, :]


class Prior(nn.Module, ABC):
    def __init__(self, data_size: list, prior_type: str = 'gmm'):
        super(Prior, self).__init__()
        # data_size = [num_component, z_dim]
        self.data_size = data_size
        self.number_components = data_size[0]
        self.output_size = data_size[1]
        self.prior_type = prior_type
        if self.prior_type == 'gmm':
            self.mu = nn.Parameter(torch.randn(data_size), requires_grad=True)
            self.logvar = nn.Parameter(torch.randn(data_size), requires_grad=True)
        else:
            self.mu = nn.Parameter(torch.zeros(1, self.output_size), requires_grad=False)
            self.logvar = nn.Parameter(torch.ones(1, self.output_size), requires_grad=False)

    def forward(self):
        return self.mu, self.logvar

    def sampling(self, num_sample):
        if self.prior_type == 'gmm':
            return sampling_gmm(self.mu, self.logvar, num_sample)
        else:
            return sampling_gaussian(self.mu, self.logvar, num_sample)


class GraphonNewEncoder(torch.nn.Module):
    def __init__(self, feature_length, hidden_size, out_size, encoder_type):
        super(GraphonNewEncoder, self).__init__()
        self.feature_length, self.hidden_size, self.out_size = feature_length, hidden_size, out_size
        self.encoder_type = encoder_type
        self.fc1 = torch.nn.Linear(feature_length, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, out_size)
        # self.gnn = GNN(2, feature_length, "last", 0.2, gnn_type='gin')
        self.gnns_en = torch.nn.ModuleList()
        self.gnn = UnsupervisedGIN(
            num_layers=2,
            num_mlp_layers=1,
            input_dim=feature_length,
            hidden_dim=hidden_size,
            output_dim=out_size,
            final_dropout=0.5,
            learn_eps=False,
            graph_pooling_type="sum",
            neighbor_pooling_type="sum",
            use_selayer=False,
        )

        for layer in range(2):
            if encoder_type == "gin":
                self.gnns_en.append(GINConv(feature_length, aggr="add"))
            elif encoder_type == "gcn":
                self.gnns_en.append(GCNConv(feature_length))
            elif encoder_type == "gat":
                self.gnns_en.append(GATConv(feature_length, feature_length))
            elif encoder_type == "graphsage":
                self.gnns_en.append(GraphSAGEConv(feature_length))

        self.fc3 = torch.nn.Linear(feature_length, out_size)
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(2):
            self.batch_norms.append(torch.nn.BatchNorm1d(feature_length))

    def forward(self, x, g):
        # node_representation = self.gnn(x, edge_index.long(), None)
        if self.encoder_type != 'mlp':
            #     h_list = [x, ]
            #     for layer in range(2):
            #         inp = h_list[layer]
            #         h = self.gnns_en[layer](inp, edge_index, edge_attr)
            #         h = self.batch_norms[layer](h)
            #         # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            #         if layer == 1:
            #             # remove relu for the last layer
            #             h = F.dropout(h, 0.2, training=self.training)
            #         else:
            #             h = F.dropout(F.relu(h), 0.2, training=self.training)
            #         h_list.append(h)
            #
            #     x = h_list[-1]

            x, all_outputs = self.gnn(g, x, None)
            # x = global_mean_pool(x, batch)
            # x = self.fc3(x)
        else:
            x = x.view(-1, self.feature_length)
            # print(x, x.shape)
            # print(edge_index, edge_index.shape)
            x = F.dropout(x, p=0.9, training=self.training)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, p=0.9, training=self.training)

            pooled_h = self.pool(g, x)
            x = self.fc2(pooled_h)
        return x


class GraphonFactorization(torch.nn.Module, ABC):
    def __init__(self, num_factors: int, graphs: TUDataset, seed: int, args, node_type: str = 'categorical'):
        """
        A basic graphon model based on Fourier transformation
        Args:
            num_factors: the number of sin/cos bases for one graphon
            graphs: the graphs used as the prior of the model
            seed: random seed
            node_type: 'binary', 'categorical' and 'continuous'
        """
        super(GraphonFactorization, self).__init__()
        self.num_factors = num_factors
        self.node_type = node_type
        self.factors_graphon = nn.ParameterList()
        self.factors_signal = nn.ParameterList()
        self.num_partitions = []
        indices = list(range(len(graphs)))
        random.seed(seed)
        random.shuffle(indices)
        # indices = np.random.RandomState(seed).permutation(len(graphs))
        for c in range(self.num_factors):
            sample = graphs[indices[c]]
            adj = torch.sparse_coo_tensor(sample.edge_index,
                                          torch.ones(sample.edge_index.shape[1]),
                                          size=[sample.num_nodes, sample.num_nodes])
            adj = adj.to_dense()
            # print(adj.shape)
            if len(adj.shape) > 2:
                adj = adj.sum(2)
            # attribute = sample.x
            degrees = torch.sum(adj, dim=1)
            idx = torch.argsort(degrees)
            # print(idx.shape)
            adj = adj[idx, :][:, idx]
            # attribute = attribute[idx, :]
            num_partitions = adj.shape[0]
            graphon = nn.Parameter(data=(adj - 0.5), requires_grad=True)
            # if self.node_type == "binary" or "categorical":
            #     signal = nn.Parameter(data=(attribute - 0.5), requires_grad=True)
            # else:
            #     signal = nn.Parameter(data=attribute, requires_grad=True)
            self.num_partitions.append(num_partitions)
            self.factors_graphon.append(graphon)
            # self.factors_signal.append(signal)
        # self.dim = self.factors_signal[0].shape[1]
        self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()
        self.softmax0 = nn.Softmax(dim=0)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)

        # 这里三层softmax是？

        self.batch_size = args.batch_size
        self.fc = torch.nn.Linear(args.batch_size, 1)
        # x = (torch.arange(0, 100) + 0.5).view(1, -1) / 100
        # self.register_buffer('positions', x)
        self.num_components = args.n_components
        self.prior_type = args.prior_type
        self.prior = Prior(data_size=[self.num_components, self.num_factors],
                           prior_type=self.prior_type)

    def sampling_z(self, num_samples):
        return self.prior.sampling(num_samples)

    def sampling(self, vs: torch.Tensor):
        """
        Sampling graphon factors
        Args:
            vs: (n_nodes)
        Returns:
            graphons: (n_factors, n_nodes, n_nodes)
            signals: (n_factors, n_nodes, n_nodes)
        """
        n_nodes = vs.shape[0]
        graphons = torch.zeros(1, self.num_factors, n_nodes, n_nodes).to(vs.device)
        # signals = torch.zeros(1, self.num_factors, n_nodes, self.dim).to(vs.device)
        for c in range(self.num_factors):
            idx = torch.floor(self.num_partitions[c] * vs).long()
            graphons[0, c, :, :] = self.factors_graphon[c][idx, :][:, idx]
            # signals[0, c, :, :] = self.factors_signal[c][idx, :]
        graphons = self.sigmoid(graphons)
        # return graphons, signals
        return graphons

    def forward(self, zs: torch.Tensor, vs: torch.Tensor):
        """
        Given a graphon model, sample a batch of graphs from it
        Args:
            zs: (batch_size, n_factors) latent representations
            vs: (n_nodes) random variables ~ Uniform([0, 1])

        Returns:
            graphon: (batch_size, n_nodes, n_nodes)
            signal: (batch_size, n_nodes, dim)
            graph: (batch_size, n_nodes, n_nodes) adjacency matrix
            attribute: (batch_size, n_nodes, dim) node attributes
        """
        tzs = zs.t()
        tzs_pad = None
        if self.batch_size - tzs.shape[0] != 0:
            pad = torch.zeros(tzs.shape[0], self.batch_size - tzs.shape[1], device=tzs.device)
            tzs_pad = torch.cat((tzs, pad), dim=1)
        zs_hat_one = self.fc(tzs_pad).t()
        zs_hat = self.softmax1(zs_hat_one)
        # graphons, signals = self.sampling(vs)  # basis
        # print('zs_hat', zs_hat.shape)

        graphons = self.sampling(vs)  # basis
        # print('graphons', graphons.shape)
        graphon = (zs_hat.view(-1, self.num_factors, 1, 1) * graphons).sum(1)  # (batch, n_nodes, n_nodes)
        graphon_est = graphon.squeeze()
        # signal = (zs_hat.view(-1, self.num_factors, 1, 1) * signals).sum(1)  # (batch, n_nodes, dim)
        # if self.node_type == 'binary':
        #     signal = self.sigmoid(signal)
        # if self.node_type == 'categorical':
        #     signal = self.softmax2(signal)
        # graphs = torch.bernoulli(graphon)  # TODO: ???!?!
        # graphs += graphs.clone().permute(0, 2, 1)  # Change: add .clone()
        # graphs[graphs > 1] = 1
        # if self.node_type == "binary":
        #     attributes = torch.bernoulli(signal)
        # elif self.node_type == "categorical":
        #     distribution = torch.distributions.one_hot_categorical.OneHotCategorical(signal)
        #     attributes = distribution.sample()
        # else:
        #     distribution = torch.distributions.normal.Normal(signal, scale=2)
        #     attributes = distribution.sample()
        # return graphon, signal, graphs, attributes
        return graphon_est


class GraphonNewFactorization(torch.nn.Module, ABC):
    def __init__(self, num_factors: int, graphs_pre, graphs_down: TUDataset, seed: int, args,
                 node_type: str = 'categorical'):
        """
        A basic graphon model based on Fourier transformation
        Args:
            num_factors: the number of sin/cos bases for one graphon
            graphs: the graphs used as the prior of the model
            seed: random seed
            node_type: 'binary', 'categorical' and 'continuous'
        """
        super(GraphonNewFactorization, self).__init__()
        self.num_factors = num_factors
        self.node_type = node_type
        self.factors_graphon = nn.ParameterList()
        self.factors_signal = nn.ParameterList()
        self.num_partitions = []

        num_mul_nodes = args.nnodes * args.ngraphs
        indices_pre = list(range(len(graphs_pre)))
        random.seed(seed)
        random.shuffle(indices_pre)
        indices_pre = indices_pre[:num_mul_nodes]
        num_pre_factors = int(self.num_factors / 2)
        print('Dealing with pretrain basis')
        for c in range(num_pre_factors):
            sample = graphs_pre[indices_pre[c]]
            node_ids = list(range(sample.x.shape[0]))
            random.shuffle(node_ids)
            node_sample = node_ids[:num_mul_nodes]
            edge, _ = subgraph(node_sample, sample.edge_index, relabel_nodes=True)
            adj = torch.sparse_coo_tensor(edge,
                                          torch.ones(edge.shape[1]),
                                          size=[num_mul_nodes, num_mul_nodes])
            adj = adj.to_dense()
            # print(adj.shape)
            if len(adj.shape) > 2:
                adj = adj.sum(2)
            # attribute = sample.x
            degrees = torch.sum(adj, dim=1)
            idx = torch.argsort(degrees)
            # print(idx.shape)
            adj = adj[idx, :][:, idx]
            # attribute = attribute[idx, :]
            num_partitions = adj.shape[0]
            graphon = nn.Parameter(data=(adj - 0.5), requires_grad=True)
            self.num_partitions.append(num_partitions)
            self.factors_graphon.append(graphon)

        indices_down = list(range(len(graphs_down)))
        random.seed(seed)
        random.shuffle(indices_down)
        # indices = np.random.RandomState(seed).permutation(len(graphs))
        print('Dealing with downstream basis')
        for c in range(self.num_factors - num_pre_factors):
            sample = graphs_down[indices_down[c]]
            # adj = torch.sparse_coo_tensor(sample.edge_index,
            #                               torch.ones(sample.edge_index.shape[1]),
            #                               size=[sample.x.shape[0], sample.x.shape[0]])
            node_ids = list(range(sample.x.shape[0]))
            random.shuffle(node_ids)
            node_sample = node_ids[:num_mul_nodes]
            edge, _ = subgraph(node_sample, sample.edge_index, relabel_nodes=True)
            adj = torch.sparse_coo_tensor(edge,
                                          torch.ones(edge.shape[1]),
                                          size=[num_mul_nodes, num_mul_nodes])
            adj = adj.to_dense()
            # print(adj.shape)
            if len(adj.shape) > 2:
                adj = adj.sum(2)
            # attribute = sample.x
            degrees = torch.sum(adj, dim=1)
            idx = torch.argsort(degrees)
            # print(idx.shape)
            adj = adj[idx, :][:, idx]
            # attribute = attribute[idx, :]
            num_partitions = adj.shape[0]
            graphon = nn.Parameter(data=(adj - 0.5), requires_grad=True)
            # if self.node_type == "binary" or "categorical":
            #     signal = nn.Parameter(data=(attribute - 0.5), requires_grad=True)
            # else:
            #     signal = nn.Parameter(data=attribute, requires_grad=True)
            self.num_partitions.append(num_partitions)
            self.factors_graphon.append(graphon)
            # self.factors_signal.append(signal)
        # self.dim = self.factors_signal[0].shape[1]
        self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()
        self.softmax0 = nn.Softmax(dim=0)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)

        # 这里三层softmax是？

        self.batch_size = args.batch_size
        self.fc = torch.nn.Linear(args.batch_size, 1)
        # x = (torch.arange(0, 100) + 0.5).view(1, -1) / 100
        # self.register_buffer('positions', x)
        self.num_components = args.n_components
        self.prior_type = args.prior_type
        self.prior = Prior(data_size=[self.num_components, self.num_factors],
                           prior_type=self.prior_type)

    def sampling_z(self, num_samples):
        return self.prior.sampling(num_samples)

    def sampling(self, vs: torch.Tensor):
        """
        Sampling graphon factors
        Args:
            vs: (n_nodes)
        Returns:
            graphons: (n_factors, n_nodes, n_nodes)
            signals: (n_factors, n_nodes, n_nodes)
        """
        n_nodes = vs.shape[0]
        graphons = torch.zeros(1, self.num_factors, n_nodes, n_nodes).to(vs.device)
        # signals = torch.zeros(1, self.num_factors, n_nodes, self.dim).to(vs.device)
        for c in range(self.num_factors):
            idx = torch.floor(self.num_partitions[c] * vs).long()
            graphons[0, c, :, :] = self.factors_graphon[c][idx, :][:, idx]
            # signals[0, c, :, :] = self.factors_signal[c][idx, :]
        graphons = self.sigmoid(graphons)
        # return graphons, signals
        return graphons

    def forward(self, zs: torch.Tensor, vs: torch.Tensor):
        """
        Given a graphon model, sample a batch of graphs from it
        Args:
            zs: (batch_size, n_factors) latent representations
            vs: (n_nodes) random variables ~ Uniform([0, 1])

        Returns:
            graphon: (batch_size, n_nodes, n_nodes)
            signal: (batch_size, n_nodes, dim)
            graph: (batch_size, n_nodes, n_nodes) adjacency matrix
            attribute: (batch_size, n_nodes, dim) node attributes
        """
        tzs = zs.t()
        tzs_pad = None
        if self.batch_size - tzs.shape[0] != 0:
            pad = torch.zeros(tzs.shape[0], self.batch_size - tzs.shape[1], device=tzs.device)
            tzs_pad = torch.cat((tzs, pad), dim=1)
        zs_hat_one = self.fc(tzs_pad).t()
        zs_hat = self.softmax1(zs_hat_one)
        # graphons, signals = self.sampling(vs)  # basis
        # print('zs_hat', zs_hat.shape)

        graphons = self.sampling(vs)  # basis
        # print('graphons', graphons.shape)
        graphon = (zs_hat.view(-1, self.num_factors, 1, 1) * graphons).sum(1)  # (batch, n_nodes, n_nodes)
        graphon_est = graphon.squeeze()
        # signal = (zs_hat.view(-1, self.num_factors, 1, 1) * signals).sum(1)  # (batch, n_nodes, dim)
        # if self.node_type == 'binary':
        #     signal = self.sigmoid(signal)
        # if self.node_type == 'categorical':
        #     signal = self.softmax2(signal)
        # graphs = torch.bernoulli(graphon)  # TODO: ???!?!
        # graphs += graphs.clone().permute(0, 2, 1)  # Change: add .clone()
        # graphs[graphs > 1] = 1
        # if self.node_type == "binary":
        #     attributes = torch.bernoulli(signal)
        # elif self.node_type == "categorical":
        #     distribution = torch.distributions.one_hot_categorical.OneHotCategorical(signal)
        #     attributes = distribution.sample()
        # else:
        #     distribution = torch.distributions.normal.Normal(signal, scale=2)
        #     attributes = distribution.sample()
        # return graphon, signal, graphs, attributes
        return graphon_est


def raml(graphons_hat, graphons_lbl, args):
    # adj = torch.sparse_coo_tensor(data.edge_index,
    #                               torch.ones(data.edge_index.shape[1]),
    #                               size=[data.x.shape[0], data.x.shape[0]])
    # adj = adj.to_dense()
    # if len(adj.shape) > 2:
    #     adj = adj.sum(2)
    # log_p_x = torch.zeros(graphons_hat.shape[0], args.n_graphs).to(graphons_hat.device)
    # for b in range(graphons_hat.shape[0]):
    # d_fgw = torch.zeros(args.n_graphs).to(graphons_hat.device)
    # adj2 = adj[data.batch == b, :][:, data.batch == b]
    # s2 = data.x[data.batch == b, :]

    # for k in range(args.n_graphs):
    #     adj0 = graphons_hat[b, k * args.n_nodes:(k + 1) * args.n_nodes, :][:,
    #            k * args.n_nodes:(k + 1) * args.n_nodes]
    #     # s0 = signals[b, k * args.n_nodes:(k + 1) * args.n_nodes, :]
    #     adj1 = graphons_lbl[b, k * args.n_nodes:(k + 1) * args.n_nodes, :][:,
    #            k * args.n_nodes:(k + 1) * args.n_nodes]

    # s1 = attributes[b, k * args.n_nodes:(k + 1) * args.n_nodes, :]
    # if node_type == 'binary':
    #     log_p_x[b, k] = F.binary_cross_entropy(input=adj0, target=adj1, reduction='mean')
    # elif node_type == 'categorical':
    #     log_p_x[b, k] = F.binary_cross_entropy(input=adj0, target=adj1, reduction='mean')
    # else:
    # log_p_x[b, k] = F.binary_cross_entropy(input=adj0, target=adj1, reduction='mean')
    # d_fgw[k] = fgw_distance(adj1, adj2, args)
    # d_fgw[k] = fgw_distance(adj1, adj0, args)
    # print(b, k, d_fgw[k])
    # print('graphons_hat, graphons_lbl')
    # print(graphons_hat.shape)
    # print(graphons_lbl.shape)
    d_fgw = fgw_distance(graphons_hat, graphons_lbl, args)
    # print('d_fgw', d_fgw)
    # q_x = F.softmax(-2 * d_fgw / torch.min(d_fgw), dim=0).detach()  # TODO: detach() ??
    # log_p_x[b, :] *= q_x
    # print(q_x.shape)
    # log_p_x[b, :] = q_x
    # return log_p_x.mean()
    return d_fgw


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


if __name__ == "__main__":
    pass
