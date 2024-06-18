import argparse

from torch_geometric.transforms import LocalDegreeProfile

DATASET = [
    "IMDB-BINARY",
    "IMDB-MULTI",
]

DATA_PARAMETER = dict()
DATA_PARAMETER["IMDB-BINARY"] = {"node_type": "continuous",
                                 "n_cluster": 2,
                                 "n_factor": 15,
                                 "transform": LocalDegreeProfile()}
DATA_PARAMETER["IMDB-MULTI"] = {"node_type": "continuous",
                                "n_cluster": 3,
                                "n_factor": 15,
                                "transform": LocalDegreeProfile()}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    # Training settings
    parser = argparse.ArgumentParser(
        description='PyTorch implementation of pre-training of graph neural networks with funetune technique')
    parser.add_argument('--gpu', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int,
                        default=300,
                        # default=32,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--basis_dim', type=int, default=300,
                        help='graphon basis dimensions (default: 15)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default='tox21',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default='./model_gin/contextpred.pth',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default='none', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    parser.add_argument("--save_path", type=str,
                        default='./',
                        help="Where to save finetuned model.")
    parser.add_argument("--save_trained", type=str,
                        default='/home/syf/workspace/GTOT-Tuning/chem/save_trained/',
                        help="Where to save finetuned model.")

    # for GraphonAE
    parser.add_argument("--n-graphs", type=int, default=10, help="The number of sampled graphs")
    parser.add_argument("--n-nodes", type=int, default=50, help="The number of sampled nodes per graph")
    parser.add_argument("--n-factors", type=int,
                        # default=15,
                        # default=32,
                        default=300,
                        help="The number of graphon factors")
    parser.add_argument("--n_components", type=int, default=3)  # TODO:
    parser.add_argument('--node_type', type=str, default="continuous", help="random or scaffold or random_scaffold")
    parser.add_argument("--prior-type", type=str, default="gmm", help="The type of prior")
    # for FGW_dist
    parser.add_argument("--n-iter", type=int, default=20, help="The number of outer iterations of FGW_dist")
    parser.add_argument("--n-sinkhorn", type=int, default=5, help="The number of inner iterations")
    parser.add_argument("--beta_gw", type=float, default=0.5, help="The weight of GW term")

    # for graphon est
    parser.add_argument('--f-result', type=str,
                        default='./graphons',
                        help='the root path saving learning results')
    parser.add_argument('--r', type=int,
                        default=1000,
                        help='the resolution of graphon')
    parser.add_argument('--num-graphs', type=int,
                        default=10,
                        help='the number of synthetic graphs')
    parser.add_argument('--num-nodes', type=int, default=200,
                        help='the number of nodes per graph')
    parser.add_argument('--graph-size', type=str, default='random',
                        help='the size of each graph, random or fixed')
    parser.add_argument('--threshold-sba', type=float, default=0.1,
                        help='the threshold of sba method')
    parser.add_argument('--threshold-usvt', type=float, default=0.1,
                        help='the threshold of usvt method')
    parser.add_argument('--alpha_est', type=float, default=0.0003,
                        help='the weight of smoothness regularizer')
    parser.add_argument('--beta', type=float, default=5e-3,
                        help='the weight of proximal term')
    parser.add_argument('--gamma', type=float, default=0.0001,
                        help='the weight of gw term')
    parser.add_argument('--inner-iters', type=int, default=50,
                        help='the number of inner iterations')
    parser.add_argument('--outer-iters', type=int, default=20,
                        help='the number of outer iterations')
    parser.add_argument('--n-trials', type=int, default=1,
                        help='the number of trials')

    # for finetune
    parser.add_argument('--regularization_type', type=str,
                        # choices=['l2_sp', 'feature_map', 'attention_feature_map',"none"],
                        # gtot_feature_map, ori: none
                        # default='gtot_feature_map',
                        default='none',
                        help='fine tune regularization.')
    parser.add_argument('--finetune_type', type=str,
                        # gtot, ori: none
                        default='none',
                        help='fine tune regularization.')  # choices=['delta', 'bitune', 'co_tune','l2_sp','none','bss'],
    parser.add_argument('--norm_type', type=str,
                        default='none', help='fine tune regularization.')

    parser.add_argument('--rec_weight', default=1.0, type=float,
                        help='trade-off for backbone regularization')
    parser.add_argument('--reg_weight', default=1.0, type=float,
                        help='trade-off for backbone regularization')

    parser.add_argument('--trade_off_backbone', default=0.0, type=float,
                        help='trade-off for backbone regularization')
    parser.add_argument('--trade_off_head', default=0.0, type=float,
                        help='trade-off for head regularization')
    ## bss
    parser.add_argument('--trade_off_bss', default=0.0, type=float,
                        help='trade-off for bss regularization')
    parser.add_argument('-k', '--k', default=1, type=int,
                        metavar='N',
                        help='hyper-parameter for BSS loss')
    parser.add_argument('--gtot_order', default=1, type=int, help='A^{k} in graph topology OT')

    # parameters for calculating channel attention
    parser.add_argument("--attention_file", type=str, default='channel_attention.pt',
                        help="Where to save and load channel attention file.")
    parser.add_argument("--data_path", type=str,
                        default='/data/syf/finetune/dataset',
                        help="Where to save and load dataset.")

    parser.add_argument('--attention-batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size for calculating channel attention (default: 32)')
    parser.add_argument('--attention_epochs', default=10, type=int, metavar='N',
                        # 10,50
                        help='number of epochs to train for training before calculating channel weight')
    parser.add_argument('--attention-lr-decay-epochs', default=30, type=int, metavar='N',
                        help='epochs to decay lr for training before calculating channel weight')
    parser.add_argument('--attention_iteration_limit', default=10, type=int, metavar='N',
                        # 10,50
                        help='iteration limits for calculating channel attention, -1 means no limits')
    ## for stochnorm
    parser.add_argument('--prob', '--probability', default=0.5, type=float,
                        metavar='P', help='Probability for StochNorm layers')

    parser.add_argument('--print_freq', default=20, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--patience', type=int, default=20,
                        help='early stop patience.')

    parser.add_argument('--save_file', default='/data/syf/finetune/results.csv', help='save file name for results')
    parser.add_argument('--tag', default='revise',
                        # gtot_best_parameter, none
                        help='tag for labeling the experiment')
    parser.add_argument('--debug', action='store_true', help='whether use the debug')

    ## for gtot
    parser.add_argument('--train_radio', default=1.0, type=float,
                        help='(train_set* train_radio) : val : test')

    parser.add_argument('--dist_metric', default='norm_cosine', type=str,
                        help='distance metric for optimal transport as cost matrix (cosine, norm_cosine)')

    args = parser.parse_args()

    return args
