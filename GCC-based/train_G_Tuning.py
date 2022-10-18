#!/usr/bin/env python
# encoding: utf-8
# File Name: train_graph_moco.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/13 16:44
# TODO:

import argparse
import copy
import os
import time
import warnings

import nni
import dgl
import numpy as np
import psutil
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.datasets import TUDataset
import methods.learner as learner
from model import GraphonEncoder, GraphonFactorization
from gcc.contrastive.criterions import NCESoftmaxLoss, NCESoftmaxLossNS
from gcc.contrastive.memory_moco import MemoryMoCo
from gcc.datasets import (
    GRAPH_CLASSIFICATION_DSETS,
    GraphClassificationDataset,
    GraphClassificationDatasetLabeled,
    LoadBalanceGraphDataset,
    NodeClassificationDataset,
    NodeClassificationDatasetLabeled,
    worker_init_fn,
)
from gcc.datasets.data_util import batcher, labeled_batcher
from gcc.models import GraphEncoder
from gcc.utils.misc import AverageMeter, adjust_learning_rate, warmup_linear

from finetune.delta import L2Regularization
from ot_distance import fgw_distance

# num_threads = '32'
# torch.set_num_threads(int(num_threads))
# os.environ['OMP_NUM_THREADS'] = num_threads
# os.environ['OPENBLAS_NUM_THREADS'] = num_threads
# os.environ['MKL_NUM_THREADS'] = num_threads
# os.environ['VECLIB_MAXIMUM_THREADS'] = num_threads
# os.environ['NUMEXPR_NUM_THREADS'] = num_threads
# os.environ['NUMEXPR_MAX_THREADS'] = '128'

finetune_data = {
    "imdb-binary": "IMDB-BINARY",
    "imdb-multi": "IMDB-MULTI",
    "collab": "COLLAB",
    "rdt-b": "REDDIT-BINARY",
    "rdt-5k": "REDDIT-MULTI-5K",
    "PROTEINS": "PROTEINS",
    "AIDS": "AIDS",
    "MSRC_21": "MSRC_21",
    "MUTAG": "MUTAG",
    "ENZYMES": "ENZYMES"
}


def parse_option():
    # fmt: off
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print-freq", type=int, default=10, help="print frequency")
    parser.add_argument("--tb-freq", type=int, default=250, help="tb frequency")
    parser.add_argument("--save-freq", type=int, default=1, help="save frequency")
    parser.add_argument("--batch-size", type=int, default=32, help="batch_size")
    parser.add_argument("--num-workers", type=int, default=12, help="num of workers to use")
    parser.add_argument("--num-copies", type=int, default=6, help="num of dataset copies that fit in memory")
    parser.add_argument("--num-samples", type=int, default=2000, help="num of samples per batch per worker")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")

    # optimization
    parser.add_argument("--optimizer", type=str, default='adam', choices=['sgd', 'adam', 'adagrad'], help="optimizer")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="learning rate")
    parser.add_argument("--lr_decay_epochs", type=str, default="120,160,200", help="where to decay lr, can be a list")
    parser.add_argument("--lr_decay_rate", type=float, default=0.0, help="decay rate for learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="beta2 for Adam")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="clip norm")

    # resume
    parser.add_argument("--resume", type=str,
                        default="/home/syf/workspace/GCC/saved/Pretrain_moco_True_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_0.999/current.pth",
                        help="path to latest checkpoint (default: none)")

    # augmentation setting
    parser.add_argument("--aug", type=str, default="1st", choices=["1st", "2nd", "all"])

    parser.add_argument("--exp", type=str, default="ENZYMES_W")

    # dataset definition
    parser.add_argument("--dataset", type=str, default="ENZYMES",
                        choices=["dgl", "wikipedia", "blogcatalog", "usa_airport", "brazil_airport", "europe_airport",
                                 "cora", "citeseer", "pubmed", "kdd", "icdm", "sigir", "cikm", "sigmod", "icde",
                                 "h-index-rand-1", "h-index-top-1", "h-index"] + GRAPH_CLASSIFICATION_DSETS)

    # model definition
    parser.add_argument("--model", type=str, default="gin", choices=["gat", "mpnn", "gin"])
    # other possible choices: ggnn, mpnn, graphsage ...
    parser.add_argument("--num-layer", type=int, default=5, help="gnn layers")
    parser.add_argument("--readout", type=str, default="avg", choices=["avg", "set2set"])
    parser.add_argument("--set2set-lstm-layer", type=int, default=3, help="lstm layers for s2s")
    parser.add_argument("--set2set-iter", type=int, default=6, help="s2s iteration")
    parser.add_argument("--norm", action="store_true", default=True, help="apply 2-norm on output feats")

    # loss function
    parser.add_argument("--nce-k", type=int, default=32)
    parser.add_argument("--nce-t", type=float, default=0.07)

    # random walk
    parser.add_argument("--rw-hops", type=int, default=256)
    parser.add_argument("--subgraph-size", type=int, default=128)
    parser.add_argument("--restart-prob", type=float, default=0.8)
    parser.add_argument("--hidden-size", type=int, default=64)
    # parser.add_argument("--basis", type=int, default=15)
    parser.add_argument("--positional-embedding-size", type=int, default=32)
    parser.add_argument("--max-node-freq", type=int, default=16)
    parser.add_argument("--max-edge-freq", type=int, default=16)
    parser.add_argument("--max-degree", type=int, default=512)
    parser.add_argument("--freq-embedding-size", type=int, default=16)
    parser.add_argument("--degree-embedding-size", type=int, default=16)

    # specify folder
    parser.add_argument("--model-path", type=str, default='saved', help="path to save model")
    parser.add_argument("--tb-path", type=str, default='tensorboard', help="path to tensorboard")
    parser.add_argument("--load-path", type=str, default=None, help="loading checkpoint at test time")

    # memory setting
    parser.add_argument("--moco", action="store_true", help="using MoCo (otherwise Instance Discrimination)")

    # finetune setting
    parser.add_argument("--finetune", action="store_true", default=True)
    # parser.add_argument("--align", action="store_true")

    parser.add_argument("--alpha", type=float, default=0.999, help="exponential moving average weight")

    # GPU setting
    parser.add_argument("--gpu", default='7', type=int, nargs='+', help="GPU id to use.")

    # cross validation
    parser.add_argument("--seed", type=int, default=0, help="random seed.")
    parser.add_argument("--fold-idx", type=int, default=0, help="random seed.")
    parser.add_argument("--cv", action="store_true", default=True)
    # fmt: on
    # settings for AE
    parser.add_argument("--n-graphs", type=int, default=5,
                        help="The number of sampled graphs")
    parser.add_argument("--nnodes", type=int, default=10,
                        help="The number of sampled nodes per graph")
    parser.add_argument("--n-factors", type=int,
                        # default=15,
                        default=25,
                        # default=32,
                        # default=300,
                        help="The number of graphon factors")
    parser.add_argument("--n_components", type=int, default=10)  # TODO:
    parser.add_argument('--node_type', type=str, default="continuous",
                        help="random or scaffold or random_scaffold")
    parser.add_argument("--prior-type", type=str,
                        default="gmm", help="The type of prior")
    # for FGW_dist
    parser.add_argument("--n-iter", type=int, default=20,
                        help="The number of outer iterations of FGW_dist")
    parser.add_argument("--n-sinkhorn", type=int, default=5,
                        help="The number of inner iterations")
    parser.add_argument("--beta_gw", type=float, default=0.5,
                        help="The weight of GW term")

    parser.add_argument('--rec_weight', default=0.5, type=float,
                        help='trade-off for backbone regularization')
    parser.add_argument('--cls_weight', default=1.0, type=float,
                        help='trade-off for backbone regularization')
    parser.add_argument('--trade_off_head', default=0.0, type=float,
                        help='trade-off for head regularization')

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
    # parser.add_argument('--graph-size', type=str, default='random',
    #                     help='the size of each graph, random or fixed')
    parser.add_argument('--graphon_method', type=str,
                        # default='GWB',
                        default='SFGWB',
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

    opt = parser.parse_args()
    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt


def option_update(opt):
    opt.model_name = "{}_moco_{}_{}_{}_layer_{}_lr_{}_decay_{}_bsz_{}_hid_{}_samples_{}_nce_t_{}_nce_k_{}_rw_hops_{}_restart_prob_{}_aug_{}_ft_{}_deg_{}_pos_{}_momentum_{}".format(
        opt.exp,
        opt.moco,
        opt.dataset,
        opt.model,
        opt.num_layer,
        opt.learning_rate,
        opt.weight_decay,
        opt.batch_size,
        opt.hidden_size,
        opt.num_samples,
        opt.nce_t,
        opt.nce_k,
        opt.rw_hops,
        opt.restart_prob,
        opt.aug,
        opt.finetune,
        opt.degree_embedding_size,
        opt.positional_embedding_size,
        opt.alpha,
    )

    if opt.load_path is None:
        opt.model_folder = os.path.join(opt.model_path, opt.model_name)
        if not os.path.isdir(opt.model_folder):
            os.makedirs(opt.model_folder)
    else:
        opt.model_folder = opt.load_path

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    return opt


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


def train_finetune(
        epoch,
        train_loader,
        head_regularization,
        model,
        WEncoder,
        WDecoder,
        output_layer,
        criterion,
        optimizer,
        output_layer_optimizer,
        sw,
        opt,
        param_args,
):
    """
    one epoch training for moco
    """
    n_batch = len(train_loader)
    model.train()
    output_layer.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    f1_meter = AverageMeter()
    epoch_loss_meter = AverageMeter()
    epoch_f1_meter = AverageMeter()
    prob_meter = AverageMeter()
    graph_size = AverageMeter()
    max_num_nodes = 0
    max_num_edges = 0

    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        graph_q, y = batch

        graph_q.to(torch.device(opt.gpu))
        y = y.to(torch.device(opt.gpu))

        bsz = graph_q.batch_size

        # ===================forward=====================

        feat_q = model(graph_q)

        assert feat_q.shape == (graph_q.batch_size, opt.hidden_size)

        graphon_weight = WEncoder(feat_q)

        vs = torch.rand(int(opt.n_graphs * args.nnodes)
                        ).to(torch.device(opt.gpu))

        graphon_est = WDecoder(graphon_weight, vs)

        graph_list = []
        edges = graph_q.edges()
        edge_index = torch.tensor([edges[0].tolist(), edges[1].tolist()])
        adj = to_scipy_sparse_matrix(edge_index).toarray()
        graph_list.append(adj)
        _, graphon_oracle = learner.estimate_graphon(
            graph_list, method='SFGWB', args=opt, resolution=opt.r)
        if epoch % 10 == 0 and idx % 10 == 0:
            # self.factors_graphon
            for id, w in enumerate(WDecoder.factors_graphon):
                np.save('./saved_f/factors_w_' + str(epoch) + '_' + str(idx) + '_' + str(id) + '.npy',
                        w.cpu().detach().numpy())
            np.save('./saved_f/w_oracle_' + str(epoch) +
                    '_' + str(idx) + '.npy', graphon_oracle)
            np.save('./saved_f/w_est_' + str(epoch) + '_' +
                    str(idx) + '.npy', graphon_est.cpu().detach().numpy())

            print("saved", epoch, idx)

        rec_loss = fgw_distance(torch.tensor(graphon_oracle, dtype=torch.double, device=graphon_est.device),
                                graphon_est.double(),
                                opt)  # TODO: ADD LOSS

        out = output_layer(feat_q)

        loss_cls = criterion(out, y)
        loss_reg_head = head_regularization()

        # loss = args.cls_weight * loss_cls + \
        #         args.trade_off_head * loss_reg_head + \
        #         args.rec_weight * rec_loss
        loss = args.cls_weight * loss_cls + \
            args.trade_off_head * loss_reg_head + \
            args.rec_weight * rec_loss

        # ===================backward=====================
        optimizer.zero_grad()
        output_layer_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1)
        torch.nn.utils.clip_grad_value_(WEncoder.parameters(), 1)
        torch.nn.utils.clip_grad_value_(WDecoder.parameters(), 1)
        torch.nn.utils.clip_grad_value_(output_layer.parameters(), 1)
        global_step = epoch * n_batch + idx
        lr_this_step = opt.learning_rate * warmup_linear(
            global_step / (opt.epochs * n_batch), 0.1
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_step
        for param_group in output_layer_optimizer.param_groups:
            param_group["lr"] = lr_this_step
        optimizer.step()
        output_layer_optimizer.step()

        preds = out.argmax(dim=1)
        f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average="micro")

        # ===================meters=====================
        f1_meter.update(f1, bsz)
        epoch_f1_meter.update(f1, bsz)
        loss_meter.update(loss.item(), bsz)
        epoch_loss_meter.update(loss.item(), bsz)
        graph_size.update(graph_q.number_of_nodes() / bsz, bsz)
        max_num_nodes = max(max_num_nodes, graph_q.number_of_nodes())
        max_num_edges = max(max_num_edges, graph_q.number_of_edges())

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            mem = psutil.virtual_memory()
            #  print(f'{idx:8} - {mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}')
            #  mem_used.append(mem.used/1024**3)
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "f1 {f1.val:.3f} ({f1.avg:.3f})\t"
                "GS {graph_size.val:.3f} ({graph_size.avg:.3f})\t"
                "mem {mem:.3f}".format(
                    epoch,
                    idx + 1,
                    n_batch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=loss_meter,
                    f1=f1_meter,
                    graph_size=graph_size,
                    mem=mem.used / 1024 ** 3,
                )
            )
            #  print(out[0].abs().max())

        # tensorboard logger
        if (idx + 1) % opt.tb_freq == 0:
            sw.add_scalar("ft_loss", loss_meter.avg, global_step)
            sw.add_scalar("ft_f1", f1_meter.avg, global_step)
            sw.add_scalar("graph_size", graph_size.avg, global_step)
            sw.add_scalar("lr", lr_this_step, global_step)
            sw.add_scalar("graph_size/max", max_num_nodes, global_step)
            sw.add_scalar("graph_size/max_edges", max_num_edges, global_step)
            #  sw.add_scalar(
            #      "learning_rate", optimizer.param_groups[0]["lr"], global_step
            #  )
            loss_meter.reset()
            f1_meter.reset()
            graph_size.reset()
            max_num_nodes, max_num_edges = 0, 0
    return epoch_loss_meter.avg, epoch_f1_meter.avg


def test_finetune(epoch, valid_loader, model, output_layer, criterion, sw, opt):
    n_batch = len(valid_loader)
    model.eval()
    output_layer.eval()

    epoch_loss_meter = AverageMeter()
    epoch_f1_meter = AverageMeter()

    for idx, batch in enumerate(valid_loader):
        graph_q, y = batch

        graph_q.to(torch.device(opt.gpu))
        y = y.to(torch.device(opt.gpu))

        bsz = graph_q.batch_size

        # ===================forward=====================

        with torch.no_grad():
            feat_q = model(graph_q)
            assert feat_q.shape == (graph_q.batch_size, opt.hidden_size)
            out = output_layer(feat_q)
        loss = criterion(out, y)

        preds = out.argmax(dim=1)
        f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average="micro")

        # ===================meters=====================
        epoch_loss_meter.update(loss.item(), bsz)
        epoch_f1_meter.update(f1, bsz)

    global_step = (epoch + 1) * n_batch
    sw.add_scalar("ft_loss/valid", epoch_loss_meter.avg, global_step)
    sw.add_scalar("ft_f1/valid", epoch_f1_meter.avg, global_step)
    print(
        f"Epoch {epoch}, loss {epoch_loss_meter.avg:.3f}, f1 {epoch_f1_meter.avg:.3f}"
    )

    # nni.report_intermediate_result(epoch_f1_meter.avg)
    return epoch_loss_meter.avg, epoch_f1_meter.avg


def clip_grad_norm(params, max_norm):
    """Clips gradient norm."""
    if max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    else:
        return torch.sqrt(
            sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None)
        )


def train_moco(
        epoch, train_loader, model, model_ema, contrast, criterion, optimizer, sw, opt
):
    """
    one epoch training for moco
    """
    n_batch = train_loader.dataset.total // opt.batch_size
    model.train()
    model_ema.eval()

    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.train()

    model_ema.apply(set_bn_train)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    epoch_loss_meter = AverageMeter()
    prob_meter = AverageMeter()
    graph_size = AverageMeter()
    gnorm_meter = AverageMeter()
    max_num_nodes = 0
    max_num_edges = 0

    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        graph_q, graph_k = batch

        graph_q.to(torch.device(opt.gpu))
        graph_k.to(torch.device(opt.gpu))

        bsz = graph_q.batch_size

        if opt.moco:
            # ===================Moco forward=====================
            feat_q = model(graph_q)
            with torch.no_grad():
                feat_k = model_ema(graph_k)

            out = contrast(feat_q, feat_k)
            prob = out[:, 0].mean()
        else:
            # ===================Negative sampling forward=====================
            feat_q = model(graph_q)
            feat_k = model(graph_k)

            out = torch.matmul(feat_k, feat_q.t()) / opt.nce_t
            prob = out[range(graph_q.batch_size),
                       range(graph_q.batch_size)].mean()

        assert feat_q.shape == (graph_q.batch_size, opt.hidden_size)

        # ===================backward=====================
        optimizer.zero_grad()
        loss = criterion(out)
        loss.backward()
        grad_norm = clip_grad_norm(model.parameters(), opt.clip_norm)

        global_step = epoch * n_batch + idx
        lr_this_step = opt.learning_rate * warmup_linear(
            global_step / (opt.epochs * n_batch), 0.1
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_step
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        epoch_loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)
        graph_size.update(
            (graph_q.number_of_nodes() +
             graph_k.number_of_nodes()) / 2.0 / bsz, 2 * bsz
        )
        gnorm_meter.update(grad_norm, 1)
        max_num_nodes = max(max_num_nodes, graph_q.number_of_nodes())
        max_num_edges = max(max_num_edges, graph_q.number_of_edges())

        if opt.moco:
            moment_update(model, model_ema, opt.alpha)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            mem = psutil.virtual_memory()
            #  print(f'{idx:8} - {mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}')
            #  mem_used.append(mem.used/1024**3)
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "prob {prob.val:.3f} ({prob.avg:.3f})\t"
                "GS {graph_size.val:.3f} ({graph_size.avg:.3f})\t"
                "mem {mem:.3f}".format(
                    epoch,
                    idx + 1,
                    n_batch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=loss_meter,
                    prob=prob_meter,
                    graph_size=graph_size,
                    mem=mem.used / 1024 ** 3,
                )
            )
            #  print(out[0].abs().max())

        # tensorboard logger
        if (idx + 1) % opt.tb_freq == 0:
            global_step = epoch * n_batch + idx
            sw.add_scalar("moco_loss", loss_meter.avg, global_step)
            sw.add_scalar("moco_prob", prob_meter.avg, global_step)
            sw.add_scalar("graph_size", graph_size.avg, global_step)
            sw.add_scalar("graph_size/max", max_num_nodes, global_step)
            sw.add_scalar("graph_size/max_edges", max_num_edges, global_step)
            sw.add_scalar("gnorm", gnorm_meter.avg, global_step)
            sw.add_scalar("learning_rate",
                          optimizer.param_groups[0]["lr"], global_step)
            loss_meter.reset()
            prob_meter.reset()
            graph_size.reset()
            gnorm_meter.reset()
            max_num_nodes, max_num_edges = 0, 0
    return epoch_loss_meter.avg


# def main(args, trial):
def main(args):
    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            pretrain_args = checkpoint["opt"]
            pretrain_args.fold_idx = args.fold_idx
            pretrain_args.gpu = args.gpu
            pretrain_args.finetune = args.finetune
            pretrain_args.resume = args.resume
            pretrain_args.cv = args.cv
            pretrain_args.dataset = args.dataset
            pretrain_args.epochs = args.epochs
            pretrain_args.num_workers = args.num_workers
            # graphon
            # pretrain_args.n_factor = args.n_factor
            pretrain_args.__setattr__('n_factors', args.n_factors)
            pretrain_args.__setattr__('n_iter', args.n_iter)
            pretrain_args.__setattr__('n_sinkhorn', args.n_sinkhorn)
            pretrain_args.__setattr__('beta_gw', args.beta_gw)
            pretrain_args.__setattr__('r', args.r)
            pretrain_args.__setattr__('threshold_sba', args.threshold_sba)
            pretrain_args.__setattr__('threshold_usvt', args.threshold_usvt)
            pretrain_args.__setattr__('alpha_est', args.alpha_est)
            pretrain_args.__setattr__('beta', args.beta)
            pretrain_args.__setattr__('gamma', args.gamma)
            pretrain_args.__setattr__('inner_iters', args.inner_iters)
            pretrain_args.__setattr__('outer_iters', args.outer_iters)
            pretrain_args.__setattr__('n_trials', args.n_trials)
            pretrain_args.__setattr__('node_type', args.node_type)
            pretrain_args.__setattr__('prior_type', args.prior_type)
            pretrain_args.__setattr__('n_components', args.n_components)
            pretrain_args.__setattr__('n_graphs', args.n_graphs)
            pretrain_args.__setattr__('nnodes', args.nnodes)
            # pretrain_args.__setattr__('graphon_method', args.graphon_method)

            if args.dataset in GRAPH_CLASSIFICATION_DSETS:
                # HACK for speeding up finetuning on graph classification tasks
                pretrain_args.num_workers = 0
            pretrain_args.batch_size = args.batch_size
            args = pretrain_args
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    args = option_update(args)
    print(args)
    assert args.gpu is not None and torch.cuda.is_available()
    print("Use GPU: {} for training".format(args.gpu))
    assert args.positional_embedding_size % 2 == 0
    print("setting random seeds")

    mem = psutil.virtual_memory()
    print("before construct dataset", mem.used / 1024 ** 3)
    if args.finetune:  # TODO: here change
        if args.dataset in GRAPH_CLASSIFICATION_DSETS:
            dataset = GraphClassificationDatasetLabeled(
                dataset=args.dataset,
                rw_hops=args.rw_hops,
                subgraph_size=args.subgraph_size,
                restart_prob=args.restart_prob,
                positional_embedding_size=args.positional_embedding_size,
            )
            labels = dataset.dataset.graph_labels.tolist()
        else:
            dataset = NodeClassificationDatasetLabeled(
                dataset=args.dataset,
                rw_hops=args.rw_hops,
                subgraph_size=args.subgraph_size,
                restart_prob=args.restart_prob,
                positional_embedding_size=args.positional_embedding_size,
            )
            labels = dataset.data.y.argmax(dim=1).tolist()

        skf = StratifiedKFold(n_splits=10, shuffle=True,
                              random_state=args.seed)
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        assert (
            0 <= args.fold_idx and args.fold_idx < 10
        ), "fold_idx must be from 0 to 9."
        train_idx, test_idx = idx_list[args.fold_idx]
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        valid_dataset = torch.utils.data.Subset(dataset, test_idx)

    elif args.dataset == "dgl":
        train_dataset = LoadBalanceGraphDataset(
            rw_hops=args.rw_hops,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
            num_workers=args.num_workers,
            num_samples=args.num_samples,
            dgl_graphs_file="./data/small.bin",
            num_copies=args.num_copies,
        )
    else:
        if args.dataset in GRAPH_CLASSIFICATION_DSETS:
            train_dataset = GraphClassificationDataset(
                dataset=args.dataset,
                rw_hops=args.rw_hops,
                subgraph_size=args.subgraph_size,
                restart_prob=args.restart_prob,
                positional_embedding_size=args.positional_embedding_size,
            )
        else:
            train_dataset = NodeClassificationDataset(
                dataset=args.dataset,
                rw_hops=args.rw_hops,
                subgraph_size=args.subgraph_size,
                restart_prob=args.restart_prob,
                positional_embedding_size=args.positional_embedding_size,
            )

    mem = psutil.virtual_memory()
    print("before construct dataloader", mem.used / 1024 ** 3)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=labeled_batcher() if args.finetune else batcher(),
        shuffle=True if args.finetune else False,
        num_workers=args.num_workers,
        worker_init_fn=None
        if args.finetune or args.dataset != "dgl"
        else worker_init_fn,
    )
    if args.finetune:
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=args.batch_size,
            collate_fn=labeled_batcher(),
            num_workers=args.num_workers,
        )
    mem = psutil.virtual_memory()
    print("before training", mem.used / 1024 ** 3)

    # create model and optimizer
    # n_data = train_dataset.total
    n_data = None

    model, model_ema = [
        GraphEncoder(
            positional_embedding_size=args.positional_embedding_size,
            max_node_freq=args.max_node_freq,
            max_edge_freq=args.max_edge_freq,
            max_degree=args.max_degree,
            freq_embedding_size=args.freq_embedding_size,
            degree_embedding_size=args.degree_embedding_size,
            output_dim=args.hidden_size,
            node_hidden_dim=args.hidden_size,
            edge_hidden_dim=args.hidden_size,
            num_layers=args.num_layer,
            num_step_set2set=args.set2set_iter,
            num_layer_set2set=args.set2set_lstm_layer,
            norm=args.norm,
            gnn_model=args.model,
            degree_input=True,
        )
        for _ in range(2)
    ]

    # copy weights from `model' to `model_ema'
    if args.moco:
        moment_update(model, model_ema, 0)

    # set the contrast memory and criterion
    contrast = MemoryMoCo(
        args.hidden_size, n_data, args.nce_k, args.nce_t, use_softmax=True
    ).cuda(args.gpu)

    if args.finetune:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = NCESoftmaxLoss() if args.moco else NCESoftmaxLossNS()
        criterion = criterion.cuda(args.gpu)

    model = model.cuda(args.gpu)
    model_ema = model_ema.cuda(args.gpu)
    WEncoder = GraphonEncoder(
        args.hidden_size, args.hidden_size, args.n_factors).cuda(args.gpu)
    # node_type = "continuous"

    graphs = TUDataset(
        root='/data/syf/TUDataset/',
        name=finetune_data[args.dataset]
        # use_node_attr=True,
        # use_edge_attr=False,
        # transform=LocalDegreeProfile(),
    )
    WDecoder = GraphonFactorization(num_factors=args.n_factors,
                                    graphs=graphs,
                                    seed=args.seed,
                                    args=args,
                                    # node_type=parser.DATA_PARAMETER[args.data_name]['node_type)
                                    node_type=args.node_type).cuda(args.gpu)

    if args.finetune:
        output_layer = nn.Linear(
            in_features=args.hidden_size, out_features=dataset.num_classes
        )
        output_layer = output_layer.cuda(args.gpu)
        output_layer_optimizer = torch.optim.Adam(
            output_layer.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )

        def clear_bn(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm") != -1:
                m.reset_running_stats()

        model.apply(clear_bn)

    model_para_group = []
    model_para_group.append({'params': model.parameters()})
    model_para_group.append({'params': WEncoder.parameters()})
    model_para_group.append({'params': WDecoder.parameters()})

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model_para_group,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model_para_group,
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(
            model_para_group,
            lr=args.learning_rate,
            lr_decay=args.lr_decay_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError
    head_regularization = L2Regularization(nn.ModuleList([output_layer]))
    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        # print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location="cpu")
        # checkpoint = torch.load(args.resume)
        # args.start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        contrast.load_state_dict(checkpoint["contrast"])
        if args.moco:
            model_ema.load_state_dict(checkpoint["model_ema"])

        print(
            "=> loaded successfully '{}' (epoch {})".format(
                args.resume, checkpoint["epoch"]
            )
        )
        del checkpoint
        torch.cuda.empty_cache()

    # tensorboard
    #  logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)
    sw = SummaryWriter(args.tb_folder)
    #  plots_q, plots_k = zip(*[train_dataset.getplot(i) for i in range(5)])
    #  plots_q = torch.cat(plots_q)
    #  plots_k = torch.cat(plots_k)
    #  sw.add_images('images/graph_q', plots_q, 0, dataformats="NHWC")
    #  sw.add_images('images/graph_k', plots_k, 0, dataformats="NHWC")

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        if args.finetune:
            loss, _ = train_finetune(
                epoch,
                train_loader,
                head_regularization,
                model,
                WEncoder,
                WDecoder,
                output_layer,
                criterion,
                optimizer,
                output_layer_optimizer,
                sw,
                args,
                param_args,
            )
        else:
            loss = train_moco(
                epoch,
                train_loader,
                model,
                model_ema,
                contrast,
                criterion,
                optimizer,
                sw,
                args,
            )
        time2 = time.time()
        print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

        # save model
        if epoch % args.save_freq == 0:
            print("==> Saving...")
            state = {
                "opt": args,
                "model": model.state_dict(),
                "contrast": contrast.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            if args.moco:
                state["model_ema"] = model_ema.state_dict()

            state["WEncoder"] = WEncoder.state_dict()
            state["WDecoder"] = WDecoder.state_dict()
            save_file = os.path.join(
                args.model_folder, "ckpt_epoch_{epoch}.pth".format(epoch=epoch)
            )
            torch.save(state, save_file)
            # help release GPU memory
            del state

        # saving the model
        print("==> Saving...")
        state = {
            "opt": args,
            "model": model.state_dict(),
            "contrast": contrast.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        if args.moco:
            state["model_ema"] = model_ema.state_dict()

        state["WEncoder"] = WEncoder.state_dict()
        state["WDecoder"] = WDecoder.state_dict()
        save_file = os.path.join(args.model_folder, "current.pth")
        torch.save(state, save_file)
        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                args.model_folder, "ckpt_epoch_{epoch}.pth".format(epoch=epoch)
            )
            torch.save(state, save_file)
        # help release GPU memory
        del state
        torch.cuda.empty_cache()

    if args.finetune:
        valid_loss, valid_f1 = test_finetune(
            epoch, valid_loader, model, output_layer, criterion, sw, args
        )
        # nni.report_intermediate_result(float(valid_f1))
        return valid_f1


if __name__ == "__main__":

    warnings.simplefilter("once", UserWarning)
    args = parse_option()
    param_args = args.__dict__

    if args.cv:
        gpus = args.gpu
        # gpus = [args.gpu]
        print(gpus)

        def variant_args_generator():
            for fold_idx in range(10):
                args.fold_idx = fold_idx
                args.num_workers = 0
                args.gpu = gpus[fold_idx % len(gpus)]
                yield copy.deepcopy(args)

        # f1 = Parallel(n_jobs=5)(
        #     delayed(main)(args) for args in variant_args_generator()
        # )
        f1 = [main(args, param_args) for args in variant_args_generator()]
        print(f1)
        # nni.report_final_result(float(np.mean(f1)))
        print(f"Mean = {np.mean(f1)}; Std = {np.std(f1)}")
    else:
        # args.gpu = args.gpu[0]
        args.gpu = args.gpu
        main(args)
    # import optuna
    # def objective(trial):
    #     args.epochs = 50
    #     args.learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    #     args.weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    #     args.restart_prob = trial.suggest_uniform('restart_prob', 0.5, 1)
    #     # args.alpha = 1 - trial.suggest_loguniform('alpha', 1e-4, 1e-2)
    #     return main(args, trial)

    # study = optuna.load_study(study_name='cat_prone', storage="sqlite:///example.db")
    # study.optimize(objective, n_trials=20)
