import time

import torch_geometric

import os

import methods.learner as learner
from loader import MoleculeDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool
import nni
import math
import torch.nn.functional as F
import torch.optim as optim
import parseArgs
from tqdm import tqdm
import numpy as np
import warnings
from model import GNN_graphpred, GraphonEncoder, GraphonFactorization, GraphonNewEncoder

from ot_distance import sliced_fgw_distance, fgw_distance
from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

# sys.path.append('..')
from commom.meter import AverageMeter, ProgressMeter
from commom.eval import Meter
import os, random
import shutil
from commom.early_stop import EarlyStopping
from commom.run_time import Runtime

import torch
from torch import nn
from ftlib.finetune.stochnorm import convert_model
from ftlib.finetune.bss import BatchSpectralShrinkage
from ftlib.finetune.delta import IntermediateLayerGetter, L2Regularization, get_attribute
from ftlib.finetune.delta import SPRegularization, FrobeniusRegularization

warnings.filterwarnings("ignore", category=Warning)

criterion = nn.BCEWithLogitsLoss(reduction="none")

# num_threads = '16'
# torch.set_num_threads(int(num_threads))
# os.environ['OMP_NUM_THREADS'] = num_threads
# os.environ['OPENBLAS_NUM_THREADS'] = num_threads
# os.environ['MKL_NUM_THREADS'] = num_threads
# os.environ['VECLIB_MAXIMUM_THREADS'] = num_threads
# os.environ['NUMEXPR_NUM_THREADS'] = num_threads


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    ## cuda
    torch.cuda.current_device()
    torch.cuda._initialized = True


def train_epoch(args, model, device, loader, optimizer,
                backbone_regularization, head_regularization, target_getter,
                WEncoder, WDecoder, bss_regularization, scheduler, epoch,
                param_args):
    model.train()

    meter = Meter()
    loss_epoch = []
    batch_iter = tqdm(loader)
    for step, batch in enumerate(batch_iter):
        batch = batch.to(device)
        intermediate_output_t, output_t = target_getter(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        pred = output_t  # 32*12
        # if args.finetune_type == 'none':
        #     fea = None
        # else:
        #     fea = target_getter._model.get_bottleneck()

        intermediate_output = intermediate_output_t['gnn.gnns.4.mlp.2']
        inter_output = global_mean_pool(intermediate_output, batch.batch)

        # TODO: build graphon factor encoder
        graphon_weight = WEncoder(intermediate_output, batch.edge_index,
                                  batch.edge_attr, batch.batch)
        vs = torch.rand(int(param_args['ngraphs'] *
                            param_args['nnodes'])).to(device)

        graphon_est = WDecoder(graphon_weight, vs)
        # TODO: build ground truth for graphon

        graph_list = []
        adj = torch_geometric.utils.to_scipy_sparse_matrix(
            batch.edge_index).toarray()
        graph_list.append(adj)

        _, graphon_oracle = learner.estimate_graphon(
            graph_list,
            method='SAS',
            args=args,
            resolution=param_args['r'])  # 算出来的graphon

        rec_loss = fgw_distance(
            torch.tensor(graphon_oracle,
                         dtype=torch.double,
                         device=graphon_est.device), graphon_est.double(),
            args)  # TODO: ADD LOSS

        y = batch.y.view(pred.shape).to(torch.float64)
        is_valid = y**2 > 0
        loss_mat = criterion(pred.double(), (y + 1) / 2)

        loss_mat = torch.where(
            is_valid, loss_mat,
            torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        cls_loss = torch.sum(loss_mat) / torch.sum(
            is_valid)  # overall classification loss
        meter.update(pred, y, mask=is_valid)

        loss_reg_head = head_regularization()
        loss = param_args['cls_weight'] * cls_loss + \
               param_args['trade_off_head'] * loss_reg_head + \
               param_args['rec_weight'] * 0.1 * rec_loss  # + \
        #    param_args['reg_loss'] * reg_loss
        # args.trade_off_backbone * loss_reg_backbone + \
        # args.trade_off_bss * loss_bss + \

        # loss = loss + 0.1 * loss_weights

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=10)
        optimizer.step()

        loss_epoch.append(cls_loss.item())

        batch_iter.set_postfix({
            "cls_loss":
            cls_loss.item(),
            "head_loss":
            loss_reg_head.item(),
            "rec_loss":
            rec_loss.item(),
            "roc_auc":
            meter.compute_metric('roc_auc_score_finetune')
        })
    avg_loss = sum(loss_epoch) / len(loss_epoch)

    if scheduler is not None: scheduler.step(cls_loss)
    # print(
    #     f'{"vanilla model || " if fea is None and args.norm_type == "none" else ""} '
    #     f'cls_loss:{avg_loss:.5f}, loss_reg_backbone: {args.trade_off_backbone * loss_reg_backbone:.5f} loss_reg_head:'
    #     f' {args.trade_off_head * loss_reg_head:.5f} bss_los: {args.trade_off_bss * loss_bss:.5f} '
    #     + print_str)
    # try:
    #     print('num_oversmooth:',
    #           backbone_regularization.num_oversmooth,
    #           end=' || ')
    #     backbone_regularization.num_oversmooth = 0
    # except:
    #     pass

    metric = np.mean(meter.compute_metric('roc_auc_score_finetune'))
    return metric, avg_loss


def eval(args, model, device, loader):
    model.eval()

    loss_sum = []
    eval_meter = Meter()
    for step, batch in enumerate(tqdm(loader, desc="Iteration", disable=True)):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr,
                         batch.batch)
            y = batch.y.view(pred.shape)
            eval_meter.update(pred, y, mask=y**2 > 0)

            is_valid = y**2 > 0
            # Loss matrix
            loss_mat = criterion(pred.double(), (y + 1.0) / 2)
            # loss matrix after removing null target
            loss_mat = torch.where(
                is_valid, loss_mat,
                torch.zeros(loss_mat.shape).to(loss_mat.device).to(
                    loss_mat.dtype))

            cls_loss = torch.sum(loss_mat) / torch.sum(is_valid)
            loss_sum.append(cls_loss.item())

    metric = np.mean(eval_meter.compute_metric('roc_auc_score_finetune'))
    return metric, sum(loss_sum) / len(loss_sum)


def Inference(args,
              model,
              device,
              loader,
              source_getter,
              target_getter,
              plot_confusion_mat=False):
    model.eval()

    loss_sum = []
    eval_meter = Meter()
    for step, batch in enumerate(tqdm(loader, desc="Iteration", disable=True)):
        batch = batch.to(device)

        with torch.no_grad():
            intermediate_output_t, output_t = target_getter(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = output_t

        y = batch.y.view(pred.shape)
        eval_meter.update(pred, y, mask=y**2 > 0)

    metric = np.mean(eval_meter.compute_metric('roc_auc_score_finetune'))

    return metric, sum(loss_sum)


def main(args, param_args):
    device = torch.device("cuda:" + str(param_args['device'])
                          ) if torch.cuda.is_available() else torch.device(
                              "cpu")
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    elif args.dataset == "zinc_standard_agent":
        num_tasks = 1
    else:
        raise ValueError("Invalid dataset name.")
    args.num_tasks = num_tasks
    # set up dataset
    dataset = MoleculeDataset(os.path.join(args.data_path, args.dataset),
                              dataset=args.dataset)

    print(dataset)  # MoleculeDataset(7831) for tox21
    train_val_test = [0.8, 0.1, 0.1]
    if args.split == "scaffold":

        smiles_list = pd.read_csv(os.path.join(args.data_path, args.dataset,
                                               'processed/smiles.csv'),
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset,
            smiles_list,
            null_value=0,
            frac_train=train_val_test[0],
            frac_valid=train_val_test[1],
            frac_test=train_val_test[2],
            train_radio=args.train_radio)

        print(
            f"scaffold, train:test:val={len(train_dataset)}:{len(valid_dataset)}:{len(test_dataset)}, train_radio:{args.train_radio}"
        )
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset,
            null_value=0,
            frac_train=1.0,
            frac_valid=0.0,
            frac_test=0.0,
            seed=args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv(os.path.join(args.data_path, args.dataset,
                                               'processed/smiles.csv'),
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset,
            smiles_list,
            null_value=0,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    # print("train_dataset[0]", train_dataset[0])
    shuffle = True
    if args.debug:
        shuffle = False
    train_loader = DataLoader(train_dataset,
                              batch_size=param_args['batch_size'],
                              shuffle=shuffle,
                              num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset,
                            batch_size=param_args['batch_size'],
                            shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset,
                             batch_size=param_args['batch_size'],
                             shuffle=False,
                             num_workers=args.num_workers)

    # set up model
    ## finetuned model
    model = GNN_graphpred(
        args.num_layer,
        param_args['emb_dim'],
        num_tasks,
        JK=args.JK,
        drop_ratio=param_args['dropout_ratio'],
        graph_pooling=args.graph_pooling,
        gnn_type=args.gnn_type,
    )
    ## pretrained model
    # source_model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio,
    #                              graph_pooling=args.graph_pooling, gnn_type=args.gnn_type, args=args)
    print(param_args['emb_dim'])
    WEncoder = GraphonNewEncoder(param_args['emb_dim'],
                                 param_args['emb_dim'],
                                 param_args['n_factors'],
                                 encoder_type=param_args['encoder_type'])
    # WEncoder = GraphonEncoder(args.emb_dim, args.emb_dim, args.basis_dim)
    # node_type = "continuous"
    WDecoder = GraphonFactorization(num_factors=param_args['n_factors'],
                                    graphs=dataset,
                                    seed=param_args['runseed'],
                                    param_args=param_args)

    if not args.input_model_file in ["", 'none'] and os.path.isfile(
            args.input_model_file):
        print('loading pretrain model from', args.input_model_file)
        model.from_pretrained(args.input_model_file)
        # source_model.from_pretrained(args.input_model_file)

    model.to(device)
    WEncoder.to(device)
    WDecoder.to(device)

    # TODO: freeze method
    # for param in source_model.parameters():
    #     param.requires_grad = False
    # source_model.eval()

    ## one of baseline methods: StochNorm
    if args.norm_type == 'stochnorm':
        print('converting model with strochnorm')
        model = convert_model(model, p=args.prob)
        # source_model = convert_model(source_model, p=args.prob)

    # set up optimizer
    # different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({
            "params": model.pool.parameters(),
            "lr": args.lr * args.lr_scale
        })
    model_param_group.append({"params": WEncoder.parameters()})
    model_param_group.append({"params": WDecoder.parameters()})  # TODO: check
    model_param_group.append({
        "params": model.graph_pred_linear.parameters(),
        "lr": args.lr * args.lr_scale
    })
    optimizer = optim.Adam(model_param_group,
                           lr=args.lr,
                           weight_decay=args.decay)
    # print(model)
    # create intermediate layer getter
    if args.gnn_type == 'gin':
        # return_layers = ['gnn.gnns.4.mlp.2']
        return_layers = ['gnn.gnns.4.mlp.2']

    elif args.gnn_type == 'gcn':
        return_layers = ['gnn.gnns.4.linear']

    elif args.gnn_type in ['gat', 'gat_ot']:
        return_layers = ['gnn.gnns.4.weight_linear']
    else:
        raise NotImplementedError(args.gnn_type)

    target_getter = IntermediateLayerGetter(model, return_layers=return_layers)

    # get regularization for finetune
    # weights_regularization = FrobeniusRegularization(source_model.gnn, model.gnn)
    backbone_regularization = lambda x: x
    bss_regularization = lambda x: x

    if args.regularization_type in ['gtot_feature_map']:
        ''' the proposed method GTOT-tuning'''
        from ftlib.finetune.gtot_tuning import GTOTRegularization
        backbone_regularization = GTOTRegularization(order=args.gtot_order,
                                                     args=args)
    # ------------------------------ baselines --------------------------------------------
    elif args.regularization_type == 'l2_sp':
        backbone_regularization = SPRegularization(source_model.gnn, model.gnn)

    elif args.regularization_type == 'feature_map':
        from ftlib.finetune.delta import BehavioralRegularization
        backbone_regularization = BehavioralRegularization()

    elif args.regularization_type == 'attention_feature_map':
        from ftlib.finetune.delta import AttentionBehavioralRegularization
        attention_file = os.path.join(
            'delta_attention',
            f'{args.gnn_type}_{args.dataset}_{args.attention_file}')
        if os.path.exists(attention_file):
            print("Loading channel attention from", attention_file)
            attention = torch.load(attention_file)
            attention = [a.to(device) for a in attention]
        else:
            print('attention_file', attention_file)
            attention = calculate_channel_attention(train_dataset,
                                                    return_layers, args,
                                                    param_args)
            torch.save(attention, attention_file)

        backbone_regularization = AttentionBehavioralRegularization(attention)

    elif args.regularization_type == 'bss':
        bss_regularization = BatchSpectralShrinkage(k=args.k)
        if args.debug:
            from ftlib.finetune.gtot_tuning import GTOTRegularization
            backbone_regularization = GTOTRegularization(order=args.gtot_order,
                                                         args=args)
    # ------------------------------ end --------------------------------------------
    elif args.regularization_type == 'none':
        backbone_regularization = lambda x: x
        bss_regularization = lambda x: x
        pass
    else:
        raise NotImplementedError(args.regularization_type)

    head_regularization = L2Regularization(
        nn.ModuleList([model.graph_pred_linear]))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.2,
        patience=6,
        verbose=False,
        threshold=0.0001,
        threshold_mode='rel',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08)
    save_model_name = os.path.join(
        args.save_path, f'{args.gnn_type}_{args.dataset}_{args.tag}.pt')
    stopper = EarlyStopping(mode='higher',
                            patience=args.patience,
                            filename=save_model_name)

    training_time = Runtime()
    test_time = Runtime()
    best_val = 0
    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch), " lr: ",
              optimizer.param_groups[-1]['lr'])
        training_time.epoch_start()
        train_acc, train_loss = train_epoch(
            args,
            model,
            device,
            train_loader,
            optimizer,
            # weights_regularization,
            backbone_regularization,  # the central regular GTOTRegularization
            head_regularization,  # L2Regularization for graph_pred_linear
            target_getter,
            # source_getter,
            WEncoder,
            WDecoder,
            bss_regularization,  # gtot_feature_map then None
            scheduler=scheduler,
            epoch=epoch,
            param_args=param_args)
        training_time.epoch_end()

        print("====Evaluation")

        test_time.epoch_start()
        val_acc, val_loss = eval(args, model, device, val_loader)
        if val_acc > best_val:
            test_acc, test_loss = eval(args, model, device, test_loader)
            nni.report_intermediate_result(test_acc)
        test_time.epoch_end()
        # try:
        #     scheduler.step(-val_acc)
        # except:
        #     scheduler.step()

        if stopper.step(val_acc,
                        model,
                        test_score=test_acc,
                        IsMaster=args.debug):
            stopper.report_final_results(i_epoch=epoch)
            break
        stopper.print_best_results(i_epoch=epoch,
                                   val_cls_loss=val_loss,
                                   train_acc=train_acc,
                                   val_score=val_acc,
                                   test_socre=test_acc,
                                   gnn_type=args.gnn_type,
                                   dataset=args.dataset,
                                   tag=args.tag)
        file_name = str(args.dataset) + "_" + str(args.nnodes) + "_" + str(
            args.dropout_ratio) + "_" + str(args.cls_weight) + "_" + str(
                args.rec_weight) + "_" + str(args.trade_off_head) + ".pt"

        print(training_time.time_epoch_list)
        print(test_time.time_epoch_list)
        print('===============================')
    ## only inference, should set the --epoch 0
    inference = False
    if inference and args.debug:
        print('Inferencing')
        if stopper.best_model is None:
            print(stopper.filename)
            stopper.load_checkpoint(model)
            print('checkpint args', model.args)
        else:
            model.load_state_dict(stopper.best_model)
        model.to(device)
        test_acc, test_loss = Inference(args,
                                        model,
                                        device,
                                        test_loader,
                                        source_getter,
                                        target_getter,
                                        plot_confusion_mat=True)
        print(f'inference test_acc:{test_acc:.5f}')
        return test_acc, stopper.best_epoch, training_time

    training_time.print_mean_sum_time(prefix='Training')
    test_time.print_mean_sum_time(prefix='Test')

    return stopper.best_test_score, stopper.best_epoch, training_time


if __name__ == "__main__":
    args = parseArgs.parser.parse_args()
    param_args = args.__dict__
    param_args.update(nni.get_next_parameter())  # 返回none
    print(param_args)
    elapsed_times = []
    # seed_nums = [
    #     42,
    # ]  # 这里把随机种子固定
    seed_nums=range(5)
    if param_args['debug']:
        seed_nums = [param_args['runseed']]
        if 'inference' in param_args['tag']:
            seed_nums = [42]
    ## 10 random seeds
    results = []
    for seed_num in seed_nums:
        print(f"seed:{seed_num}/{seed_nums}")
        param_args['runseed'] = seed_num
        setup_seed(seed_num)
        test_acc, best_epoch, training_time_epoch = main(args, param_args)
        results.append(test_acc)
        print(f'avg_test_acc={sum(results) / len(results):.5f}')
        elapsed_times.append(training_time_epoch.sum_elapsed_time())
        print(f'Seed {seed_num}/{max(seed_nums)} Acc array: ', results)
        # print(args)

    results = np.array(results)
    nni.report_final_result(float(results.mean()))
    elapsed_times = np.array(elapsed_times)
    print(f'avg_test_acc={results.mean():.5f}')
    print(
        f"avg_test_acc={results.mean() * 100:.3f}$\pm${results.std() * 100:.3f}\n  "
        f"elapsed_times:{elapsed_times.mean():.4f}+-{elapsed_times.std():.4f}s."
    )
