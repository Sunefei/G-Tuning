# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Early stopping"""
# pylint: disable= no-member, arguments-differ, invalid-name

import datetime
import torch
import copy

__all__ = ['EarlyStopping']


# pylint: disable=C0103
class EarlyStopping(object):
    """Early stop tracker

    Save model checkpoint when observing a performance improvement on
    the validation set and early stop if improvement has not been
    observed for a particular number of epochs.

    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
        If ``metric`` is not None, then mode will be determined
        automatically from that.
    patience : int
        The early stopping will happen if we do not observe performance
        improvement for ``patience`` consecutive epochs.
    filename : str or None
        Filename for storing the model checkpoint. If not specified,
        we will automatically generate a file starting with ``early_stop``
        based on the current time.
    metric : str or None
        A metric name that can be used to identify if a higher value is
        better, or vice versa. Default to None. Valid options include:
        ``'r2'``, ``'mae'``, ``'rmse'``, ``'roc_auc_score'``.

    Examples
    --------
    Below gives a demo for a fake training process.

    >>> import torch
    >>> import torch.nn as nn
    >>> from torch.nn import MSELoss
    >>> from torch.optim import Adam
    >>> from dgllife.utils import EarlyStopping

    >>> model = nn.Linear(1, 1)
    >>> criterion = MSELoss()
    >>> # For MSE, the lower, the better
    >>> stopper = EarlyStopping(mode='lower', filename='test.pth')
    >>> optimizer = Adam(params=model.parameters(), lr=1e-3)

    >>> for epoch in range(1000):
    >>>     x = torch.randn(1, 1) # Fake input
    >>>     y = torch.randn(1, 1) # Fake label
    >>>     pred = model(x)
    >>>     loss = criterion(y, pred)
    >>>     optimizer.zero_grad()
    >>>     loss.backward()
    >>>     optimizer.step()
    >>>     early_stop = stopper.step(loss.detach().data, model)
    >>>     if early_stop:
    >>>         break

    >>> # Load the final parameters saved by the model
    >>> stopper.load_checkpoint(model)
    """

    def __init__(self, mode='higher', patience=10, filename=None, metric=None):
        if filename is None:
            dt = datetime.datetime.now()
            filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
                dt.date(), dt.hour, dt.minute, dt.second)

        if metric is not None:
            assert metric in ['r2', 'mae', 'rmse', 'roc_auc_score'], \
                "Expect metric to be 'r2' or 'mae' or " \
                "'rmse' or 'roc_auc_score', got {}".format(metric)
            if metric in ['r2', 'roc_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.best_epoch = -1
        self.best_test_score = None
        self.best_model = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        """Check if the new score is higher than the previous best score.

        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.

        Returns
        -------
        bool
            Whether the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """Check if the new score is lower than the previous best score.

        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.

        Returns
        -------
        bool
            Whether the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model, test_score=None, IsMaster=True):
        """Update based on a new score.

        The new score is typically model performance on the validation set
        for a new epoch.

        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.

        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_test_score = test_score if test_score is not None else -100
            if IsMaster:
                self.save_checkpoint(model)
            self.best_model = copy.deepcopy(model.state_dict())

        elif self._check(score, self.best_score):
            self.best_score = score
            self.best_test_score = test_score
            if IsMaster:
                self.save_checkpoint(model)
            self.best_model = copy.deepcopy(model.state_dict())

            self.counter = 0
        else:
            self.counter += 1
            # if IsMaster:
            #     print(
            #         f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.

        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        import copy
        args2 = copy.deepcopy(model.args)
        # non_load_keys = ['smiles_to_graph', 'pdb_to_graph', 'node_featurizer', 'edge_featurizer',
        #                  'pdb_node_featurizer', 'pdb_edge_featurizer',
        #                  'device', 'batch_size', 'data_path', 'reload']
        # for k in non_load_keys:
        #     del args2[k]
        torch.save({'model_state_dict': model.state_dict(), 'args': args2}, self.filename)

    def load_checkpoint(self, model):
        '''Load the latest checkpoint

        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        checkpoint=torch.load(self.filename)

        # args = checkpoint.get('args',None)
        model.load_state_dict(checkpoint['model_state_dict'])

    def load_best_model(self, model):
        '''Load the latest checkpoint

        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        model.load_state_dict(self.best_model)
        self.save_checkpoint(model)

    def report_final_results(self, i_epoch=None):
        print(f'Early stop!')
        if i_epoch is not None:
            self.best_epoch = i_epoch - self.counter
            print(f'best_epoch={self.best_epoch}')
        print(f'best_val_score={self.best_score:.6f}')
        if self.best_test_score is not None: print(f'best_test_socre={self.best_test_score:.6f}')

    def print_best_results(self, i_epoch=None, **kargs):
        s = ''
        for k, v in kargs.items():
            if type(v) == str:
                s += (k + '=' + v + ' || ')
            else:
                s += (k + '=' + str(v)[:7] + ' || ')
        # val_score=kargs.get('val_score',None)
        # test_score = kargs.get('test_socre',None)
        # if val_score is not None or test_score is not None:
        print(s, end='')
        if i_epoch is not None:
            self.best_epoch = i_epoch - self.counter
            print(f'best_epoch={self.best_epoch}', end=' || ')
        print(f'best_val_score={self.best_score:.6f}', end=' || ')
        if self.best_test_score is not None: print(f'best_test_socre={self.best_test_score:.6f}')
