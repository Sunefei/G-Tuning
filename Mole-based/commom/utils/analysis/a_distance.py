from torch.utils.data import TensorDataset
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler

try:
    from commom.utils.metric import binary_accuracy
    from commom.utils.meter import AverageMeter
except:
    from ..meter import AverageMeter
    from ..metric import binary_accuracy


class ANet(nn.Module):
    def __init__(self, in_feature):
        super(ANet, self).__init__()
        self.layer = nn.Linear(in_feature, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        return x


def calculate(source_feature: torch.Tensor, target_feature: torch.Tensor,
              device, progress=True, training_epochs=10):
    """
    Calculate the :math:`\mathcal{A}`-distance, which is a measure for distribution discrepancy.

    The definition is :math:`dist_\mathcal{A} = 2 (1-2\epsilon)`, where :math:`\epsilon` is the
    test error of a classifier trained to discriminate the source from the target.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        device (torch.device)
        progress (bool): if True, displays a the progress of training A-Net
        training_epochs (int): the number of epochs when training the classifier

    Returns:
        :math:`\mathcal{A}`-distance
    """

    source_label = torch.ones((source_feature.shape[0], 1))
    target_label = torch.zeros((target_feature.shape[0], 1))
    rus = RandomUnderSampler(random_state=0,
                             sampling_strategy=0.5)  # sampling_strategy = N_m/N_sM (最小类别样本数/ 最大类别样本数在采样后的数目)

    feature = torch.cat([source_feature, target_feature], dim=0)
    label = torch.cat([source_label, target_label], dim=0)
    feature, label = rus.fit_resample(feature.cpu().numpy(), label.cpu().numpy())
    feature, label = torch.tensor(feature), torch.tensor(label)
    label = label.unsqueeze(-1)
    dataset = TensorDataset(feature, label)
    length = len(dataset)
    print('Data size:', length)
    train_size = int(0.8 * length)
    val_size = length - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    anet = ANet(feature.shape[1]).to(device)
    optimizer = SGD(anet.parameters(), lr=0.01)
    a_distance = 2.0
    for epoch in range(training_epochs):
        anet.train()
        for (x, label) in train_loader:
            x = x.to(device)
            label = label.to(device)
            anet.zero_grad()
            y = anet(x)
            loss = F.binary_cross_entropy(y, label)
            loss.backward()
            optimizer.step()

        anet.eval()
        y_true = []
        y_pred = []
        meter = AverageMeter("accuracy", ":4.2f")
        with torch.no_grad():
            for (x, label) in val_loader:
                x = x.to(device)
                label = label.to(device)
                y = anet(x)
                y_true.append(label.detach())
                y_pred.append(y.detach())
                acc = binary_accuracy(y, label)
                meter.update(acc, x.shape[0])
        error = 1 - meter.avg / 100
        a_distance = 2 * (1 - 2 * error)
        if progress:
            print("epoch {} accuracy: {} A-dist: {}".format(epoch, meter.avg, a_distance), end=' ')
            print(f'error:{error}')
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        y_pred = (y_pred >= 0.5).float()
        cm = confusion_matrix(y_true.cpu().squeeze(-1).numpy(), y_pred.cpu().squeeze(-1).numpy())
        print(cm)

    '''                     predicted negative class | predicted positive class
    autual negative class         TN                           FP
    autual positive class         FN                           TP
    '''

    return a_distance


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # target_dataset = 'clintox'
    # target_dataset = 'toxcast'
    # target_dataset = 'bace'
    # target_dataset = 'bbbp'
    # target_dataset = 'muv'
    target_dataset = 'hiv'
    # target_dataset = 'tox21'
    # target_dataset = 'sider'
    source_dataset = 'zinc_standard_agent'
    print(f'{source_dataset} vs {target_dataset}')
    try:
        target_feas = torch.load(
            f'embs/{target_dataset}_gin_feas.pt',
            map_location='cpu')
        source_feas = torch.load(
            f'embs/{source_dataset}_gin_feas.pt',
            map_location='cpu')
    except:
        source_feas = torch.load(f'embs/{source_dataset}_gin_feas.pt',
                                 map_location='cpu')
        target_feas = torch.load(f'embs/{target_dataset}_gin_feas.pt',
                                 map_location='cpu')
    A_distance = calculate(source_feas.cpu(), target_feas.cpu(), device=device)
    print(f'{source_dataset} vs {target_dataset} a_distance:{A_distance}')
