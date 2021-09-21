from typing import Union
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import scipy.sparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import (random,
                          coo_matrix,
                          csr_matrix,
                          vstack)
from tqdm import tqdm
import os
class SparseDataset(Dataset):
    """
    Custom Dataset class for scipy sparse matrix
    """
    def __init__(self, data:Union[np.ndarray, coo_matrix, csr_matrix],
                 # targets:Union[np.ndarray, coo_matrix, csr_matrix],
                 transform:bool = None):

        # Transform data coo_matrix to csr_matrix for indexing
        if type(data) == coo_matrix:
            self.data = data.tocsr()
        else:
            self.data = data

        # Transform targets coo_matrix to csr_matrix for indexing
        # if type(targets) == coo_matrix:
        #     self.targets = targets.tocsr()
        # else:
        #     self.targets = targets

        self.transform = transform # Can be removed

    def __getitem__(self, index:int):
        return self.data[index] #, self.targets[index]

    def __len__(self):
        return self.data.shape[0]

def sparse_coo_to_tensor(coo:coo_matrix):
    """
    Transform scipy coo matrix to pytorch sparse tensor
    """
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = torch.Size(shape)

    return torch.sparse.FloatTensor(i, v, s).unsqueeze(1)

def sparse_batch_collate(data_batch):
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    # data_batch, targets_batch = zip(*batch)
    if type(data_batch[0]) == csr_matrix:
        data_batch = vstack(data_batch).tocoo()
        data_batch = sparse_coo_to_tensor(data_batch)
    else:
        data_batch = torch.FloatTensor(data_batch)

    # if type(targets_batch[0]) == csr_matrix:
    #     targets_batch = vstack(targets_batch).tocoo()
    #     targets_batch = sparse_coo_to_tensor(targets_batch)
    # else:
    #     targets_batch = torch.FloatTensor(targets_batch)
    return data_batch#, targets_batch


class AutoEncoder(nn.Module):
    def __init__(self, n_latent, n_init):
        super(AutoEncoder, self).__init__()
        # self.pool_op = torch.nn.AvgPool1d(2, )
        # torch.nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        input_size = n_init
        output_size = input_size // 59
        self.e1 = nn.Linear(input_size, output_size, bias=False)
        input_size = output_size
        output_size = input_size // 8
        self.e2 = nn.Linear(input_size, output_size, bias=False)
        # input_size = output_size
        # output_size = input_size // 2
        # self.e3 = nn.Linear(input_size, output_size, bias=False)
        # input_size = output_size
        # output_size = input_size // 2
        # self.e4 = nn.Linear(input_size, output_size, bias=False)
        input_size_d = output_size
        output_size = n_latent
        self.e3 = nn.Linear(input_size_d, output_size, bias=False)


        input_size = output_size
        output_size = input_size_d
        self.d1 = nn.Linear(input_size, output_size, bias=False)
        input_size = output_size
        output_size = input_size * 8
        self.d2 = nn.Linear(input_size, output_size, bias=False)
        input_size = output_size
        output_size = input_size * 59
        self.d3 = nn.Linear(input_size, output_size, bias=False)
        # input_size = output_size
        # output_size = input_size * 32
        # self.d4 = nn.Linear(input_size, output_size, bias=False)
        # input_size = output_size
        # output_size = input_size * 24
        # self.d5 = nn.Linear(input_size, output_size, bias=False)
        self.leackyRelu1 = nn.LeakyReLU(0.1)
        self.leackyRelu2 = nn.LeakyReLU(0.1)
        self.leackyRelu3 = nn.LeakyReLU(0.1)
        self.leackyRelu4 = nn.LeakyReLU(0.1)

        self.leackyRelu5 = nn.LeakyReLU(0.1)

    def forward(self, x):

        x = self.leackyRelu1(self.e1(x))
        x = self.leackyRelu2(self.e2(x))
        # x = torch.relu(self.e3(x))
        # x = torch.relu(self.e4(x))
        latent = self.leackyRelu3(self.e3(x))

        x = self.leackyRelu4(self.d1(latent))
        x = self.leackyRelu5(self.d2(x))
        # x = torch.relu(self.d3(x))
        # x = torch.relu(self.d4(x))
        x = torch.sigmoid(self.d3(x))
        return x, latent

def kl_divergence(rho, rho_hat, device):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0) # sigmoid because we need the probability distributions
    rho = torch.tensor([rho] * len(rho_hat)).to(device)
    return torch.sum(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))
# define the sparse loss function

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, input, target):
        """ loss function called at runtime """

        # Class 1 - Indices [0:50]
        class_1_loss = F.nll_loss(
            F.log_softmax(input[:, 0:50], dim=1),
            torch.argmax(target[:, 0:50])
        )
        # Class 2 - Indices [50:100]
        class_2_loss = F.nll_loss(
            F.log_softmax(input[:, 50:100], dim=1),
            torch.argmax(target[:, 50:100])
        )
        # Class 3 - Indices [100:150]
        class_3_loss = F.nll_loss(
            F.log_softmax(input[:, 100:150], dim=1),
            torch.argmax(target[:, 100:150])
        )
        return class_1_loss + class_2_loss + class_3_loss

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid acitvation
        # inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


def train(model, train_dataloader, val_dataloader, lr, EPOCHES, Loss, device, use_sparsity = False, Loss2 = None):
    model = model.to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    # optimizer = torch.optim.SGD(model.parameters(), lr)
    optimizer = torch.optim.Adam(model.parameters())
    train_loss = []
    val_loss = []

    learning_rates = [lr] * EPOCHES

    epoch = 0
    for lr in learning_rates:
        model.train()
        optimizer.param_groups[0]['lr'] = lr
        observed = 0
        for i, x in enumerate(tqdm(train_dataloader)):
            # x.unsqueeze(1)
            x = x.to_dense().to('cuda')
            x = Variable(x)
            optimizer.zero_grad()

            reconst_x, latent = model(x)
            loss = Loss(reconst_x, x)

            if use_sparsity:
                rho = 5e-5
                model_children = list(model.children())
                values = x
                sparse_loss = 0
                for j in range(len(model_children)):
                    values = model_children[j](values)
                    sparse_loss += kl_divergence(rho, values, device)
                loss = loss + 0.1 * sparse_loss

            train_loss.append(loss.cpu().data.item())
            if Loss2 is not None:
                loss += 0.1 * Loss2(reconst_x, x)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
            optimizer.step()

            counter = 0
            running_loss = 0
            with torch.no_grad():
                for n, v in enumerate(val_dataloader):
                    counter += 1
                    v.unsqueeze(1)
                    v = v.to_dense()
                    v = v.to(device)
                    reconst_v, latent_v = model(v)
                    vloss = Loss(reconst_v, v)
                    running_loss += vloss.cpu().item()

                val_loss.append(running_loss / counter)
            observed += x.shape[0]
            if i % 5 == 0 or i == len(train_dataloader):

                print('\r Train Epoch: {} [{}/{} ({:.0f}%)] with lr: {} \ttrain_loss: {:e} val_loss: {:e}, n_pos :{} {}\n'.format(
                    epoch + 1,
                    observed,
                    len(train_dataloader.dataset),
                    observed / len(train_dataloader.dataset) * 100.,
                    lr,
                    loss.cpu().item(), val_loss[-1],
                    reconst_x.sum(),
                    x.sum()),
                    end='')

        epoch += 1
        torch.save(model, os.path.join(os.getcwd(), 'conv_ae' + str(epoch)))


            

