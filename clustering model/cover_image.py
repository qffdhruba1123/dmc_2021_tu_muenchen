# transform = transforms.Compose([transforms.Resize(255),
#                                 transforms.CenterCrop(224),
#                                 transforms.ToTensor()])
# dataset = datasets.ImageFolder(os.path.join(os.getcwd(), 'Cover_Pics'), transform=transform)
import os, imageio
from textwrap import wrap
from PIL import Image
import torch
import argparse
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torchvision
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.distributions import kl_divergence, Normal
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
class Dataset(Dataset):
    """
    Custom Dataset class for scipy sparse matrix
    """
    def __init__(self, data,
                 targets,
                 transform:bool = None):

        # Transform data coo_matrix to csr_matrix for indexing
        self.data = data
        # Transform targets coo_matrix to csr_matrix for indexing
        self.targets = targets

        self.transform = transform # Can be removed

    def __getitem__(self, index:int):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]

#
# def train_val_dataset(dataset, val_split=0.25):
#     train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
#     datasets = {}
#     datasets['train'] = Subset(dataset, train_idx)
#     datasets['val'] = Subset(dataset, val_idx)
#     return datasets

import numpy as np
import argparse
import random
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data
from scipy.io import loadmat
from GMVAE.model.GMVAE import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class GMVAE(torch.nn.Module):
    def __init__(self, input_dims=128, hidden_dims=256, z_dims=64, n_component=10):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=8,
                      kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8,
                      out_channels=16,
                      kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=16,
                      kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()

        )

        self.mu, self.var = nn.Linear(hidden_dims, z_dims), nn.Linear(hidden_dims, z_dims)
        self.dec_init = nn.Linear(z_dims, hidden_dims)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16,
                               64,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64,
                               32,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32,
                               16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16,
                               8,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8,
                               3,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

        self.n_component = n_component
        self.z_dims = z_dims

        self._build_mu_lookup()
        self._build_logvar_lookup(pow_exp=-2)

    def encode(self, x):
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        q_zx = self.encoder(x)
        q_zx_flatten = q_zx.view(q_zx.shape[0], -1)
        mu, var = self.mu(q_zx_flatten), self.var(q_zx_flatten).exp_()
        z = repar(mu, var)

        log_logits_z, cls_z_prob = self.approx_qy_x(z, self.mu_lookup,
                                                    self.logvar_lookup,
                                                    n_component=self.n_component)

        return z, mu, var, log_logits_z, cls_z_prob

    def decode(self, z):
        h = self.dec_init(z)
        h = h.view(h.shape[0], 16, 7, 7)
        x_hat = self.decoder(h)

        return x_hat

    def forward(self, x):
        z, mu, var, log_logits_z, cls_z_prob = self.encode(x)
        x_hat = self.decode(z)

        return x_hat, z, mu, var, log_logits_z, cls_z_prob

    def _build_mu_lookup(self):
        mu_lookup = nn.Embedding(self.n_component, self.z_dims)
        nn.init.xavier_uniform_(mu_lookup.weight, gain=1.0)
        mu_lookup.weight.requires_grad = True
        self.mu_lookup = mu_lookup

    def _build_logvar_lookup(self, pow_exp=0, logvar_trainable=False):
        logvar_lookup = nn.Embedding(self.n_component, self.z_dims)
        init_sigma = np.exp(pow_exp)
        init_logvar = np.log(init_sigma ** 2)
        nn.init.constant_(logvar_lookup.weight, init_logvar)
        logvar_lookup.weight.requires_grad = logvar_trainable
        self.logvar_lookup = logvar_lookup

    def approx_qy_x(self, z, mu_lookup, logvar_lookup, n_component):
        def log_gauss_lh(z, mu, logvar):
            """
            Calculate p(z|y), the likelihood of z w.r.t. a Gaussian component
            """
            llh = - 0.5 * (torch.pow(z - mu, 2) / torch.exp(logvar) + logvar + np.log(2 * np.pi))
            llh = torch.sum(llh, dim=1)  # sum over dimensions
            return llh

        logLogit_qy_x = torch.zeros(z.shape[0], n_component).cuda()  # log-logit of q(y|x)
        for k_i in torch.arange(0, n_component):
            mu_k, logvar_k = mu_lookup(k_i.cuda()), logvar_lookup(k_i.cuda())
            logLogit_qy_x[:, k_i] = log_gauss_lh(z, mu_k, logvar_k) + np.log(1 / n_component)

        qy_x = torch.nn.functional.softmax(logLogit_qy_x, dim=1)
        return logLogit_qy_x, qy_x

    def loss_function(self, step, x_hat, x, mu, var, cls_z_prob, labels, beta=1):
        # kl annealing
        beta_1 = min(step / 1000 * beta, beta)

        recon_loss = torch.nn.BCELoss()(x_hat, x)
        recon_loss_2 = torch.nn.BCELoss(reduce=False)(x_hat, x).mean(-1)

        mu_ref, var_ref = self.mu_lookup(labels.cuda().long()), \
                          self.logvar_lookup(labels.cuda().long()).exp_()
        dis_ref = Normal(mu_ref, var_ref)
        dis = Normal(mu, var)
        kl_loss = kl_divergence(dis, dis_ref).mean()

        cls_loss = torch.nn.CrossEntropyLoss()(cls_z_prob, labels.cuda().long())


        loss = recon_loss + beta_1 * kl_loss + cls_loss

        return loss, recon_loss, kl_loss


class VAE(nn.Module):
    def __init__(self, n_latent, init_dim):
        super(VAE, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(16)
        # Latent vectors mu and sigma
        self.n_latent = n_latent
        self.init_size = init_dim

        self.final_size = init_dim // 2 // 2 // 2 // 2 // 2
        self.fc1 = nn.Linear(self.final_size * self.final_size * 16, self.n_latent)
        self.fc_bn1 = nn.BatchNorm1d(self.n_latent)
        self.fc21 = nn.Linear(self.n_latent, self.n_latent)
        self.fc22 = nn.Linear(self.n_latent, self.n_latent)
        # Sampling vector
        self.fc3 = nn.Linear(self.n_latent, self.n_latent)
        self.fc_bn3 = nn.BatchNorm1d(self.n_latent)
        self.fc4 = nn.Linear(self.n_latent, self.final_size * self.final_size * 16)
        self.fc_bn4 = nn.BatchNorm1d(self.final_size * self.final_size * 16)
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.debn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1,bias=False)
        self.debn2 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.debn3 = nn.BatchNorm2d(16)
        self.deconv4 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.debn4 = nn.BatchNorm2d(8)

        self.deconv5 = nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3)))
        conv5 = self.relu(self.bn5(self.conv5(conv4))).view(-1, self.final_size * self.final_size * 16)


        fc1 = self.relu(self.fc_bn1(self.fc1(conv5)))
        mu = self.fc21(fc1)
        logvar = self.fc22(fc1)


        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        z = eps.mul(std).add_(mu)

        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, self.final_size, self.final_size)
        deconv1 = self.relu(self.debn1(self.deconv1(fc4)))
        deconv2 = self.relu(self.debn2(self.deconv2(deconv1)))
        deconv3 = self.relu(self.debn3(self.deconv3(deconv2)))
        deconv4 = self.relu(self.debn4(self.deconv4(deconv3)))

        out = self.sigmoid(self.deconv5(deconv4)).view(-1, 3, self.init_size, self.init_size)
        return out, mu, std, z


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
    def forward(self, recon_x, x, mu, logvar):
        MSE = self.mse_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2)-logvar.exp())
        return MSE + KLD
def kl_divergence(rho, rho_hat, device):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0) # sigmoid because we need the probability distributions
    rho = torch.tensor([rho] * len(rho_hat)).to(device)
    return torch.sum(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))
# define the sparse loss function


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n.rsplit('.')[0])
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=max(max_grads).cpu().numpy())  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

def train(model, train_dataloader, val_dataloader, lr, EPOCHES, device):
    model = model.to(device)
    loss_mse = Loss()

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
        for i, (x, y) in enumerate(tqdm(train_dataloader)):
            x = torch.divide(x, 255).to(device)
            optimizer.zero_grad()


            out, mu, std, _ = model(x)
            loss = loss_mse(out, x, mu, std)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5e5)
            optimizer.step()

            counter = 0
            running_loss = 0

            observed += x.shape[0]
            if i % 5 == 0 or i == len(train_dataloader):

                with torch.no_grad():
                    for n, (v, _w) in enumerate(val_dataloader):
                        v = torch.divide(v, 255).to(device)
                        counter += v.shape[0]
                        v_out, v_mu, v_std, _ = model(v)
                        vloss = loss_mse(v_out, v, v_mu, v_std)
                        running_loss += vloss.cpu().item()

                train_loss.append(loss.cpu().data.item() / x.shape[0])
                val_loss.append(running_loss / counter)
                print(
                    '\r Train Epoch: {} [{}/{} ({:.0f}%)] with lr: {} \ttrain_loss: {:e} val_loss: {:e}\n'.format(
                        epoch + 1,
                        observed,
                        len(train_dataloader.dataset),
                        observed / len(train_dataloader.dataset) * 100.,
                        lr,
                        train_loss[-1], val_loss[-1]),
                    end='')
        # plot_grad_flow(model.named_parameters())
        # plt.show()

        epoch += 1
        plt.plot(train_loss, label = "train loss")
        plt.plot(val_loss, label = "val loss")
        plt.legend()
        plt.show()

        fig = plt.figure(figsize=[10, 5])
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(x[6].permute(1, 2, 0).detach().cpu().numpy())
        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(out[6].permute(1, 2, 0).detach().cpu().numpy())
        plt.show()

        torch.save(model, os.path.join(os.getcwd(), f'conv_vae_{model.n_latent}_' + str(epoch)))

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img
if __name__  == '__main__':

    from torchvision.datasets import ImageFolder
    dim = 128
    dataset = ImageFolder(os.path.join(os.getcwd(), 'Cover_Pics', 'dataset'), transform=Resize((dim, dim)))
    # for i, (data, target) in enumerate(dataset):
    #     orig_path = dataset.imgs[i][0]
    #     id = dataset.imgs[i][0].rsplit('\\')[-1].rsplit('.')[0]
    #     print(id)
    #     _dir = os.path.join(os.getcwd(), 'Cover_Pics', 'transformed_pics', str(id))
    #     if not (os.path.isdir(_dir)):
    #         os.makedirs(_dir)
    #     if not (os.path.isfile(os.path.join(_dir, 'img.png'))):
    #         torchvision.utils.save_image(data, os.path.join(_dir, 'img.png'))

    # dataset = ImageFolder(os.path.join(os.getcwd(), 'Cover_Pics', 'transformed_pics'), transform=ToTensor())
    #
    #
    if not os.path.isfile(f'data_{dim}.npy'):
        images = np.empty((len(dataset), 3, dim, dim), dtype='uint8')
        labels = []
        for i, (data, label) in enumerate(dataset):
            labels.append(dataset.imgs[i][0].rsplit('\\')[-1].rsplit('.')[0])
            images[i] = np.transpose(np.asarray(data), (2, 0, 1))
        np.save('data_' + str(dim), images)
        np.save('labels', labels)


    dataset = np.load(f'data_{dim}.npy')
    labels = np.load('labels.npy')
    labels = [int(l) for l in labels]
    print(len(dataset))
    train_data, val_data = train_test_split(dataset, random_state=1, test_size = 0.1)
    train_label, val_label = train_test_split(labels, random_state=1, test_size = 0.1)

    del dataset, labels
    dataloaders = {}
    train_dataset = TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
    dataloaders['train'] = DataLoader(train_dataset, 1536, shuffle=True, num_workers=0)
    val_dataset = TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))


    dataloaders['val'] = DataLoader(val_dataset, 256, shuffle=True, num_workers=0)
    # x, y = next(iter(dataloaders['train']))
    # print(x.shape, y.shape)
    # fig = plt.figure(figsize=[25, 25])
    # for i in range(0, 100):
    #     z = x[i].numpy()
    #     ax = fig.add_subplot(10, 10, i + 1)
    #     ax.imshow(np.transpose(z, (1, 2, 0)))

    device = 'cuda'
    n_latent = 32
    vae = VAE(n_latent= n_latent, init_dim = dim)

    # vae = torch.load('conv_vae_128_1')
    # vae.load_state_dict(checkpoint['state_dict'])
    # train(vae, dataloaders['train'], dataloaders['val'], 1e-3, 100, device)





    images = np.load(f'data_{dim}.npy')
    labels = np.load('labels.npy')
    labels = [int(l) for l in labels]
    dataset = TensorDataset(torch.Tensor(images), torch.Tensor(labels))

    vae = torch.load(f'conv_vae_{n_latent}_70').to(device)

    latent_variables = np.empty((len(dataset), n_latent))

    vae.eval()
    for i, data in enumerate(dataset):
        _out = vae(data[0].unsqueeze(0).to(device))

        latent_variables[i] = _out[-1].squeeze(0).detach().cpu().numpy()

    min_max_scaler = preprocessing.MinMaxScaler()
    normed_matrix = min_max_scaler.fit_transform(latent_variables)
    np.save('latent_variables', latent_variables)

    neigh = NearestNeighbors(n_neighbors=20, radius=1)
    neigh.fit(normed_matrix)

    import random
    from preprocessing import *
    items, _, _ = preprocessing()

    # indexes = random.sample(range(0, images.shape[0]), 10)
    # imgs = []
    # for j, r in enumerate(indexes):
    #     fig = plt.figure(figsize=[32, 8])
    #
    #     distance, neighbors_index = neigh.kneighbors([normed_matrix[r]], 100, return_distance=True)
    #
    #     for i, ind in enumerate(neighbors_index[0][:6]):
    #         img = images[ind].transpose((1, 2, 0))
    #         if i == 0:
    #             img = img / 255
    #         ax = fig.add_subplot(1, 6, i + 1)
    #         ax.imshow(img)
    #
    #
    #         ax.imshow(img)
    #
    #         if i == 0:
    #             ax.set_xlabel(
    #                     'original item: \n'  + "\n".join(wrap(f'{items[items["itemID"] == labels[r]]["title"].values} ({r})',
    #                         25)))
    #         else:
    #
    #             ax.set_xlabel(f'recommendation {i}: \n' + "\n".join(wrap(f'{items[items["itemID"] == labels[ind]]["title"].values} ({labels[ind]})', 25)))
    #         plt.subplots_adjust(left=0.125,
    #                             bottom=0.1,
    #                             right=0.9,
    #                             top=0.9,
    #                             wspace=0.2,
    #                             hspace=0.35)
    #     img = fig2img(fig)
    #     plt.show()
    #     imgs.append(img)
    # imageio.mimsave(os.path.join('covers.gif'), imgs, duration=3)



    # rec_imgs = []
    # for j, r in enumerate(indexes):
    #     fig = plt.figure(figsize=[32, 8])
    #
    #     distance, neighbors_index = neigh.kneighbors([normed_matrix[r]], 100, return_distance=True)
    #
    #     for i, ind in enumerate(neighbors_index[0][:6]):
    #         img = vae(dataset.tensors[0][ind].unsqueeze(0).to(device))[0].squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))
    #         ax = fig.add_subplot(1, 6, i + 1)
    #         ax.imshow(img)
    #
    #
    #         if i == 0:
    #             ax.set_xlabel(
    #                 'original item: \n' + "\n".join(wrap(f'{items[items["itemID"] == labels[r]]["title"].values} ({r})',
    #                                                      25)))
    #         else:
    #
    #             ax.set_xlabel(f'recommendation {i}: \n' + "\n".join(
    #                 wrap(f'{items[items["itemID"] == labels[ind]]["title"].values} ({labels[ind]})', 25)))
    #         plt.subplots_adjust(left=0.125,
    #                             bottom=0.1,
    #                             right=0.9,
    #                             top=0.9,
    #                             wspace=0.2,
    #                             hspace=0.35)
    #     img = fig2img(fig)
    #     plt.show()
    #     rec_imgs.append(img)
    # imageio.mimsave(os.path.join('rec_imgs.gif'), rec_imgs, duration=5)
    #
    # final_recommendation = pd.read_csv('finalrecs_20210627 - finalrecs_20210627.csv')
    # grouped_rec = final_recommendation.groupby('itemId')
    #
    # for name, group in grouped_rec:
    #
    #     fig = plt.figure(figsize=[32, 8])
    #     indexes = group.rec_id.values
    #     size = len(indexes)
    #     ax = fig.add_subplot(1, size + 1, 1)
    #     if name not in labels:
    #         img = np.zeros((128, 128, 3))
    #     else:
    #         img = images[labels.index(name)].transpose((1, 2, 0))
    #
    #     ax.imshow(img)
    #     ax.set_xlabel('original item: \n' + "\n".join(wrap(
    #         f'{items[items["itemID"] == name]["title"].values} ({name})', 25)))
    #     i = 2
    #     for ind in indexes:
    #         ax = fig.add_subplot(1, size + 1, i)
    #         ax.set_xlabel(f'recommendation {i - 1}: \n' + "\n".join(
    #                         wrap(f'{items[items["itemID"] == ind]["title"].values} ({ind})', 25)))
    #         if ind not in labels:
    #             img = np.zeros((128, 128, 3))
    #         else:
    #             img = images[labels.index(ind)].transpose((1, 2, 0))
    #         ax.imshow(img)
    #         i += 1
    #     plt.savefig(os.path.join(os.getcwd(), 'recommendations', f'rec_{name}.png'))
    #
    #





    similarity_scores = pd.read_csv('cover_similarity_score.csv')
    grouped_rec = similarity_scores.groupby('itemId')

    for name, group in grouped_rec:

        fig = plt.figure(figsize=[32, 8])
        indexes = group.rec_id.values
        score = group.cover_similarity_score.values
        size = len(indexes)
        ax = fig.add_subplot(1, size + 1, 1)
        if name not in labels:
            img = np.zeros((128, 128, 3))
        else:
            img = images[labels.index(name)].transpose((1, 2, 0))

        ax.imshow(img)
        ax.set_xlabel('original item: \n' + "\n".join(wrap(
            f'{items[items["itemID"] == name]["title"].values} ({name})', 25)))
        i = 2
        for ind in indexes:
            ax = fig.add_subplot(1, size + 1, i)
            ax.set_xlabel(f'recommendation {i - 1}: \n' + "\n".join(
                            wrap(f'{items[items["itemID"] == ind]["title"].values} ({ind})  : {score[i -2]}', 25)))
            if ind not in labels:
                img = np.zeros((128, 128, 3))
            else:
                img = images[labels.index(ind)].transpose((1, 2, 0))
            ax.imshow(img)
            i += 1
        plt.savefig(os.path.join(os.getcwd(), 'similarity_scores', f'rec_{name}.png'))






