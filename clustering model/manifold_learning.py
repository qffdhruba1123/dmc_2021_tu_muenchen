import numpy as np
import os
from collections import OrderedDict
from functools import partial
from time import time
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from preprocessing import *
import umap

# Next line to silence pyflakes. This import is needed.
Axes3D

# sparse_data = scipy.sparse.load_npz('OHE_sparse_matrix.npz')
#
# svd = TruncatedSVD(n_components= 1200, n_iter= 10, random_state=42, algorithm= 'randomized')
# svd.fit(sparse_data)
# dense_data = sparse_data
# print(svd.explained_variance_ratio_.sum())
# X = svd.fit_transform(sparse_data)

# fig = plot_2dhist(['80000038'], 'matsum', 'internal_energy', curves_ndarray, stats_plot = True, save = False)
#
# n_neighbors = 10
# n_components = 3
#
# # Create figure
# fig = plt.figure(figsize=(15, 8))
# fig.suptitle("Manifold Learning", fontsize=14)
#
# # Add 3d scatter plot
# # ax = fig.add_subplot(251, projection='3d')
# # ax.scatter(t, X, cmap=plt.cm.Spectral)
# # ax.view_init(4, -72)
#
# # Set-up manifold methods
#
# # LLE = partial(manifold.LocallyLinearEmbedding,
# #               n_neighbors, n_components, eigen_solver='auto')
# methods = OrderedDict()
# # methods['LLE'] = LLE(method='standard')
# # methods['LTSA'] = LLE(method='ltsa')
# # methods['Hessian LLE'] = LLE(method='hessian')
# # methods['Modified LLE'] = LLE(method='modified')
# # methods['Isomap'] = manifold.Isomap(n_neighbors, n_components)
# # methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)
# # methods['SE'] = manifold.SpectralEmbedding(n_components=n_components,
# #                                            n_neighbors=n_neighbors)
# methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='random',
#                                  random_state=0)
# print('start')
# # Plot results
# for i, (label, method) in enumerate(methods.items()):
#     t0 = time()
#     Y = method.fit_transform(X)
#     t1 = time()
#     print("%s: %.2g sec" % (label, t1 - t0))
#     # ax = fig.add_subplot(2, 5, 2 + i + (i > 3))
#     ax = fig.add_subplot( projection='3d')
#
#     ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s = 1, cmap=plt.cm.Spectral)
#     ax.set_title("%s (%.2g sec)" % (label, t1 - t0))
#     ax.xaxis.set_major_formatter(NullFormatter())
#     ax.yaxis.set_major_formatter(NullFormatter())
#     ax.axis('tight')
#
#

def learn_manifold(x_data, umap_min_dist = 0.00, umap_metric = 'euclidean', umap_dim = 10, umap_neighbors = 30):
    md = float(umap_min_dist)
    umap_learning = umap.UMAP(random_state=0, metric=umap_metric, n_components=umap_dim, n_neighbors=umap_neighbors,
              min_dist=md)
    umap_learning.fit_transform(x_data)
    return umap


X = np.load('svd_dataset_2500.npy')

neigh = NearestNeighbors(n_neighbors=15, radius=0.4)
neigh.fit(X)
distance, neighbors_index = neigh.kneighbors([X[22]], 100, return_distance=True)

# index = [False] * X.shape[0]
# not_index = [True] * X.shape[0]
color = ['b'] * X.shape[0]
size = [1] * X.shape[0]

for i in neighbors_index[0]:
    # index[i] = True
    # not_index[i] = False
    color[i] = 'r'
    size[i] = 3

fig = plt.figure(figsize=(15, 8))
fig.suptitle("Manifold Learning", fontsize=14)
ax = fig.add_subplot( projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s = size, c=color, alpha = 0.3)

# ax.scatter(X[index, 0], X[index, 1], X[index, 2], s = 3, cmap=plt.cm.Spectral)
# ax.scatter(X[not_index, 0], X[not_index, 1], X[not_index, 2], s = 1, cmap=plt.cm.Spectral)
items,_ = preprocessing()
rec = items.loc[neighbors_index[0]]
# print(items[neighbors_index])