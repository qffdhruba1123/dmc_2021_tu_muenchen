import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, csv
import scipy
from scipy.sparse import hstack
from cover_image import *
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from preprocessing import *

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import minmax_scale
# sparse_data_t = scipy.sparse.load_npz('tokens_sparse_OHE_matrix.npz')
# sparse_data_a = scipy.sparse.load_npz('authors_sparse_OHE_matrix.npz')
# sparse_data_m = scipy.sparse.load_npz('main_topic_sparse_OHE_matrix.npz')
# sparse_data_s = scipy.sparse.load_npz('subtopics_sparse_OHE_matrix.npz')
# sparse_data_p = scipy.sparse.load_npz('publisher_sparse_OHE_matrix.npz')
# sparse_tapms_3000 = hstack((sparse_data_t, sparse_data_a, sparse_data_m, sparse_data_s, sparse_data_p))
# np.save('sparse_tapms_3000', sparse_tapms_3000)

# svd = TruncatedSVD(n_components= 3000, n_iter= 10, random_state=42, algorithm= 'randomized')
# data = svd.fit_transform(sparse_data_p)
# print(svd.explained_variance_ratio_.sum())
# np.save('publishers_svd_3000.npy', data)

# svd_p = np.load('publishers_svd_800.npy')
# svd_m = np.load('main_topics_svd_100.npy')
# svd_s = np.load('subtopics_svd_100.npy')
# svd_a = np.load('authors_svd_4000.npy')
# svd_t = np.load('tokens_svd_3000.npy')
# svd_tapms = np.concatenate((svd_t, svd_a, svd_p, svd_m, svd_s))

if True:

    # X = np.load('sparse_tapms_3000.npy', allow_pickle=True).item()
    # svd = TruncatedSVD(n_components= 3000, n_iter= 10, random_state=42, algorithm= 'randomized')
    # tapms_svd_3000 = svd.fit_transform(X)
    # print(svd.explained_variance_ratio_.sum())
    # np.save('tapms_svd_3000.npy', tapms_svd_3000)
    X = np.load('tapms_svd_3000.npy', allow_pickle=True)

    items, _, eval = preprocessing()
    loc = []

    # X = data_svd_3500
    rec = {}
    neigh = NearestNeighbors(n_neighbors=20, radius=1)
    neigh.fit(X)

    items.set_index('itemID')
    # itemID_dict = {for i in range(items.itemID)}
    final_string_list = []
    distances = []
    for e in eval.itemID:
        # _r = str(e) + ":"
        row_ = items.query('itemID == ' + str(e)).index[0]
        distance, neighbors_index = neigh.kneighbors([X[row_]], 20, return_distance=True)
        indexes = neighbors_index[0][:10]
        distances.append(distance[0][:10])
        if np.max(distance) > 5:
            print(f'to large distance {e}: {distance}')
        # if np.prod(indexes.shape) == 0:
        #     print(e)
        indexes = [i for i in indexes if row_ != i][:5]
        indexes.insert(0, row_)

        rec[e] = items.loc[indexes]
        item_list = items.loc[indexes].itemID.values.tolist()[1:]
        str_list = [str(i) for i in item_list]
        _r = '|'.join(str_list)

        final_string_list.append(_r)
        # print(f'{e}:{_r}')
        rec[e] = items.loc[indexes]

        loc.append(_r)
    recommendations = pd.DataFrame({'itemID' : eval.itemID, 'recommendations' : final_string_list}).set_index('itemID')
    recommendations.to_csv('recommendations_3.csv')
    np.save('rec_3', rec)
    print(loc)
from sklearn.preprocessing import normalize
#
# dim = 128
# images = np.load(f'data_{dim}.npy')
# labels = np.load('labels.npy')
# labels = [int(l) for l in labels]
#
# dataset = TensorDataset(torch.Tensor(images), torch.Tensor(labels))
# device = 'cuda'
# n_latent = 16
# vae = VAE(n_latent= n_latent, init_dim = dim)
#
# vae = torch.load(f'conv_vae_{n_latent}_1')
# # vae.load_state_dict(checkpoint['state_dict'])
# # train(vae, dataloaders['train'], dataloaders['val'], 1e-3, 100, device)
#
# latent_variables = np.empty((len(dataset), n_latent))
# vae.eval()
# for i, data in enumerate(dataset):
#     latent_variables[i] = vae(data[0].unsqueeze(0).cuda())[-1].squeeze(0).detach().cpu().numpy()
#
# min_max_scaler = preprocessing.MinMaxScaler()
# normed_matrix = min_max_scaler.fit_transform(latent_variables)
# np.save('latent_variables', latent_variables)
#
# neigh = NearestNeighbors(n_neighbors=20, radius=1)
# neigh.fit(normed_matrix)
#
# import random
# indexes = random.sample(range(0, images.shape[0]), 5)
#
#
# for j, r in enumerate(indexes):
#     fig = plt.figure(figsize=[80, 10])
#
#     distance, neighbors_index = neigh.kneighbors([normed_matrix[r]], 100, return_distance=True)
#
#     for i, ind in enumerate(neighbors_index[0][:10]):
#         img = images[ind].transpose((1, 2, 0))
#         if i == 0:
#             img = img/255
#         ax = fig.add_subplot(1, 10, i + 1)
#         ax.imshow(img)
#     plt.show()
#
# items, _, eval = preprocessing()
# loc = []
# X = np.load('latent_variables.npy')
# min_max_scaler = MinMaxScaler()
# X = min_max_scaler.fit_transform(X)
# # X = data_svd_3500
# neigh = NearestNeighbors(n_neighbors=20, radius=1)
# neigh.fit(X)
# final_recommendation = pd.read_csv('finalrecs_20210627 - finalrecs_20210627.csv')
# grouped_rec = final_recommendation.groupby('itemId')
# labels = np.load('labels.npy')
#
# distance = {}
#
# for name, group in grouped_rec:
#         distance[name] = {}
#
#
#         _x = X[np.where(labels == str(name))]
#         # row_ = items.query('itemID == ' + str(name)).index[0]
#         # distance, neighbors_index = neigh.kneighbors([X[row_]], 1000, return_distance=True)
#         # ids = labels[neighbors_index]
#
#
#
#     #     fig = plt.figure(figsize=[32, 8])
#         indexes = group.rec_id.values
#         if str(name) not in labels:
#             for ind in indexes:
#                 distance[name].update({ind: 100})
#             continue
#
#         for ind in indexes:
#             if str(ind) not in labels:
#                 distance[name].update({ind:100})
#                 continue
#             _y = X[np.where(labels == str(ind))]
#             d = np.linalg.norm(_x - _y)
#             distance[name].update({ind: d})
            #     size = len(indexes)
    #     ax = fig.add_subplot(1, size + 1, 1)
    #     img = images[labels.index(name)].transpose((1, 2, 0))
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
# cover_similarity_score = []
# cover_similarity_score_2 = []
#
# scores = []
# for key, subdict in distance.items():
#     distance[key] = {k: v for k, v in sorted(subdict.items(), key=lambda item: item[1])}
#     for subkey, value in distance[key].items():
#         if value > 5:
#             cover_similarity_score.append(', '.join([str(key), str(subkey), str(0)]))
#             cover_similarity_score_2.append(', '.join([str(key), str(subkey), str(0)]))
#
#             scores.append(0)
#         else:
#             cover_similarity_score.append(', '.join([str(key), str(subkey), str(1 - value / 5)]))
#             cover_similarity_score_2.append(', '.join([str(key), str(subkey), str((1 - value / 5) ** 2)]))
#
#             scores.append(1 - value / 5)
#
# import csv
#
# cover_similarity_score = [s.rsplit(', ') for s in cover_similarity_score]
# with open('cover_similarity_score', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#
#     write.writerow(['itemId', 'rec_id', 'cover_similarity_score'])
#     write.writerows(cover_similarity_score)
# cover_similarity_score_2 = [s.rsplit(', ') for s in cover_similarity_score_2]
# with open('cover_similarity_score_squared', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#
#     write.writerow(['itemId', 'rec_id', 'cover_similarity_score_squared'])
#     write.writerows(cover_similarity_score_2)
