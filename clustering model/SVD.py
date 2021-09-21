import sklearn.preprocessing
import pandas as pd
import numpy as np
import scipy
from sklearn.decomposition import TruncatedSVD





# load data
sparse_data = scipy.sparse.load_npz('OHE_sparse_matrix.npz')

svd = TruncatedSVD(n_components= 500, n_iter= 10, random_state=42, algorithm= 'randomized')
svd.fit(sparse_data)

print(svd.explained_variance_ratio_.sum())
new_data = svd.fit_transform(sparse_data)

reduced_data = learn_manifold(new_data, umap_dim=250)