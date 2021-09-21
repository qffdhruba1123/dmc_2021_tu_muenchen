import scipy
import matplotlib.pyplot as plt
from sklearn.decomposition import *
sparse_data = scipy.sparse.load_npz('OHE_sparse_matrix.npz')

X = sparse_data
# kpca = KernelPCA(kernel="rbf", n_components= 1200, fit_inverse_transform=True, eigen_solver = 'arpack', n_jobs = -1) #arpack
# X_kpca = kpca.fit_transform(X)
# X_back = kpca.inverse_transform(X_kpca)

# pca = PCA(n_components= 1200, svd_solver = 'arpack')
# X_pca = pca.fit(X.toarray())


svd = TruncatedSVD(n_components= 2500, n_iter= 10, random_state=42, algorithm= 'randomized')
data_svd = svd.fit(sparse_data)
