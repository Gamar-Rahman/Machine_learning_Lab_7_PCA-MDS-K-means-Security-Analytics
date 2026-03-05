import numpy as np
import pandas as pd
import urllib.request
import shutil
from scipy.spatial import distance

np.random.seed(1)


# Problem 1

# Problem 1.2
from sklearn.preprocessing import StandardScaler
def featureNormalization(X):
    """
    Normalize each feature for the input set
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    X = np.asarray(X, dtype=float)

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)

    X_std_safe = X_std.copy()
    X_std_safe[X_std_safe == 0] = 1.0

    X_normalized = (X - X_mean) / X_std_safe
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return X_normalized, X_mean, X_std


# Problem 1.3
def compute_covariance_vanilla(X):
    """
    Compute covariance using vanilla formula
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    X = np.asarray(X, dtype=float)
    m, d = X.shape
    Sigma = np.zeros((d, d))

    for i in range(m):
        xi = X[i].reshape(d, 1)
        Sigma += xi @ xi.T

    Sigma = Sigma / m
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return Sigma


def compute_covariance(X):
    """
    Compute covariance using matrix multiplication
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    X = np.asarray(X, dtype=float)
    m = X.shape[0]
    Sigma = (X.T @ X) / m
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return Sigma


# Problem 1.4
def PCA_SVD(X, k):
    """
    PCA using SVD
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    X = np.asarray(X, dtype=float)

    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    V = Vt.T[:, :k]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return V, X_mean


def PCA_Eigen(X, k):
    """
    PCA using Eigen-decomposition
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    X = np.asarray(X, dtype=float)

    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    Sigma = (X_centered.T @ X_centered) / X_centered.shape[0]

    eigvals, eigvecs = np.linalg.eigh(Sigma)

    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    V = eigvecs[:, :k]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return V, X_mean


# Problem 1.6
def recon_pca(X, k_max):
    """
    Reconstruction error for k = 1,...,k_max
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    X = np.asarray(X, dtype=float)

    X_mean = np.mean(X, axis=0)
    Xc = X - X_mean

    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vfull = Vt.T

    error = np.zeros(k_max)

    for k in range(1, k_max + 1):
        Vk = Vfull[:, :k]
        Xhat = (Xc @ Vk @ Vk.T) + X_mean
        error[k - 1] = np.linalg.norm(X - Xhat, 'fro') ** 2
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return error


# Problem 2
def MDS(dist, k):
    """
    Classical MDS
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dist = np.asarray(dist, dtype=float)
    n = dist.shape[0]

    D2 = dist ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J

    eigvals, eigvecs = np.linalg.eigh(B)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    Lk = np.clip(eigvals[:k], 0, None)
    Vk = eigvecs[:, :k]

    embedding = Vk * np.sqrt(Lk)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return embedding


# Problem 3.1
def kmeans(X, k, maxIter, tol):

    np.random.seed(1)

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    init_idx = np.random.choice(n, k, replace=False)
    M = X[init_idx].copy()

    for _ in range(maxIter):

        dists = np.linalg.norm(X[:, None, :] - M[None, :, :], axis=2)
        cluster = np.argmin(dists, axis=1)

        M_new = M.copy()

        for i in range(k):
            pts = X[cluster == i]
            if len(pts) > 0:
                M_new[i] = np.mean(pts, axis=0)

        shift = np.linalg.norm(M_new - M, axis=1)
        M = M_new

        if np.all(shift < tol):
            break
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return M


# Problem 3.2
def pred_kmeans(centers, Xtest):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    Xtest = np.asarray(Xtest, dtype=float)
    centers = np.asarray(centers, dtype=float)

    dists = np.linalg.norm(Xtest[:, None, :] - centers[None, :, :], axis=2)
    cluster = np.argmin(dists, axis=1)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return cluster


# Problem 3.3
def softkmeans(X, k, beta, maxIter, tol):

    np.random.seed(1)

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    init_idx = np.random.choice(n, k, replace=False)
    M = X[init_idx].copy()

    for _ in range(maxIter):

        # Euclidean distance (NOT squared)
        dists = np.linalg.norm(X[:, None, :] - M[None, :, :], axis=2)

        W = np.exp(-beta * dists)
        W_sum = np.sum(W, axis=1, keepdims=True)
        W_sum[W_sum == 0] = 1.0
        B = W / W_sum

        M_new = np.zeros_like(M)

        for i in range(k):
            weights = B[:, i].reshape(-1, 1)
            denom = np.sum(weights)

            if denom > 0:
                M_new[i] = np.sum(weights * X, axis=0) / denom
            else:
                M_new[i] = M[i]

        shift = np.linalg.norm(M_new - M, axis=1)
        M = M_new

        if np.all(shift < tol):
            break
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return M


# Problem 3.4
def pred_softkmeans(centers, Xtest, beta):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    Xtest = np.asarray(Xtest, dtype=float)
    centers = np.asarray(centers, dtype=float)

    dists = np.linalg.norm(Xtest[:, None, :] - centers[None, :, :], axis=2)

    W = np.exp(-beta * dists)
    W_sum = np.sum(W, axis=1, keepdims=True)
    W_sum[W_sum == 0] = 1.0
    B = W / W_sum
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return B