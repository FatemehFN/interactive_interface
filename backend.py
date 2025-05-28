import pandas as pd
import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA

def build_knn_graph(expr_matrix, k):
    adata = sc.AnnData(expr_matrix)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=k)
    return adata

def run_leiden(adata, resolution):
    sc.tl.leiden(adata, resolution=resolution)
    return adata.obs['leiden']

def compute_eigengenes(expr_matrix, cluster_labels):
    modules = {}
    labels = sorted(set(cluster_labels))
    for label in labels:
        idx = [i for i, val in enumerate(cluster_labels) if val == label]
        sub_expr = expr_matrix.iloc[:, idx]
        pca = PCA(n_components=1)
        eigengene = pca.fit_transform(sub_expr.T).flatten()
        modules[label] = eigengene
    return pd.DataFrame(modules, index=sub_expr.columns)

def correlate_eigengenes(eigengenes, phenotypes):
    corr = eigengenes.corrwith(phenotypes.loc[eigengenes.index], axis=0, method='pearson')
    return corr
