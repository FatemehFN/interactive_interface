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

    if len(cluster_labels) != expr_matrix.shape[0]:
        raise ValueError("Length of cluster_labels must match number of genes (rows in expr_matrix).")

    modules = {}
    labels = sorted(set(cluster_labels))
    
    for label in labels:
        # Get gene indices in this module
        idx = [i for i, val in enumerate(cluster_labels) if val == label]
        
        # Subset expression: genes (rows) in module × all samples (columns)
        sub_expr = expr_matrix.iloc[idx, :]
        
        # PCA on genes × samples → need to transpose to samples × genes
        pca = PCA(n_components=1)
        eigengene = pca.fit_transform(sub_expr.T).flatten()  # 1 value per sample

        modules[f"Module_{label}"] = eigengene

    # Return: samples × modules
    return pd.DataFrame(modules, index=expr_matrix.columns)


def correlate_eigengenes(eigengenes, phenotypes):
    """
    Compute Pearson correlations between module eigengenes and phenotype traits.

    Parameters:
        eigengenes (pd.DataFrame): samples × modules
        phenotypes (pd.DataFrame): samples × traits

    Returns:
        pd.DataFrame: modules × traits correlation matrix
    """
    # Align samples (index) in both dataframes
    common_samples = eigengenes.index.intersection(phenotypes.index)
    eigengenes = eigengenes.loc[common_samples]
    phenotypes = phenotypes.loc[common_samples]
    corr_df= pd.DataFrame(index=eigengenes.columns, columns=phenotypes.columns)
    for m in eigengenes.columns:
        for ph in phenotypes.columns:
            corr_value=eigengenes[m].corr(phenotypes[ph], method='pearson')
            corr_df.loc[m, ph] = corr_value

    return corr_df



def analyze_best_k(expr_matrix, k_values, resolution=1.0):
    results = []
    for k in k_values:
        adata = build_knn_graph(expr_matrix, k)
        sc.tl.leiden(adata, resolution=resolution)
        clusters = adata.obs['leiden']
        
        n_clusters = clusters.nunique()
        # Optional: compute modularity (scanpy has it in adata.uns['modularity'] if computed)
        # or silhouette score - but silhouette needs embedding or distance matrix
        
        results.append({'k': k, 'n_clusters': n_clusters})
    return pd.DataFrame(results)