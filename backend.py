import pandas as pd
import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA

def build_knn_graph(expr_matrix, k):
    adata = sc.AnnData(expr_matrix)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=k)
    
    # Compute UMAP and t-SNE for visualization
    sc.tl.umap(adata)
    sc.tl.tsne(adata, n_pcs=20)  # You can adjust this
    return adata

def run_leiden(adata, resolution):
    sc.tl.leiden(adata, resolution=resolution)
    return adata.obs['leiden']


def compute_eigengenes(expr_matrix, cluster_labels):
    """
    Compute module eigengenes using the first principal component of each module,
    and ensure consistent sign by aligning eigengene with first gene in the module.

    Parameters:
        expr_matrix (pd.DataFrame): genes × samples
        cluster_labels (list or pd.Series): cluster labels for each gene (length = number of genes)

    Returns:
        pd.DataFrame: samples × modules (Module_0, Module_1, ...)
    """
    if len(cluster_labels) != expr_matrix.shape[0]:
        raise ValueError("Length of cluster_labels must match number of genes (rows in expr_matrix).")

    modules = {}

    try:
        unique_labels = sorted(set(cluster_labels), key=lambda x: int(x))
    except ValueError:
        unique_labels = sorted(set(cluster_labels))

    for label in unique_labels:
        idx = [i for i, val in enumerate(cluster_labels) if val == label]
        sub_expr = expr_matrix.iloc[idx, :]  # genes × samples

        pca = PCA(n_components=1)
        eigengene = pca.fit_transform(sub_expr.T).flatten()  # 1 value per sample

        # Align sign with the first gene in the module
        first_gene_expr = sub_expr.iloc[0, :].values
        sign = np.sign(np.corrcoef(eigengene, first_gene_expr)[0, 1])
        eigengene *= sign

        modules[f"Module_{label}"] = eigengene

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

    # Ensure both indices are strings for proper alignment
    eigengenes.index = eigengenes.index.astype(str)
    phenotypes.index = phenotypes.index.astype(str)

    # Align samples (rows)
    common_samples = eigengenes.index.intersection(phenotypes.index)
    print(f"Found {len(common_samples)} common samples.")

    eigengenes = eigengenes.loc[common_samples]
    phenotypes = phenotypes.loc[common_samples]

    # Convert to numeric in case of non-numeric columns
    eigengenes = eigengenes.apply(pd.to_numeric, errors='coerce')
    phenotypes = phenotypes.apply(pd.to_numeric, errors='coerce')

    # Drop samples with NaNs in either dataframe
    aligned = eigengenes.join(phenotypes, how='inner')
    eigengenes = aligned[eigengenes.columns]
    phenotypes = aligned[phenotypes.columns]

    # Initialize correlation DataFrame
    corr_df = pd.DataFrame(index=eigengenes.columns, columns=phenotypes.columns, dtype=float)

    # Compute correlations
    for m in eigengenes.columns:
        for ph in phenotypes.columns:
            corr_value = eigengenes[m].corr(phenotypes[ph], method='pearson')
            corr_df.loc[m, ph] = corr_value

    return corr_df



# def analyze_best_k(expr_matrix, k_values, resolution=1.0):
#     results = []
#     for k in k_values:
#         adata = build_knn_graph(expr_matrix, k)
#         sc.tl.leiden(adata, resolution=resolution)
#         clusters = adata.obs['leiden']
        
#         n_clusters = clusters.nunique()
#         # Optional: compute modularity (scanpy has it in adata.uns['modularity'] if computed)
#         # or silhouette score - but silhouette needs embedding or distance matrix
        
#         results.append({'k': k, 'n_clusters': n_clusters})
#     return pd.DataFrame(results)