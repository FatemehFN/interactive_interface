import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import scanpy as sc
# Optional: Tweak font size to avoid warnings from scanpy
matplotlib.rcParams.update({'font.size': 10})
from backend import (
    build_knn_graph,
    run_leiden,
    compute_eigengenes,
    correlate_eigengenes
)

st.set_page_config(page_title="Network Module Analyzer", layout="wide")
st.title("🔬 Network Module Analyzer")

st.markdown("""
Upload your expression matrix (genes × samples) and phenotype data (samples × traits),  
or use our example dataset to explore the analysis pipeline.
""")

# --- Upload Option ---
use_own_data = st.radio(
    "📂 Do you want to upload your own data?",
    ("No, use example data", "Yes, I want to upload my own"),
    index=0
)

# --- Load Data ---
expr_df, pheno_df = None, None

if use_own_data == "Yes, I want to upload my own":
    expr_file = st.file_uploader("🧬 Upload Expression Data (CSV)", type="csv")
    pheno_file = st.file_uploader("📊 Upload Phenotype Data (CSV)", type="csv")

    if expr_file and pheno_file:
        expr_df = pd.read_csv(expr_file, index_col=0)
        pheno_df = pd.read_csv(pheno_file, index_col=0)
else:
    try:
        expr_df = pd.read_csv("example_data/expression.csv", index_col=0)
        pheno_df = pd.read_csv("example_data/phenotype.csv", index_col=0)
        st.success("✅ Example data loaded.")
    except FileNotFoundError:
        st.error("❌ Example data not found. Please make sure `example_data/expression.csv` and `example_data/phenotype.csv` exist.")
        st.stop()

# --- Parameter Selection ---
k = st.slider("🔧 Select `k` for KNN graph", min_value=2, max_value=50, value=10)
resolution = st.slider("🌀 Select resolution for Leiden clustering", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

if expr_df is not None and pheno_df is not None:
    st.write("Expression Data:", expr_df.shape)
    st.write("Phenotype Data:", pheno_df.shape)

    if st.button("🚀 Run Analysis"):
        with st.spinner("Processing..."):
            try:
                # Step 1: Build graph & cluster
                adata = build_knn_graph(expr_df, k)
                leiden_clusters = run_leiden(adata, resolution)

                st.subheader("📂 Leiden Clusters")
                st.dataframe(leiden_clusters.value_counts().sort_index())


                # --- UMAP + t-SNE side-by-side ---
                st.subheader("🧭 UMAP and 🌀 t-SNE of Samples")

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                # UMAP
                sc.pl.umap(adata, color='leiden', ax=axes[0], show=False, title='UMAP')

                # t-SNE
                sc.pl.tsne(adata, color='leiden', ax=axes[1], show=False, title='t-SNE')

                plt.tight_layout()
                st.pyplot(fig)

                # Step 2: Eigengenes
                st.subheader("🧬 Module Eigengenes")
                st.write("Expression matrix shape:", expr_df.shape)
                eigengenes_df = compute_eigengenes(expr_df, leiden_clusters.tolist())
                st.dataframe(eigengenes_df)

                # Step 3: Correlation
                st.subheader("📈 Correlation with Phenotypes")
                corr = correlate_eigengenes(eigengenes_df, pheno_df)
                st.dataframe(corr)

                corr=corr.astype(float)
                st.subheader("📊 Heatmap of Correlations (Phenotypes x Modules)")
                # Transpose if needed: rows = phenotypes, columns = modules
                fig, ax = plt.subplots(figsize=(10, 2))
                sns.heatmap(corr.T.values, annot=True, fmt=".2f", cmap="vlag", cbar_kws={"label": "Pearson r"}, ax=ax)
                plt.xlabel("Modules")
                plt.ylabel("Phenotypes")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {e}")



# if expr_df is not None:

#     if st.button("Analyze k"):
#         with st.spinner("Running k analysis..."):
#             k_range = list(range(2, 10))
#             best_k_df = analyze_best_k(expr_df, k_range, resolution)
#             st.dataframe(best_k_df)
#             fig2, ax2 = plt.subplots(figsize=(10, 5))
#             ax2.plot(best_k_df['k'], best_k_df['n_clusters'], marker='o')
#             ax2.set_xlabel('k (neighbors)')
#             ax2.set_ylabel('Number of Leiden clusters')
#             ax2.set_title('Number of clusters vs k (res=1.0)')
#             st.pyplot(fig2)