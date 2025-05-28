import streamlit as st
import pandas as pd
from backend import (
    build_knn_graph,
    run_leiden,
    compute_eigengenes,
    correlate_eigengenes
)

st.set_page_config(page_title="Network Module Analyzer", layout="wide")
st.title("🔬 Network Module Analyzer")

st.markdown("""
Upload your expression matrix (genes × samples) and phenotype data (samples × traits).  
Adjust parameters for KNN graph and Leiden clustering, and explore module eigengenes and their correlation with phenotypes.
""")

# --- File Uploads ---
expr_file = st.file_uploader("🧬 Upload Expression Data (CSV)", type="csv")
pheno_file = st.file_uploader("📊 Upload Phenotype Data (CSV)", type="csv")

# --- Parameter Selection ---
k = st.slider("🔧 Select `k` for KNN graph", min_value=2, max_value=50, value=10)
resolution = st.slider("🌀 Select resolution for Leiden clustering", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

if expr_file and pheno_file:
    try:
        expr_df = pd.read_csv(expr_file, index_col=0)
        pheno_df = pd.read_csv(pheno_file, index_col=0)

        st.success("✅ Data uploaded successfully!")
        st.write("Expression Data:", expr_df.shape)
        st.write("Phenotype Data:", pheno_df.shape)

        # Button to trigger analysis
        if st.button("🚀 Run Analysis"):
            with st.spinner("Processing..."):
                # Transpose expr_df: rows = cells/samples
                adata = build_knn_graph(expr_df.T, k)
                leiden_clusters = run_leiden(adata, resolution)

                # Display cluster assignments
                st.subheader("📂 Leiden Clusters")
                st.dataframe(leiden_clusters.value_counts().sort_index())

                # Compute eigengenes
                st.subheader("🧬 Module Eigengenes")
                eigengenes_df = compute_eigengenes(expr_df, leiden_clusters.tolist())
                st.dataframe(eigengenes_df)

                # Compute correlations
                st.subheader("📈 Correlation with Phenotypes")
                corr = correlate_eigengenes(eigengenes_df, pheno_df)
                st.dataframe(corr)

                # Plot
                st.subheader("📊 Bar Plot of Correlations")
                st.bar_chart(corr)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
else:
    st.info("📁 Please upload both expression and phenotype CSV files to continue.")
