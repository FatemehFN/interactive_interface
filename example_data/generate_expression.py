# generate_expression.py
import pandas as pd
import numpy as np

np.random.seed(42)
genes = [f"Gene_{i}" for i in range(1, 101)]
samples = [f"Sample_{j}" for j in range(1, 21)]

# Create a random expression matrix
data = np.random.rand(100, 20) * 10
expression_df = pd.DataFrame(data, index=genes, columns=samples)

expression_df.to_csv("expression.csv")
