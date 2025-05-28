# generate_phenotype.py
import pandas as pd
import numpy as np

np.random.seed(123)
samples = [f"Sample_{j}" for j in range(1, 21)]
phenotypes = {
    "Phenotype_1": np.random.randn(20) * 2 + 5,   # normal distribution
    "Phenotype_2": np.random.randint(0, 2, 20)    # binary trait (0 or 1)
}
phenotype_df = pd.DataFrame(phenotypes, index=samples)

phenotype_df.to_csv("phenotype.csv")
