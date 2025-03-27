import pandas as pd
import matplotlib.pyplot as plt
from simulate import OUTPUT_DIR

if __name__ == "__main__":
    configs = pd.read_csv(OUTPUT_DIR / "best_configs.csv")
    
    if "Unnamed: 0" in configs.columns:
        configs = configs.drop(columns=["Unnamed: 0"])
    
    variables = configs.columns.tolist()
    n = len(variables)
    fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))
    
    for i, var_y in enumerate(variables):
        for j, var_x in enumerate(variables):
            ax = axes[i, j]
            # On the diagonal, plot a histogram
            if i == j:
                ax.hist(configs[var_x], bins=20, color='gray', edgecolor='black')
                ax.set_title(f"{var_x} histogram", fontsize=10)
            else:
                # Off-diagonal: scatter plot of var_x vs var_y
                ax.scatter(configs[var_x], configs[var_y], s=10, alpha=0.7)
                ax.set_title(f"{var_y} vs {var_x}", fontsize=10)
            
            # Label the edge plots for clarity
            if i == n - 1:
                ax.set_xlabel(var_x, fontsize=8)
            if j == 0:
                ax.set_ylabel(var_y, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pairwise_plots.png")
