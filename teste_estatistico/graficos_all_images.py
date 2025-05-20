import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import itertools
import pandas as pd

# List of all transformation result files
files = [
    'transformation_results_enzo_estrela.txt',
    'transformation_results_marcelo_estrela.txt',
    'transformation_results_rafael_estrela.txt',
    'transformation_results_bruno_estrela.txt'
]

# Dictionary to store data from each file
data_dict = {}

# Read data from each file
for file in files:
    with open(file, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip()]
        name = file.replace('transformation_results_', '').replace('_estrela.txt', '')
        data_dict[name] = data

# Create a DataFrame for t-test results
combinations = list(itertools.combinations(data_dict.keys(), 2))
results = []

for (name1, name2) in combinations:
    t_stat, p_value = stats.ttest_ind(data_dict[name1], data_dict[name2])
    results.append({
        'Comparison': f'{name1} vs {name2}',
        't-statistic': t_stat,
        'p-value': p_value,
        'p-value (%)': p_value * 100
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Plot 1: Box plot of all distributions
data_to_plot = [data_dict[name] for name in data_dict.keys()]
ax1.boxplot(data_to_plot, labels=list(data_dict.keys()))
ax1.set_title('Distribuição dos Resultados por Pessoa', fontsize=14, pad=20)
ax1.set_ylabel('Valores', fontsize=12)
ax1.grid(True, alpha=0.3)

# Plot 2: Bar plot of p-values
sns.barplot(data=results_df, x='Comparison', y='p-value (%)', ax=ax2)
ax2.set_title('P-valores dos Testes T (%)', fontsize=14, pad=20)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.axhline(y=5, color='r', linestyle='--', label='5%')
ax2.set_ylabel('P-valor (%)', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('comparacao_resultados.png', dpi=300, bbox_inches='tight')
plt.close()

# Print t-test results with formatted output
print("\nResultados dos Testes T:")
print("=" * 50)
for _, row in results_df.iterrows():
    print(f"\nComparação: {row['Comparison']}")
    print(f"T-statistic: {row['t-statistic']:.6f}")
    print(f"P-valor: {row['p-value (%)']:.6f}%")
    print(f"Significância: {'Significativo' if row['p-value'] < 0.05 else 'Não significativo'}")
    print("-" * 50)
