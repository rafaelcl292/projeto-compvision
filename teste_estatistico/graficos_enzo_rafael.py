import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Read data from files
with open('transformation_results_enzo_estrela.txt', 'r') as f:
    enzo_data = [float(line.strip()) for line in f if line.strip()]

with open('transformation_results_rafael_estrela.txt', 'r') as f:
    rafael_data = [float(line.strip()) for line in f if line.strip()]

# Perform t-test
t_stat, p_value = stats.ttest_ind(enzo_data, rafael_data)

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Plot 1: Box plot
data_to_plot = [enzo_data, rafael_data]
ax1.boxplot(data_to_plot, labels=['Enzo', 'Rafael'])
ax1.set_title('Distribuição dos Resultados: Enzo vs Rafael', fontsize=14, pad=20)
ax1.set_ylabel('Valores', fontsize=12)
ax1.grid(True, alpha=0.3)

# Plot 2: Histogram with KDE
sns.histplot(data=enzo_data, bins=30, alpha=0.5, label='Enzo', ax=ax2)
sns.histplot(data=rafael_data, bins=30, alpha=0.5, label='Rafael', ax=ax2)
ax2.set_title('Histograma das Distribuições', fontsize=14, pad=20)
ax2.set_xlabel('Valores', fontsize=12)
ax2.set_ylabel('Frequência', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add t-test results as text
plt.figtext(0.5, 0.01, 
            f'T-statistic: {t_stat:.6f}\nP-valor: {p_value*100:.6f}%\n' + 
            f'Significância: {"Significativo" if p_value < 0.05 else "Não significativo"}', 
            ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('comparacao_enzo_rafael.png', dpi=300, bbox_inches='tight')
plt.close()

# Print detailed statistics
print("\nEstatísticas Detalhadas:")
print("=" * 50)
print(f"\nEnzo:")
print(f"Média: {np.mean(enzo_data):.6f}")
print(f"Desvio Padrão: {np.std(enzo_data):.6f}")
print(f"Mediana: {np.median(enzo_data):.6f}")
print(f"Min: {np.min(enzo_data):.6f}")
print(f"Max: {np.max(enzo_data):.6f}")

print(f"\nRafael:")
print(f"Média: {np.mean(rafael_data):.6f}")
print(f"Desvio Padrão: {np.std(rafael_data):.6f}")
print(f"Mediana: {np.median(rafael_data):.6f}")
print(f"Min: {np.min(rafael_data):.6f}")
print(f"Max: {np.max(rafael_data):.6f}")

print(f"\nTeste T:")
print(f"T-statistic: {t_stat:.6f}")
print(f"P-valor: {p_value*100:.6f}%")
print(f"Significância: {'Significativo' if p_value < 0.05 else 'Não significativo'}")