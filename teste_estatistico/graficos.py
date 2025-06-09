import matplotlib.pyplot as plt
import seaborn as sns

# Read the data from the files
with open('/Users/marcelomarchetto/Desktop/viscomproj/teste_estatistico/transformation_results/linus/transformation_results_linus.txt', 'r') as file:
    data = [float(line.strip()) for line in file if line.strip()]

with open('/Users/marcelomarchetto/Desktop/viscomproj/teste_estatistico/transformation_results/transformation_results_enzo_raposa.txt', 'r') as file:
    data2 = [float(line.strip()) for line in file if line.strip()]

# Create figure with specific size
plt.figure(figsize=(12, 6))

# Create the distribution plots with distinct labels and slight transparency for overlap visibility
sns.histplot(data=data, bins=100, label='Marcelo', alpha=0.5)

# Set x-axis limits
plt.xlim(-1, 1)

# Add legend
plt.legend(title='Datasets')

# Customize the plot
plt.title('Distribuição dos Resultados Estatísticos', fontsize=14, pad=20)
plt.xlabel('Valores', fontsize=12)
plt.ylabel('Frequência', fontsize=12)

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('distribuicao_resultados.png', dpi=300, bbox_inches='tight')
plt.close()