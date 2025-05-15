import matplotlib.pyplot as plt
import seaborn as sns

# Read the data from the file
with open('transformation_results.txt', 'r') as file:
    data = [float(line.strip()) for line in file if line.strip()]

# Read the data from the file
with open('transformation_results2.txt', 'r') as file:
    data2 = [float(line.strip()) for line in file if line.strip()]

# Create figure with specific size
plt.figure(figsize=(12, 6))

# Create the distribution plot
sns.histplot(data=data, bins=20)
sns.histplot(data=data2, bins=20)

# Set x-axis limits from 0 to 1
plt.xlim(0, 1)

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
