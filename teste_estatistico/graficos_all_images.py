import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import itertools
import pandas as pd
import numpy as np
import os

# Define the results directory
RESULTS_DIR = './transformation_results'

# List of all transformation result files for both star and mack
RESULTS_DIR = os.path.join('transformation_results')

# Define subdirectories for each dataset
STAR_DIR = os.path.join(RESULTS_DIR, 'estrela')
MACK_DIR = os.path.join(RESULTS_DIR, 'mack')
RAPOSA_DIR = os.path.join(RESULTS_DIR, 'raposa')

# List of players
PLAYERS = ['enzo', 'marcelo', 'rafael', 'bruno']

# Generate file paths for each dataset
star_files = [os.path.join(STAR_DIR, f'transformation_results_{player}_estrela.txt') for player in PLAYERS]
mack_files = [os.path.join(MACK_DIR, f'transformation_results_{player}_mack.txt') for player in PLAYERS]

def print_statistics(data_dict, title_prefix):
    """Print detailed statistics for each player's data"""
    print(f"\n{'='*80}")
    print(f"ESTATÍSTICAS DETALHADAS - {title_prefix}")
    print(f"{'='*80}")
    
    for name, data in data_dict.items():
        print(f"\n{name.upper()}:")
        print(f"{'-'*40}")
        print(f"Média: {np.mean(data):.4f}")
        print(f"Mediana: {np.median(data):.4f}")
        print(f"Desvio Padrão: {np.std(data):.4f}")
        print(f"Valor Mínimo: {np.min(data):.4f}")
        print(f"Valor Máximo: {np.max(data):.4f}")
        print(f"Quantidade de amostras: {len(data)}")

def analyze_data(files, title_prefix):
    # Dictionary to store data from each file
    data_dict = {}

    # Read data from each file
    for file in files:
        try:
            with open(file, 'r') as f:
                data = [float(line.strip()) for line in f if line.strip()]
                name = os.path.basename(file).replace('transformation_results_', '').replace('_estrela.txt', '').replace('_mack.txt', '').replace('_raposa.txt', '')
                data_dict[name] = data
        except FileNotFoundError:
            print(f"ERRO: Arquivo não encontrado: {file}")
            continue
        except Exception as e:
            print(f"ERRO ao processar arquivo {file}: {str(e)}")
            continue

    if not data_dict:
        print(f"ERRO: Nenhum dado válido encontrado para {title_prefix}")
        return

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
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

    # Plot 1: Box plot of all distributions
    data_to_plot = [data_dict[name] for name in data_dict.keys()]
    ax1.boxplot(data_to_plot, tick_labels=list(data_dict.keys()))
    ax1.set_title(f'Distribuição dos Resultados por Pessoa - {title_prefix}', fontsize=14, pad=20)
    ax1.set_ylabel('Valores', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bar plot of p-values
    sns.barplot(data=results_df, x='Comparison', y='p-value (%)', ax=ax2)
    ax2.set_title(f'P-valores dos Testes T (%) - {title_prefix}', fontsize=14, pad=20)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.axhline(y=5, color='r', linestyle='--', label='5%')
    ax2.set_ylabel('P-valor (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Distribution plot
    for name, data in data_dict.items():
        sns.histplot(data=data, bins=50, label=name, alpha=0.5, ax=ax3)
    
    ax3.set_title(f'Distribuição dos Resultados Estatísticos - {title_prefix}', fontsize=14, pad=20)
    ax3.set_xlabel('Valores', fontsize=12)
    ax3.set_ylabel('Frequência', fontsize=12)
    ax3.legend(title='Datasets')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-1, 1)

    # Adjust layout
    plt.tight_layout()

    # Create output directory if it doesn't exist
    output_dir = 'resultados_estatisticos'
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    output_file = os.path.join(output_dir, f'comparacao_resultados_{title_prefix.lower()}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    # Print t-test results with formatted output
    print(f"\n{'='*80}")
    print(f"RESULTADOS DOS TESTES T - {title_prefix}")
    print(f"{'='*80}")
    
    for _, row in results_df.iterrows():
        print(f"\nComparação: {row['Comparison']}")
        print(f"{'-'*40}")
        print(f"T-statistic: {row['t-statistic']:.6f}")
        print(f"P-valor: {row['p-value (%)']:.6f}%")
        significancia = "SIGNIFICATIVO" if row['p-value'] < 0.05 else "NÃO SIGNIFICATIVO"
        print(f"Significância: {significancia}")
        print(f"{'-'*40}")

    print(f"\nGráfico salvo em: {output_file}")

def main():
    print("\nIniciando análise estatística...")
    print(f"Diretório de resultados: {os.path.abspath(RESULTS_DIR)}")
    
    # Analyze both star and mack data
    analyze_data(star_files, "Estrela")
    # analyze_data(mack_files, "Mack")
    # analyze_data(raposa_files, "Raposa")
    
    print("\nAnálise estatística concluída!")

if __name__ == "__main__":
    main()
