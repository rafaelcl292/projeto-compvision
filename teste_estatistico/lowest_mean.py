import os
import numpy as np
from itertools import combinations

def read_transformation_results(file_path):
    with open(file_path, 'r') as f:
        values = [float(line.strip()) for line in f if line.strip()]
    return values

# Directory containing the transformation results
results_dir = 'transformation_results/raposa'

# Get all result files
result_files = [f for f in os.listdir(results_dir) if f.endswith('.txt')]

# Calculate mean for each player and store in dictionary
player_means = {}
for file_name in result_files:
    file_path = os.path.join(results_dir, file_name)
    values = read_transformation_results(file_path)
    mean_value = np.mean(values)
    player_name = file_name.replace('transformation_results_', '').replace('_raposa.txt', '')
    player_means[player_name] = mean_value
    print(f"\nPlayer: {player_name}")
    print(f"Number of values: {len(values)}")
    print(f"Mean: {mean_value:.6f}")

# Calculate differences between all pairs of players
min_diff = float('inf')
min_diff_pair = None

for (player1, mean1), (player2, mean2) in combinations(player_means.items(), 2):
    diff = abs(mean1 - mean2)
    if diff < min_diff:
        min_diff = diff
        min_diff_pair = (player1, player2, mean1, mean2)

print(f"\nSmallest difference between players:")
print(f"Players: {min_diff_pair[0]} and {min_diff_pair[1]}")
print(f"Means: {min_diff_pair[2]:.6f} and {min_diff_pair[3]:.6f}")
print(f"Difference: {min_diff:.6f}")
