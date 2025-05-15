import numpy as np
from scipy import stats

# Read the data from both files
with open('transformation_results.txt', 'r') as f1:
    data1 = np.array([float(line.strip()) for line in f1 if line.strip()])

with open('transformation_results2.txt', 'r') as f2:
    data2 = np.array([float(line.strip()) for line in f2 if line.strip()])

# Perform t-test
t_stat, p_value = stats.ttest_ind(data1, data2)

# Print results
print("T-Test Results:")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print("\nSummary Statistics:")
print("\nDataset 1:")
print(f"Mean: {np.mean(data1):.4f}")
print(f"Std: {np.std(data1):.4f}")
print(f"Size: {len(data1)}")
print("\nDataset 2:")
print(f"Mean: {np.mean(data2):.4f}")
print(f"Std: {np.std(data2):.4f}")
print(f"Size: {len(data2)}")