import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from model.cy_utils import coupling_experiment, identify_causality, calculate_causal_history, generate_pattern_dictionary, calculate_contribution_analysis, average_weighted_entropy, discrete
import matplotlib.ticker as mtick

def save_dataset_to_json(num_pairs=2000, seq_len=1500, transients=500, filename='dataset/experiment-2-data.json'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    raw_data = coupling_experiment(num_pairs, seq_len, transients)
    
    json_ready_data = {}
    for phi, sequences in raw_data.items():
        json_ready_data[str(phi)] = {
            'X': sequences['X'].tolist(),
            'Y': sequences['Y'].tolist()
        }
    
    with open(filename, 'w') as f:
        json.dump(json_ready_data, f)
    
    print(f"Dataset successfully saved to {filename}")

def extract_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    processed_data = {}
    for phi, sequences in data.items():
        processed_data[phi] = {
            'X': np.array(sequences['X']),
            'Y': np.array(sequences['Y'])
        }
    return processed_data

def run_causal_analysis(data, output_file, ground_truth='Y -> X'):
    phi_stats = []

    for phi, sequences in data.items():
        phi_val = float(phi)
        h_bar_list_xy, h_bar_list_yx = [], []
        tp, fp, fn, total_samples = 0, 0, 0, 0

        X_batch = sequences['X']
        Y_batch = sequences['Y']

        num_trials = X_batch.shape[0]
        seq_len = X_batch.shape[1]

        for i in range(num_trials):
            X_bin, Y_bin = discrete(X_batch[i]), discrete(Y_batch[i])

            gx_y = calculate_causal_history(Y_bin, X_bin)
            px_y = generate_pattern_dictionary(gx_y)
            stats_xy = calculate_contribution_analysis(px_y, X_bin, Y_bin)
            h_xy = average_weighted_entropy(stats_xy, seq_len)

            gy_x = calculate_causal_history(X_bin, Y_bin)
            py_x = generate_pattern_dictionary(gy_x)
            stats_yx = calculate_contribution_analysis(py_x, Y_bin, X_bin)
            h_yx = average_weighted_entropy(stats_yx, seq_len)

            prediction, _, _ = identify_causality(stats_xy, stats_yx, X_bin, Y_bin)
            
            if prediction == ground_truth:
                tp += 1
            else:
                fn += 1
            
            total_samples += 1
            h_bar_list_xy.append(h_xy)
            h_bar_list_yx.append(h_yx)

        accuracy = tp / total_samples if total_samples > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        phi_stats.append({
            'phi': phi_val,
            'sum_H_xy': np.mean(h_bar_list_xy),
            'sum_H_yx': np.mean(h_bar_list_yx),
            'std_H_xy': np.std(h_bar_list_xy),
            'std_H_yx': np.std(h_bar_list_yx),
            'accuracy': accuracy,
            'f1': f1
        })

    df = pd.DataFrame(phi_stats).sort_values('phi')
    os.makedirs('results', exist_ok=True)
    print(f'Results saved: {output_file}')
    df.to_csv(output_file, index=False)
    return df

# # plot functions
def plot_mean_std(df, output_file):
    df_plot = df.groupby('phi', as_index=False).agg({
        'sum_H_xy': 'mean',
        'sum_H_yx': 'mean',
        'std_H_xy': 'mean', 
        'std_H_yx': 'mean'
    }).sort_values('phi')

    plt.figure(figsize=(9, 7))
    
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_major_locator(mtick.MaxNLocator(nbins=10)) 

    plt.plot(df_plot['phi'], df_plot['sum_H_xy'], color='C0', alpha=0.3, linewidth=1, label='_nolegend_')
    plt.plot(df_plot['phi'], df_plot['sum_H_yx'], color='C1', alpha=0.3, linewidth=1, label='_nolegend_')
    
    plt.errorbar(df_plot['phi'], df_plot['sum_H_xy'], yerr=df_plot['std_H_xy'], 
                 fmt='o', capsize=5, color='C0', markersize=8, label=r'$H_{X \to Y}$')
    plt.errorbar(df_plot['phi'], df_plot['sum_H_yx'], yerr=df_plot['std_H_yx'], 
                 fmt='s', capsize=5, color='C1', markersize=8, label=r'$H_{Y \to X}$')  
    
    plt.xlabel(r'Coupling Strength ($\eta$)', fontsize=21, fontweight='bold')
    plt.ylabel('Mean Entropy (H)', fontsize=21, fontweight='bold')
    
    plt.xticks(df_plot['phi'].unique(), fontsize=14, rotation=90)
    
    plt.tick_params(axis='x', labelsize=26)
    plt.tick_params(axis='y', labelsize=26)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend([r'$\bar{H}_{X\to Y}$', r'$\bar{H}_{Y\to X}$'],fontsize=22, loc='best', frameon=True, shadow=True)
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    print(f'Figure:{output_file}')
    plt.savefig(output_file, dpi=300)
    plt.show()
    plt.close()

# # =======================================
# Experiment-2 AR(1) Coupling Process MSE
# # =======================================
# loading dataset
# file_path = 'dataset/experiment-2-data.json'
# data = extract_json_data(file_path)

# # run analysis
# results_df = run_causal_analysis(data, 'results/experiment-2-mse-results.csv')
# print(results_df)

# # analysis and plotting
# results_df = pd.read_csv('results/experiment-2-mse-results.csv')
# plot_mean_std(results_df, 'results/experiment-2-mean_std.png')

# # =======================================
# Experiment-4 1D Coupled Skewtent Maps
# # =======================================
# loading dataset
# file_path = 'dataset/experiment-4-data.json'
# data = extract_json_data(file_path)

# # run analysis
# results_df = run_causal_analysis(data, 'results/experiment-4-mse-results.csv')
# print(results_df)

# # analysis and plotting
# results_df = pd.read_csv('results/experiment-4-mse-results.csv')
# plot_mean_std(results_df, 'results/experiment-4-mean_std.png')