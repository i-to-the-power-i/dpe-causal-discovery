import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import array
from model.cy_utils import run_causal_analysis, discrete, generate_coupled_data
from model.utils import extract_json_data
import ETCPy.ETC.CCMC.pairs as pairs
import json

def symbolic_to_array(seq_str):

    unique_symbols = sorted(list(set(seq_str)))
    symbol_map = {symbol: i+1 for i, symbol in enumerate(unique_symbols)}
    int_list = [symbol_map[symbol] for symbol in seq_str]
    return array.array('I', int_list)

def run_comprehensive_experiment(data, file_name='results/experiment-2-all-model-performance.csv'):
    results_list = []

    for eta, content in data.items():
        eta_val = float(eta)
        print(f"Processing η = {eta}...")
        
        master_trials = content['Y']
        slave_trials = content['X']
        num_trials = len(master_trials)

        stats = {
            'DPE': {'tp': 0, 'decisions': 0},
            'ETC-P':     {'tp': 0, 'decisions': 0},
            'ETC-E':     {'tp': 0, 'decisions': 0},
            'LZ-P':      {'tp': 0, 'decisions': 0}
        }

        for i in range(num_trials):
            m_raw = np.array(master_trials[i], dtype=np.float32)
            s_raw = np.array(slave_trials[i], dtype=np.float32)
            
            m_bin = discrete(m_raw, n_bins=2)
            s_bin = discrete(s_raw, n_bins=2)

            verdict, _, _ = run_causal_analysis(s_bin, m_bin)
            if verdict != "Independent": 
                stats['DPE']['decisions'] += 1
                if verdict == "Y -> X" and eta_val > 0: stats['DPE']['tp'] += 1

            m_etc = symbolic_to_array(m_bin.astype(int).astype(str))
            s_etc = symbolic_to_array(s_bin.astype(int).astype(str))
            etc_res = pairs.CCM_causality(s_etc, m_etc)

            for m_key, res_prefix in [('ETC-P', 'ETCP'), ('ETC-E', 'ETCE'), ('LZ-P', 'LZP')]:
                direction = etc_res.get(f'{res_prefix}_direction')
                if direction in ["y_causes_x", "x_causes_y"]:
                    stats[m_key]['decisions'] += 1
                    if direction == "y_causes_x" and eta_val > 0: stats[m_key]['tp'] += 1

        for model_name, val in stats.items():
            acc = val['tp'] / num_trials
            dr = val['decisions'] / num_trials
            results_list.append({
                'eta': eta_val,
                'model': model_name,
                'accuracy': round(acc, 4),
                'decision_rate': round(dr, 4)
            })
    results_list = pd.DataFrame(results_list)
    results_list.to_csv(file_name)
    print(f'Results saved: {file_name}')
    return results_list

def plot_accuracy_vs_eta(df, fig_name='results/experiment-2-all-model-performance.png'):
    plt.figure(figsize=(9, 7))
    markers = ['o', 'v', 's', 'p']
    
    for i, model in enumerate(df['model'].unique()):
        subset = df[df['model'] == model].sort_values('eta')
        plt.plot(subset['eta'], subset['accuracy'], 
                 marker=markers[i % len(markers)], 
                 label=model, linewidth=2)
    
    fs = 26
    
    plt.xlabel(r'Coupling Strength ($\eta$)', fontsize=fs-5, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=fs-5, fontweight='bold')

    ax = plt.gca()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        # label.set_fontweight('bold')
        label.set_fontsize(fs)

    leg = plt.legend(
        bbox_to_anchor=(0.98, 0.80), 
        loc='center right',
        frameon=True,
        framealpha=0.8,
        fontsize=fs-5
    )
    
    # for text in leg.get_texts():
    #     text.set_fontweight('bold')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    print(f'Figure: {fig_name}')
    plt.savefig(fig_name, dpi=300)
    plt.show()


# # Extra Plots

# def plot_decision_rate_vs_eta(df):
#     plt.figure(figsize=(10, 6))
#     for model in df['model'].unique():
#         subset = df[df['model'] == model].sort_values('eta')
#         plt.plot(subset['eta'], subset['decision_rate'], marker='s', linestyle='--', label=model)
    
#     # plt.title('Decision Rate vs. Coupling Strength ($\eta$)', fontsize=14)
#     plt.xlabel('Coupling Strength ($\eta$)')
#     plt.ylabel('Decision Rate (Decisions / Total Trials)')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.savefig('results/all_models_decision_rate_eta.png')
#     plt.show()

# def plot_accuracy_vs_decision_rate(df):
#     plt.figure(figsize=(10, 6))
#     for model in df['model'].unique():
#         subset = df[df['model'] == model]
#         plt.scatter(subset['decision_rate'], subset['accuracy'], label=model, s=80, alpha=0.7)
        
#     # plt.title('Accuracy vs. Decision Rate', fontsize=14)
#     plt.xlabel('Decision Rate')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.savefig('results/all_models_accuracy_decision_rate.png')
#     plt.grid(True, alpha=0.3)
#     plt.show()


# # =============================
# # Experiment-2: AR(1) Coupling
# # =============================
# # Generate data
# file_path_1 = 'dataset/experiment-2-data.json'

# Don't run this part again as it will create new data (dataset is available on the file path)
# data = coupling_experiment(num_sequences= 2000, sequence_length= 1500, transients= 500)
# with open(file_path_1, 'w') as file:
#     json.dump(data, file)

# Extract data
# data = extract_json_data(file_path_1)

# # Run experiment
# results_df = run_comprehensive_experiment(data)

# results_df = pd.read_csv('results/experiment-2-all-model-performance.csv') # In case if you want to plot results again and again.

# plot_accuracy_vs_eta(results_df, 'results/experiment-2-all-model-performance.png')

# # =============================
# # Experiment-4: 1D Coupled Skewtent Maps
# # =============================
# # Generate data
# file_path_2 = 'dataset/experiment-4-data.json'

# Don't run this part again as it will create new data (dataset is available on the file path)
# data = generate_coupled_data(b1=0.35, b2=0.76, initial_values=None)
# with open(file_path_2, 'w') as file:
#     json.dump(data, file)

# Extract data
# file_path_2 = 'dataset/experiment-4-data.json'
# data = extract_json_data(file_path_2)

# # Run experiment
# results_df = run_comprehensive_experiment(data, file_name='results/experiment-4-all-model-performance.csv')

# results_df = pd.read_csv('results/experiment-4-all-model-performance.csv') # In case if you want to plot results again and again.

# # Plots
# plot_accuracy_vs_eta(results_df, fig_name='results/experiment-4-all-model-performance.png')





