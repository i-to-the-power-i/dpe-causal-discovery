# Experiment 4
# Varying Sparsity

import numpy as np
import cvxpy as cp
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import array
from model.cy_utils import run_causal_analysis #, discrete
import ETCPy.ETC.CCMC.pairs as pairs

# # Method to understand- How Sparsity works?
# def sparse_signals(n, k, alpha=0.8, beta=0.08, gamma=0.75):
#     t_indices = np.arange(n-1)
#     t1_set = np.random.choice(t_indices, size=k, replace=False)
#     t2_set = t1_set + 1

#     Z1 = np.zeros(n)
#     z1 = np.zeros(n)
#     Z2 = np.zeros(n)
#     z2 = np.zeros(n)

#     eps1 = np.random.normal(0, np.sqrt(0.1), n)
#     eps2 = np.random.normal(0, np.sqrt(0.1), n)

#     for t in range(1, n):
#         Z1[t] = alpha * Z1[t-1] + eps1[t]

#         if t in t1_set:
#             z1[t] = Z1[t]

#         Z2[t] = beta * Z2[t-1] + gamma * z1[t-1] + eps2[t]

#         if t in t2_set:
#             z2[t] = Z2[t]
    
#     return Z1, Z2, z1, z2

# Z1, Z2, z1, z2 = sparse_signals(10, 5)

# print(f'{Z1}\n{Z2}\n{z1}\n{z2}')

def generate_sparse_data(n=2000, m=200, k_values=range(5, 55, 5), num_trials=100):
    alpha, beta, gamma = 0.8, 0.08, 0.75
    # transients = 500
    all_data = {}

    for k in k_values:
        k_key = str(k)
        all_data[k_key] = []
        print(f"Generating data for k={k}...")

        for trial in range(num_trials):
            total_len = n 
            eps1 = np.random.normal(0, np.sqrt(0.1), total_len)
            eps2 = np.random.normal(0, np.sqrt(0.1), total_len)
            Z1, Z2 = np.zeros(total_len), np.zeros(total_len)
            rng = np.random.default_rng()
            t1_set = rng.choice(n - 1, size=k, replace=False)
            t2_set = t1_set + 1
            z1, z2 = np.zeros(total_len), np.zeros(total_len)

            for t in range(1, n):
                Z1[t] = alpha * Z1[t-1] + eps1[t]
                if t in t1_set:
                    z1[t] = Z1[t]
            
            for t in range(1, n):
                Z2[t] = beta * Z2[t-1] + gamma * z1[t-1] + eps2[t]
                if t in t2_set:
                    z2[t] = Z2[t]
            
            C = np.random.randn(m, n) / np.sqrt(m)
            y1, y2 = C @ z1, C @ z2
            
            z1_rec = reconstruction(y1, C)
            z2_rec = reconstruction(y2, C)

            trial_entry = {
                "initial_Z1": Z1[0], "initial_Z2": Z2[0],
                "z1_orig": z1.tolist(), "z2_orig": z2.tolist(),
                "z1_rec": z1_rec.tolist(), "z2_rec": z2_rec.tolist()
            }
            all_data[k_key].append(trial_entry)

    with open('dataset/experiment-3.json', 'w') as f:
        json.dump(all_data, f)
    return all_data

def reconstruction(y, C):
    n = C.shape[1]
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.norm1(x)), [C @ x == y])
    
    try:
        prob.solve() 
    except cp.error.SolverError:
        prob.solve(solver=cp.SCS)
        
    return x.value if x.value is not None else np.zeros(n)

def discrete(sequence, n_bins=2):
    arr = np.array(sequence)
    
    return (arr != 0).astype(int)

def analyze_all_methods(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = []
    models = ['DPE', 'ETC-P', 'ETC-E', 'LZ-P']
    
    for k_val, trials in data.items():
        print(f"Analyzing sparsity k={k_val}...")
        for data_type in ['Original', 'Reconstructed']:
            stats = {m: {'x_to_y': 0, 'y_to_x': 0, 'ind': 0} for m in models}
            
            for trial in trials:
                s1 = np.array(trial['z1_orig' if data_type == 'Original' else 'z1_rec'])
                s2 = np.array(trial['z2_orig' if data_type == 'Original' else 'z2_rec'])
                
                str_z1 = ''.join(map(str, discrete(s1, n_bins=2)))
                str_z2 = ''.join(map(str, discrete(s2, n_bins=2)))

                # Previous ---

                # etc_z1 = array.array('I', (((s1 - s1.min()) / (s1.max() - s1.min()) * 255)+1).astype(np.uint32))
                # etc_z2 = array.array('I', (((s2 - s2.min()) / (s2.max() - s2.min()) * 255)+1).astype(np.uint32))

                bin_result_1 = discrete(s1, 2)
                bin_result_2 = discrete(s2, 2)

                etc_z1 = array.array('I', (bin_result_1 + 1).astype(np.uint32))
                etc_z2 = array.array('I', (bin_result_2 + 1).astype(np.uint32))

                # new ---
                # z1_final = ((s1 != 0).astype(np.uint32))
                # z2_final = ((s2 != 0).astype(np.uint32))

                # etc_z1 = array.array('I', z1_final)
                # etc_z2 = array.array('I', z2_final)

                verdict, _, _ = run_causal_analysis(str_z1, str_z2)
                update_stats(stats, 'DPE', verdict)

                res = pairs.CCM_causality(etc_z1, etc_z2)
                if res:
                    for m, key in [('ETC-P', 'ETCP_direction'), ('ETC-E', 'ETCE_direction'), ('LZ-P', 'LZP_direction')]:
                        update_stats(stats, m, res.get(key))

            for m in models:
                results.append({
                    'k': int(k_val), 'Model': m, 'Type': data_type,
                    'X_to_Y': stats[m]['x_to_y'], 'Y_to_X': stats[m]['y_to_x'], 'Ind': stats[m]['ind'],
                    'Accuracy': stats[m]['x_to_y'] / len(trials)
                })
    final_df = pd.DataFrame(results)
    
    os.makedirs('results', exist_ok=True)
    final_df.to_csv('results/discrete_cs_causality_results.csv', index=False)
    print(f'Results saved: results/discrete_cs_causality_results.csv')
    return final_df

def update_stats(stats, model, verdict):
    if verdict in ["X -> Y", "x_causes_y"]: stats[model]['x_to_y'] += 1
    elif verdict in ["Y -> X", "y_causes_x"]: stats[model]['y_to_x'] += 1
    else: stats[model]['ind'] += 1


# # Extra plot
# def plot_results(df):

#     models = df['Model'].unique()
#     k_values = sorted(df['k'].unique())

#     for data_type in ['Original', 'Reconstructed']:
#         fig, axes = plt.subplots(
#             2, 2,
#             figsize=(10, 8),
#             constrained_layout=True
#         )
#         axes = axes.flatten()

#         type_df = df[df['Type'] == data_type]

#         y = np.arange(len(k_values))

#         height = 0.18
#         offsets = [-height, 0, height]

#         for i, model in enumerate(models):
#             model_df = type_df[type_df['Model'] == model].sort_values('k')
#             ax = axes[i]

#             rects1 = ax.barh(
#                 y + offsets[0],
#                 model_df['X_to_Y'],
#                 height,
#                 label='X → Y (True)',
#                 color="#242020",
#                 edgecolor='black'
#             )
#             rects2 = ax.barh(
#                 y + offsets[1],
#                 model_df['Y_to_X'],
#                 height,
#                 label='Y → X (False)',
#                 color="#4e4e4e",
#                 edgecolor='black'
#             )
#             rects3 = ax.barh(
#                 y + offsets[2],
#                 model_df['Ind'],
#                 height,
#                 label='Independence',
#                 color="#e0e0e0",
#                 edgecolor='black'
#             )

#             for rects in [rects1, rects2, rects3]:
#                 for rect in rects:
#                     value = rect.get_width()

#                     inside = value > 15

#                     x = value - 4 if inside else value + 2
#                     ha = 'right' if inside else 'left'
#                     color = 'white' if inside else 'black'

#                     ax.text(
#                         x,
#                         rect.get_y() + rect.get_height() / 2,
#                         f'{int(value)}',
#                         va='center',
#                         ha=ha,
#                         fontsize=14,
#                         fontweight='bold',
#                         color=color
#                     )

#             ax.set_title(model, fontsize=16, fontweight='bold')

#             if i in [0, 2]:
#                 ax.set_yticks(y)
#                 ax.set_yticklabels(k_values, fontsize=14)
#                 ax.set_ylabel("Sparsity (k)", fontsize=16, fontweight='bold')
#             else:
#                 ax.set_yticks([])

#             if i in [2, 3]:
#                 ax.tick_params(axis='x', labelsize=14)
#                 ax.set_xlabel("Trial Count", fontsize=16, fontweight='bold')
#             else:
#                 ax.set_xticks([])

#             ax.set_xlim(0, 120)

#             ax.grid(axis='x', linestyle='--', alpha=0.4)

#             if i == 0:
#                 ax.legend(fontsize=13, loc='upper right')

#         plt.savefig(
#             f'results/discrete_barcharts_all_models_horizontal_{data_type.lower()}.png',
#             dpi=300,
#             bbox_inches='tight'
#         )
#         plt.show()

plt.rcParams['figure.dpi'] = 120
def plot_results_acc(df):
    sns.set_style("whitegrid")
    
    model_markers = {
        'DPE': 'o', 
        'ETC-P': 's', 
        'ETC-E': 'D', 
        'LZ-P': '^'
    }
    
    for data_type in ['Original', 'Reconstructed']:
        plt.figure(figsize=(9, 7))
        sub_df = df[df['Type'] == data_type]
        
        sns.lineplot(
            data=sub_df,
            x='k',
            y='Accuracy',
            hue='Model',
            style='Model',
            markers=model_markers,
            markersize=10,
            linewidth=2.5
        )
        
        plt.tick_params(axis='both', which='major', labelsize=26, length=6)
        plt.ylabel("Accuracy", fontsize=21, fontweight='bold')
        plt.xlabel("Sparsity (k)", fontsize=21, fontweight='bold')
        plt.ylim(-0.05, 1.05)
        plt.legend(bbox_to_anchor=(0.98, 0.75), loc='center right', fontsize=22)
        plt.tight_layout()
        print(f'Figure 5: results/discrete_accuracy_line_{data_type.lower()}.png')
        plt.savefig(f"results/discrete_accuracy_line_{data_type.lower()}.png", dpi=300)
        plt.show()

    # # Extra plot
    # =======================
    # Per-model bar charts
    # =======================
    # models = df['Model'].unique()
    # k_values = sorted(df['k'].unique())
    
    # type_df = df[df['Type'] == 'Original']

    # for model in models:
    #     model_df = type_df[type_df['Model'] == model].sort_values('k')

    #     fig, ax = plt.subplots(figsize=(10, 8))
    #     plt.subplots_adjust(
    #         left=0.10,
    #         right=0.98,
    #         bottom=0.12,
    #         top=0.90
    #     )

    #     x = np.arange(len(k_values))
    #     width = 0.30
    #     gap = 1.35

    #     rects1 = ax.bar(
    #         x - gap * width, model_df['X_to_Y'], width,
    #         label='X → Y (True)', color="#2c3e50", edgecolor='black'
    #     )
    #     rects2 = ax.bar(
    #         x, model_df['Y_to_X'], width,
    #         label='Y → X (False)', color="#95a5a6", edgecolor='black'
    #     )
    #     rects3 = ax.bar(
    #         x + gap * width, model_df['Ind'], width,
    #         label='Independence', color="#ecf0f1", edgecolor='black'
    #     )

    #     ax.bar_label(rects1, padding=12, fontsize=22, rotation=90, fontweight='bold')
    #     ax.bar_label(rects2, padding=12, fontsize=22, rotation=90, fontweight='bold')
    #     ax.bar_label(rects3, padding=12, fontsize=22, rotation=90, fontweight='bold')

    #     ax.set_title(model, fontsize=22, fontweight='bold', pad=20)
    #     ax.set_xticks(x)
    #     ax.set_xticklabels(k_values)

    #     ax.set_xlabel("Sparsity (k)", fontsize=20, fontweight='bold')
    #     ax.set_ylabel("Trial Count", fontsize=20, fontweight='bold')

    #     ax.tick_params(axis='both', which='major', labelsize=24, length=6)

    #     max_height = model_df[['X_to_Y', 'Y_to_X', 'Ind']].values.max()
    #     ax.set_ylim(0, max_height * 1.6)

    #     ax.legend(loc='upper right', fontsize=18)
    #     ax.grid(axis='y', linestyle='--', alpha=0.5)

    #     plt.tight_layout()
    #     plt.savefig(f"results/discrete_barchart_{model.lower()}.png", dpi=300)
    #     plt.show()


# Don't run this part again as it will create new data (dataset is available on the file path)
# # data = generate_sparse_data(num_trials=100)

results_df = analyze_all_methods('dataset/experiment-3.json')
# results_df = pd.read_csv('results/discrete_cs_causality_results.csv') # In case if you want to plot results again and again.

plot_results_acc(results_df)
# plot_results(results_df)