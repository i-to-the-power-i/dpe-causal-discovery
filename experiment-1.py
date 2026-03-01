# Experiment-1
# Delayed bit flip for pattern 1101

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import array
import random
import json
import ETCPy.ETC.CCMC.pairs as pairs
from model.cy_utils import calculate_causal_history, generate_pattern_dictionary, calculate_contribution_analysis, identify_causality, generate_coupled_data, run_causal_analysis
import os

PATTERN = '1101'

def generate_main_pattern(length=100):
    sequence = ""
    bits = ['0', '1', PATTERN]

    while len(sequence) < length:
        choice = random.choice(bits)
        sequence += choice

    return sequence[:length]

def generate_delayed_flip(main_sequence, delay):
    length = len(main_sequence)
    delayed_seq = ['0'] * length
    
    for i in range(len(PATTERN) - 1, length):
        if main_sequence[i - len(PATTERN) + 1 : i + 1] == PATTERN:
            target_idx = i + delay
            if target_idx < length:
                delayed_seq[target_idx] = '1'
                
    return "".join(delayed_seq)

def generate_n_sequences(trials=1000, length=100):
    data = {delay: [] for delay in range(7)}

    for i in range(trials):
        main_seq = generate_main_pattern(length)

        for delay in range(7):
            delayed_seq = generate_delayed_flip(main_seq, delay)
            
            data[delay].append({
                'trial_id': i,
                'X': main_seq,
                'Y': delayed_seq
            })

    with open('dataset/experiment-1.json', 'w') as file:
        json.dump(data, file, indent=4)
        
    return data

def analyze_models_by_delay(data):
    results = []
    models = ['DPE', 'ETC-P', 'ETC-E', 'LZ-P']

    for delay, trials in data.items():
        print(f"Analyzing Delay: {delay}...")
        counts = {model: {'Y -> X': 0, 'X -> Y': 0, 'Ind': 0} for model in models}

        for trial in trials:
            x_str, y_str = trial['X'], trial['Y']
            
            x_raw = np.array(list(x_str), dtype=np.float32)
            y_raw = np.array(list(y_str), dtype=np.float32)
            
            x_etc = array.array('I', [int(b) + 1 for b in x_str])
            y_etc = array.array('I', [int(b) + 1 for b in y_str])

            verdict_nm, _, _ = run_causal_analysis(x_raw, y_raw)
            nm_key = verdict_nm if verdict_nm in ['Y -> X', 'X -> Y'] else 'Ind'
            counts['DPE'][nm_key] += 1

            try:
                res = pairs.CCM_causality(x_etc, y_etc)
                
                def map_dir(val):
                    mapping = {'y_causes_x': 'Y -> X', 'x_causes_y': 'X -> Y'}
                    return mapping.get(val, 'Ind')

                counts['ETC-P'][map_dir(res['ETCP_direction'])] += 1
                counts['ETC-E'][map_dir(res['ETCE_direction'])] += 1
                counts['LZ-P'][map_dir(res['LZP_direction'])] += 1
                
            except Exception as e:
                print(f"Error processing delay {delay}: {e}")
                for m in ['ETC-P', 'ETC-E', 'LZ-P']:
                    counts[m]['Ind'] += 1

        for model in models:
            results.append({
                'Delay': delay,
                'Model': model,
                'Y -> X': counts[model]['Y -> X'],
                'X -> Y': counts[model]['X -> Y'],
                'Independence': counts[model]['Ind']
            })

    return pd.DataFrame(results)

# # Extra plot
# def plot_model_comparison_bars(df_results):
#     import numpy as np
#     import matplotlib.pyplot as plt

#     models = df_results['Model'].unique()
#     delays = df_results['Delay'].unique()

#     fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
#     axes = axes.flatten()

#     y = np.arange(len(delays))
#     height = 0.25

#     for i, model in enumerate(models):
#         model_df = df_results[df_results['Model'] == model].sort_values('Delay')
#         ax = axes[i]

#         rects1 = ax.barh(y - height, model_df['Y -> X'], height,
#                          label='Y -> X (False)', color="#242020", edgecolor='black')
#         rects2 = ax.barh(y, model_df['X -> Y'], height,
#                          label='X -> Y (True)', color="#4e4e4e", edgecolor='black')
#         rects3 = ax.barh(y + height, model_df['Independence'], height,
#                          label='Independence', color="#fffefe", edgecolor='black')

#         for rects in [rects1, rects2, rects3]:
#             for rect in rects:
#                 value = rect.get_width()
#                 ax.text(
#                     value + 15,                              
#                     rect.get_y() + rect.get_height() / 2,    
#                     f'{int(value)}',
#                     va='center',
#                     fontsize=20,
#                     # fontweight='bold',
#                     bbox=dict(
#                         boxstyle='round,pad=0.10',           
#                         facecolor='white',
#                         edgecolor='none',
#                         alpha=0.5
#                     )
#                 )

#         ax.set_title(f'{model}', fontsize=18, fontweight='bold')

#         if i in [0, 2]:
#             ax.set_ylabel('Delay Value', fontsize=21, fontweight='bold')
#         if i in [2, 3]:
#             ax.set_xlabel('Trial Count', fontsize=21, fontweight='bold')

#         if i in [0, 2]:
#             ax.set_yticks(y)
#             ax.set_yticklabels(delays, fontsize=22)
#         else:
#             ax.set_yticks([])

#         if i in [2, 3]:
#             ax.tick_params(axis='x', labelsize=22)
#         else:
#             ax.set_xticks([])

#         ax.set_xlim(0, 1200)

#         if i == 3:
#             ax.legend(fontsize=16, loc='upper center')

#     plt.savefig('results/model_delay_comparison_horizontal.png', dpi=300)
#     plt.show()

def extract_json_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file at {file_path} does not exist.")
        return None

    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
def plot_model_accuracy(df_results, fig_name='results/model_accuracy_comparison.png'):
    df_results['Total'] = df_results['Y -> X'] + df_results['X -> Y'] + df_results['Independence']
    df_results['Accuracy'] = df_results['X -> Y'] / df_results['Total']
    
    df_results['Delay'] = df_results['Delay'].astype(int)

    plt.figure(figsize=(9, 7))
    
    marker_map = {'DPE': 'o', 'ETC-P': 's', 'ETC-E': 'D', 'LZ-P': '^'}
    line_map = {'DPE': '-', 'ETC-P': '-.', 'ETC-E': '--', 'LZ-P': '-.'}
    # color_map = {
    #     'DPE': "#00BFFF",    
    #     'ETC-P': "#FF6200",  
    #     'ETC-E': "#51D051",  
    #     'LZ-P': '#FF0000'    
    # }

    fs = 26
    models = ['DPE', 'ETC-P', 'ETC-E', 'LZ-P']

    for model in models:
        if model in df_results['Model'].values:
            subset = df_results[df_results['Model'] == model].sort_values('Delay')
            plt.plot(
                subset['Delay'], 
                subset['Accuracy'], 
                label=model,
                marker=marker_map.get(model, 'o'),
                linestyle=line_map.get(model, '-'),
                # color=color_map.get(model, 'black'),
                linewidth=3,        
                markersize=12,      
                # markeredgecolor='black',
                # markeredgewidth=1
            )

    plt.xlabel('Delay Value', fontsize=fs-5, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=fs-5, fontweight='bold')

    ax = plt.gca()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(fs)

    plt.legend(
        bbox_to_anchor=(0.98, 0.75), 
        loc='center right',
        frameon=True,
        framealpha=0.8,
        fontsize=fs-7  
    )

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.ylim(-0.05, 1.05)
    
    plt.tight_layout()
    print(f'Figure 2: {fig_name}')
    plt.savefig(fig_name, dpi=300)
    plt.show()


# # Generate Data
# Don't run this part again as it will create new data (dataset is available on the file path)
# # data = generate_n_sequences(1000)

file_path = 'dataset/experiment-1.json'
data = extract_json_data(file_path)
# print(data)

# 2. Run Analysis
results_df = analyze_models_by_delay(data)

# 3. Plot
# plot_model_comparison_bars(results_df)
plot_model_accuracy(results_df)
