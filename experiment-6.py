# Experiment-6
#  Real Data: Predator-Prey 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.cy_utils import run_causal_analysis, discrete
import ETCPy.ETC.CCMC.pairs as pairs
import array

FILE = 'dataset/prey_predator_final.csv'

# extract the pairs
# def extract_csv(file_path):
#     data = pd.read_csv(file_path)
#     # remove first 10 transients
#     data = data.iloc[9: ].reset_index(drop=True)

#     return data

# def csv_to_str(data):
#     predator = data['Didinium'].to_numpy()
#     prey = data['Paramecium'].to_numpy()

#     predator_bin = discrete(predator)
#     prey_bin = discrete(prey)

#     predator_str = ''.join(str(i) for i in predator_bin)
#     prey_str = ''.join(str(i) for i in prey_bin)

#     return predator_str, prey_str

def plot_data(data):
    plt.figure(figsize=(9, 7))
    plt.plot(
    data['Didinium'],
    color='black',
    marker='^',
    linestyle='-',
    linewidth=1,
    markersize=4,
    label='Predator (Didinium)'
    )

    plt.plot(
        data['Paramecium'],
        color='red',
        marker='o',
        linestyle='--',
        linewidth=1,
        markersize=4,
        label='Prey (Paramecium)'
    )

    plt.xlabel("Time (days)", fontsize=21, fontweight='bold')
    plt.ylabel("Abundance (# ml)", fontsize=21, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.legend(fontsize=22, loc='upper left')
    plt.grid(True, linestyle='-', alpha=0.3)

    plt.xlim(-1, 65)
    plt.ylim(0, 400)

    plt.tight_layout()
    plt.savefig('results/predator_prey.png', dpi=300)
    print(f'Figure 9: results/predator_prey.png')
    plt.show()

# data = extract_csv(FILE)
# plot_data(data)

# predator, prey = csv_to_str(data)
# verdict, Havg_predator_prey, Havg_prey_predator = run_causal_analysis(predator, prey)

# print(f'''
# Direction: {verdict.replace('X', 'Predator').replace('Y', 'Prey')}
# Entropy (Predator -> Prey): {round(Havg_predator_prey, 4)}
# Entropy (Prey -> Predactor): {round(Havg_prey_predator, 4)}
# ''')

# # =============================

def extract_csv(file_path):
    data = pd.read_csv(file_path)
    data = data.iloc[9:].reset_index(drop=True)
    return data

def symbolic_to_array(seq_str):
    unique_symbols = sorted(list(set(seq_str)))
    symbol_map = {symbol: i+1 for i, symbol in enumerate(unique_symbols)}
    int_list = [symbol_map[symbol] for symbol in seq_str]
    return array.array('I', int_list)

def run_predator_prey_comparison(file_path):
    data = extract_csv(file_path)
    
    predator_raw = data['Didinium'].to_numpy()
    prey_raw = data['Paramecium'].to_numpy()

    pred_bin = discrete(predator_raw, n_bins=2)
    prey_bin = discrete(prey_raw, n_bins=2)

    pred_str = ''.join(str(int(i)) for i in pred_bin)
    prey_str = ''.join(str(int(i)) for i in prey_bin)

    pred_etc = symbolic_to_array(pred_str)
    prey_etc = symbolic_to_array(prey_str)

    results = {}

    verdict_dpe, h_pred_prey, h_prey_pred = run_causal_analysis(pred_str, prey_str)
    
    results['DPE'] = {
        'direction': verdict_dpe.replace('X', 'Predator').replace('Y', 'Prey'),
        'Predator_to_Prey_Score': round(h_pred_prey, 4),
        'Prey_to_Predator_Score': round(h_prey_pred, 4),
        'Strength': round(abs(h_pred_prey - h_prey_pred), 4)
    }

    etc_res = pairs.CCM_causality(pred_etc, prey_etc)
    print(etc_res)

    model_mapping = {
        'ETC-P': 'ETCP',
        'ETC-E': 'ETCE',
        'LZ-P': 'LZP'
    }

    for model_label, prefix in model_mapping.items():
        # Map raw directions to Predator/Prey labels
        raw_dir = etc_res.get(f'{prefix}_direction')
        threshold = etc_res.get(f'{prefix}_threshold')
        if raw_dir == "x_causes_y": clean_dir = "Predator -> Prey"
        if raw_dir == "y_causes_x": clean_dir = "Prey -> Predator"
        if raw_dir == "n_or_m": clean_dir = "Independent"

        results[model_label] = {
            'direction': clean_dir,
            'Predator_to_Prey_Score': round(etc_res.get(f'{prefix}_x_to_y', 0), 4),
            'Prey_to_Predator_Score': round(etc_res.get(f'{prefix}_y_to_x', 0), 4),
            'Strength': round(etc_res.get(f'{prefix}_strength', 0), 4)
        }

    return results

data = extract_csv(FILE)
all_model_results = run_predator_prey_comparison(FILE)
plot_data(data)

print(f"{'Model':<10} | {'Direction':<20} | {'Strength':<10} | {'Pred->Prey':<10} | {'Prey->Pred':<10}")
print("-" * 75)
for model, data in all_model_results.items():
    print(f"{model:<10} | {data['direction']:<20} | {data['Strength']:<10} | {data['Predator_to_Prey_Score']:<10} | {data['Prey_to_Predator_Score']:<10}")

df_final = pd.DataFrame.from_dict(all_model_results, orient='index')
df_final.to_csv('results/predator_prey_multi_model.csv')
print(f'Table 6: results/predator_prey_multi_model.csv')

