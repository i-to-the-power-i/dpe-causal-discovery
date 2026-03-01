import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import array
from Bio import SeqIO
from model.cy_utils import run_causal_analysis
import ETCPy.ETC.CCMC.pairs as pairs
from scipy.stats import trim_mean

def dna_to_int_array(seq_str):
    mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    return np.array([mapping[base] for base in seq_str.upper() if base in mapping], dtype=np.uint8)

def run_genomic_experiment(df, rs_fasta_path, output_file='results/country_causal_results.csv'):
    rs_record = next(SeqIO.parse(rs_fasta_path, "fasta"))
    rs_int = dna_to_int_array(str(rs_record.seq))
    
    rs_etc_format = array.array('I', rs_int.tolist())

    results = []
    
    for country, group in df.groupby('Country'):
        print(f"Processing {country}...")
        
        cw_seq_str = group.iloc[0]['Sequence']
        cw_int = dna_to_int_array(cw_seq_str)
        cw_etc_format = array.array('I', cw_int.tolist())
        
        other_sequences = group.iloc[1:] 

        stats_rs = {'DPE': 0, 'ETC-P': 0, 'ETC-E': 0, 'LZ-P': 0}
        stats_cw = {'DPE': 0, 'ETC-P': 0, 'ETC-E': 0, 'LZ-P': 0}

        for _, row in group.iterrows():
            target_int = dna_to_int_array(row['Sequence'])
            target_etc = array.array('I', target_int.tolist())


            verdict_rs, _ = run_causal_analysis(target_int, rs_int)
            if verdict_rs == "Y -> X": stats_rs['DPE'] += 1
            
            res_rs = pairs.CCM_causality(target_etc, rs_etc_format)
            if res_rs.get('ETCP_direction') == "y_causes_x": stats_rs['ETC-P'] += 1
            if res_rs.get('ETCE_direction') == "y_causes_x": stats_rs['ETC-E'] += 1
            if res_rs.get('LZP_direction') == "y_causes_x": stats_rs['LZ-P'] += 1


            if not np.array_equal(target_int, cw_int):
                verdict_cw, _ = run_causal_analysis(target_int, cw_int)
                if verdict_cw == "Y -> X": stats_cw['DPE'] += 1
                
                res_cw = pairs.CCM_causality(target_etc, cw_etc_format)
                if res_cw.get('ETCP_direction') == "y_causes_x": stats_cw['ETC-P'] += 1
                if res_cw.get('ETCE_direction') == "y_causes_x": stats_cw['ETC-E'] += 1
                if res_cw.get('LZP_direction') == "y_causes_x": stats_cw['LZ-P'] += 1

        total = len(group)
        for model in stats_rs.keys():
            results.append({
                'Country': country,
                'Model': model,
                'RS_Causes_Prop': round(stats_rs[model] / total, 4),
                'CW_Causes_Prop': round(stats_cw[model] / (total - 1), 4) if total > 1 else 0
            })

    final_df = pd.DataFrame(results)
    final_df.to_csv(output_file, index=False)
    return final_df

def plot_barpattern_chart(file_path='results/country_causal_results.csv'):
    df = pd.read_csv(file_path)

    model_order = ['DPE', 'ETC-P', 'ETC-E', 'LZ-P']
    models = [m for m in model_order if m in df['Model'].unique()]
    countries = df['Country'].unique()

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(13, 10))

    fs = 20
    n_models = len(models)
    x = np.arange(len(countries))
    bar_width = 0.8 / n_models

    color_map = {
        'DPE': 'skyblue',
        'ETC-P': 'orange',
        'ETC-E': 'lightgreen',
        'LZ-P': 'red'
    }

    legend_handles = []

    ax = axes[0]

    for i, model in enumerate(models):
        model_df = df[df['Model'] == model]
        offset = (i - (n_models - 1)/2) * bar_width

        bars = ax.bar(
            x + offset,
            model_df['RS_Causes_Prop'],
            width=bar_width,
            color=color_map.get(model, 'gray'),
            edgecolor='black',
            label=model
        )

        # Save only one handle per model (from first subplot)
        legend_handles.append(bars[0])

    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.tick_params(axis='x', length=5, width=2)
    ax.tick_params(axis='y', labelsize=24)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Proportion', fontsize=fs, fontweight='bold')
    ax.set_title('RS Proportions', fontsize=fs+2, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    ax = axes[1]

    for i, model in enumerate(models):
        model_df = df[df['Model'] == model]
        offset = (i - (n_models - 1)/2) * bar_width

        ax.bar(
            x + offset,
            model_df['CW_Causes_Prop'],
            width=bar_width,
            color=color_map.get(model, 'gray'),
            edgecolor='black'
        )

    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=90, ha='right', fontsize=26)
    ax.set_xlabel('Country', fontsize=fs, fontweight='bold')
    ax.tick_params(axis='x', length=5, width=2)
    ax.tick_params(axis='y', labelsize=26)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Proportion', fontsize=fs, fontweight='bold')
    ax.set_title('CW Proportions', fontsize=fs+2, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Single clean legend (no duplicates)
    ax.legend(
        legend_handles,
        models,
        loc='upper right',
        bbox_to_anchor=(1.08, 0.99),
        ncol=1,
        fontsize=16,
        frameon=False
    )

    plt.tight_layout(h_pad=1)
    plt.subplots_adjust(top=0.92)
    plt.savefig('results/model_RS_CW_barplot.png', dpi=600)
    plt.show()

def compare_causal_models(file_path='results/country_causal_results.csv', threshold=0.05):
    df = pd.read_csv(file_path)
    total_countries = df['Country'].nunique()

    comparison = df.groupby('Model').apply(lambda x: pd.Series({
        'RS >= 5% (Count)': (x['RS_Causes_Prop'] >= threshold).sum(),
        'CW >= 5% (Count)': (x['CW_Causes_Prop'] >= threshold).sum(),
        'CW > RS (Count)': (x['CW_Causes_Prop'] > x['RS_Causes_Prop']).sum(),
        'Avg RS Prop': x['RS_Causes_Prop'].mean(),
        'Avg CW Prop': x['CW_Causes_Prop'].mean()
    })).reset_index()

    comparison['Sort_Order'] = comparison['Model'].apply(lambda x: 0 if x == 'DPE' else 1)
    comparison = comparison.sort_values(['Sort_Order', 'Model']).drop(columns='Sort_Order')

    print(f"Model Comparison Summary ({total_countries} Countries Total) ---")
    print(comparison.to_string(index=False))
    print("\nModel Performance vs Paper Hypotheses ---")
    
    for _, row in comparison.iterrows():
        print(f"\nModel: {row['Model']}")
        print(f"  - RS Direction Found: {int(row['RS >= 5% (Count)'])}/{total_countries} countries")
        print(f"  - CW Direction Found: {int(row['CW >= 5% (Count)'])}/{total_countries} countries")
        print(f"  - CW > RS Dominance:  {int(row['CW > RS (Count)'])}/{total_countries} countries")
        print(f"  - Avg Proportions:    RS: {row['Avg RS Prop']:.4f} | CW: {row['Avg CW Prop']:.4f}")

    return comparison


# It is advised to use biopython (Entrez) to extract these sequences
df = pd.read_csv('dataset/country-wise.csv')
results = run_genomic_experiment(df, 'dataset/sars-cov-2.fasta')

# # Plot barchart
plot_barpattern_chart('results/country_causal_results.csv')

# # Hypothesis 
compare_causal_models()