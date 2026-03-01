from model.utils import barchart_analysis, generate_causal_network, generate_coupled_data
from model.cy_utils import calculate_causal_history, generate_pattern_dictionary, calculate_contribution_analysis, identify_causality
from model.sequence_extractor import multi_sequence_extractor
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def weighted_binary_entropy(ratio, count, seq_len, pattern_len):
    # 1. Calculate standard binary entropy Hb(r)
    if ratio <= 0.0 or ratio >= 1.0:
        entropy = 0.0
    else:
        # Use np.log2 for compatibility, Cython handles this well
        entropy = -(ratio * np.log2(ratio) + (1.0 - ratio) * np.log2(1.0 - ratio))
    
    # 2. Calculate weight: Occurrences / Total possible windows
    total_windows = (seq_len - pattern_len + 1)
    
    if total_windows <= 0:
        return 0.0
        
    weight = count / total_windows
    
    # 3. Return weighted entropy
    # print(f'Ratio:{ratio}, Entropy:{weight * entropy}')
    return weight * entropy

def average_weighted_entropy(stats, seq_len):
    if stats is None or stats.empty:
        return 0.0
    
    weighted_entropies = stats.apply(
        lambda row: weighted_binary_entropy(
            row['Ratio'], 
            (row['Change'] + row['No-change']), 
            seq_len, 
            len(str(row.name).strip())
        ), 
        axis=1
    )
    # print(f'Average weighted entropy: {weighted_entropies.mean()}')
    return weighted_entropies

def draw_network(W_p_x, W_p_y, file_name='causal_network_dual'):
    G = nx.DiGraph()

    G.add_node("X", color='skyblue', size=7000)
    G.add_node("Y", color='#e67e22', size=7000)

    all_patterns = sorted(list(set(W_p_x.index) | set(W_p_y.index)))
    
    for pattern in all_patterns:
        pattern_str = str(pattern).strip()
        G.add_node(pattern_str, size=5000)
        
        if pattern in W_p_x.index:
            ent = W_p_x[pattern]
            G.add_edge(pattern_str, "X", entropy=ent, label=f"{ent:.2f}")
        if pattern in W_p_y.index:
            ent = W_p_y[pattern]
            G.add_edge(pattern_str, "Y", entropy=ent, label=f"{ent:.2f}")

    plt.figure(figsize=(10, 8))
    
    hub_x_pos = np.array([-7, 0])
    hub_y_pos = np.array([7, 0])
    pos = {"X": hub_x_pos, "Y": hub_y_pos}

    x_only, y_only, shared = [], [], []
    for node in all_patterns:
        successors = list(G.successors(node))
        if "X" in successors and "Y" in successors:
            shared.append(node)
        elif "X" in successors:
            x_only.append(node)
        else:
            y_only.append(node)

    x_only.sort(key=lambda n: G.edges[(n, "X")]['entropy'])
    y_only.sort(key=lambda n: G.edges[(n, "Y")]['entropy'])
    shared.sort(key=lambda n: (G.edges[(n, "X")]['entropy'] + G.edges[(n, "Y")]['entropy']))

    def position_nodes(nodes, hub_pos, side_mult, hub_key):
        if not nodes: return
        count = len(nodes)
        angles = np.linspace(-np.pi/2.2, np.pi/2.2, count) if count > 1 else [0]
        for i, node in enumerate(nodes):
            ent = G.edges[(node, hub_key)]['entropy']
            dist = (1 + ent) * 20
            pos[node] = hub_pos + np.array([side_mult * np.cos(angles[i]), np.sin(angles[i])]) * dist
            G.nodes[node]['color'] = '#aed6f1' if hub_key == "X" else '#f9e79f'

    position_nodes(x_only, hub_x_pos, -1, "X")
    position_nodes(y_only, hub_y_pos, 1, "Y")

    if shared:
        y_space = np.linspace(-20, 20, len(shared)) if len(shared) > 1 else [0]
        for i, node in enumerate(shared):
            pos[node] = np.array([0, y_space[i]])
            G.nodes[node]['color'] = '#9b59b6'

    nodes_list = G.nodes(data=True)
    node_sizes = [d.get('size', 2500) for n, d in nodes_list]
    
    nx.draw_networkx_nodes(G, pos, 
                           node_color=[d.get('color', '#cccccc') for n, d in nodes_list],
                           node_size=node_sizes,
                           edgecolors='black', linewidths=2)

    nx.draw_networkx_labels(G, pos, font_size=19, font_weight='bold')

    nx.draw_networkx_edges(
        G, pos,
        arrowstyle='-|>', 
        arrowsize=40, 
        width=2.0, 
        edge_color='black', 
        alpha=0.6,
        node_size=node_sizes, 
        min_source_margin=15, 
        min_target_margin=25  
    )

    edge_labels = nx.get_edge_attributes(G, 'label')
    # label_pos=0.5 is the exact center of the edge
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                 font_size=24, font_color='red', 
                                 label_pos=0.5, rotate=True)

    # plt.title(r'$Distance = 1 + Entropy$', fontsize=22, fontweight='bold', pad=30)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'results/{file_name}.png')
    plt.show()



# =============================
# for testing
# =============================
# # example given in the paper

x = '011101111010011001110101101001'
y = '000001000010000000000100001000'

# y = input('Enter sequence Y\n')
# x = input('Enter sequence X\n')

# causality evaluation
Gx_y = calculate_causal_history(y, x)
Gy_x = calculate_causal_history(x, y)
# direction = causality_inference(Gx_y, Gy_x)

print(f'''Final Dictionaries
    Gx->y: {Gx_y}
    Gy->x: {Gy_x}
''')

# finding patterns
Px_y = generate_pattern_dictionary(Gx_y)
Py_x = generate_pattern_dictionary(Gy_x)

print("\nFinal Pattern Dictionary PX->Y:")
print(sorted(list(Px_y), key=len))
print("\nFinal Pattern Dictionary PY->X:")
print(sorted(list(Py_x), key=len))

# contribution analysis
print('\nContribution analysis')
stats1 = calculate_contribution_analysis(Px_y, x, y)
print(stats1)
stats2 = calculate_contribution_analysis(Py_x, y, x)
print(stats2)

# plotting bar chart
# barchart_analysis(stats1, 'causal_analysis_chart_xy')
# barchart_analysis(stats2, 'causal_analysis_chart_yx')


# constructing causal network
# generate_causal_network(stats1, 'causal_network_graph_xy', 'Y')
# generate_causal_network(stats2, 'causal_network_graph_yx', 'X')

# Weighted Entropy for each pattern.
len_x, len_y = len(x), len(y)
W_p_x = average_weighted_entropy(stats1, len_x)
W_p_y = average_weighted_entropy(stats2, len_y)
print(W_p_x, W_p_y)

# finding causality
verdict, Hxy, Hyx = identify_causality(stats1, stats2, x, y)
print(f"\nFINAL VERDICT: {verdict}")
print(r'Havg_xy: ', Hxy)
print(r'Havg_yx: ', Hyx)

draw_network(W_p_x, W_p_y)

# =============================
# Let's try with the sars-cov dna sample
# =============================

# print('\n\nsars cov1 & 2 example')
# cov_1 = """
# 010110221111103310333022000023300330033132013131121020132113113131000320031110000131212102312132313223123012331021230331032302101001100100310011031213211203022030320210031321310131131230223123110322111321332121123023320130130230301310221113213322212120332000221002012202023311213331211113003202000030303213300313021112331211110302113202002330102202301013112323111103021132020023301022023010131123231111030211320200233010220230101311232311110302113202002330102202301013112323111103021132020023301022023010131123231111030211320200233010220230101311232311110302113202002330102202301013112323111103021132020023301022023010131123231111030211320200233010220230101311232311110302113202002330102202301013112323111103021132020023301022023010131123231111030211320200233010220230101311232311110302113202002330102202301013112323111103021132020023301022023010131123231111030211320200233010220230101311232311110302113202002330102202301013112323111103021132020023301022023010131123231111030211320200233010220230101311232311110302113202002330102202301013112323111103021132020023301022023010131123231111030211320200233010220230101311232311110302113202002330102202301013112323111103021132020023301022023010131123231111030211320200233010220230101311232311110302113202002330102202301013112323111103021132020023301022023010131123231111030211320200233010220230101311232311110302
# """
# cov_2 = """
# 011000221110103311333022100300033003300311132013131121020131211311313100032003111000013121212231213031322312301231102123031303230210100110010031001103121321120302203032021003132131013113123022312311032211132133212112302332013013023030131022111321332221212033200022100201220202331121333121111300320200003030321330031302111233121111030211320200233010220230101311232311110302113202002330102202301013112323111103021132020023301022023010131123231111030211320200233010220230101311232311110302113202002330102202301013112323111103021132020023301022023010131123231111030211320200233010220230101311232311110302113202002330102202301013112323111103021132020023301022023010131123231111030211320200233010220230101311232311110302113202002330102202301013112323111103021132020023301022023010131123231111030211320200233010220230101311232311110302113202002330102202301013112323111103021132020023301022023010131123231111030211320200233010220230101311232311110302113202002330102202301013112323111103021132020023301022023010131123231111030211320200233010220230101311232311110302113202002330102202301013112323111103021132020023301022023010131123231111030211320200233010220230101311232311110302113202002330102202301013112323111103021132020023301022023010131123231111030211320200233010220230101311232311110302113202002330102202301013112323111103021132020023301022023010131123231111
# """
# print(len(cov_1), len(cov_2))
# # causality evaluation
# Gx_y = calculate_causal_history(cov_2, cov_1)
# Gy_x = calculate_causal_history(cov_1, cov_2)
# # direction = causality_inference(Gx_y, Gy_x)

# print(f'''Final Dictionaries
#     Gx->y: {Gx_y}
#     Gy->x: {Gy_x}
# ''')

# # finding patterns
# Px_y = generate_pattern_dictionary(Gx_y)
# Py_x = generate_pattern_dictionary(Gy_x)

# print("\nFinal Pattern Dictionary PX->Y:")
# print(sorted(list(Px_y), key=len))
# print("\nFinal Pattern Dictionary PY->X:")
# print(sorted(list(Py_x), key=len))

# # contribution analysis
# print('\nContribution analysis')
# stats1 = calculate_contribution_analysis(Px_y, cov_1, cov_2)
# print(stats1)
# stats2 = calculate_contribution_analysis(Py_x, cov_2, cov_1)
# print(stats2)

# plotting bar chart
# barchart_analysis(stats1, 'causal_analysis_chart_xy')
# barchart_analysis(stats2, 'causal_analysis_chart_yx')


# # constructing causal network
# generate_causal_network(stats1, 'causal_network_graph_xy', 'Y')
# generate_causal_network(stats2, 'causal_network_graph_yx', 'X')

# # finding causality
# verdict = identify_causality(stats1, stats2, cov_1, cov_2)
# print(f"\nFINAL VERDICT: {verdict}")

# # testing the functions

# # single sequence extractor
# sequence = single_sequence_extractor("NC_045512.2")
# print(sequence[:10])

# =============================
# Sequence extractor toolkit
# =============================
# # multiple sequence extractor
# id_list = ['PX583645', 'PX575891', 'PV998713', 'PV998714', 'PV998716', 'PV998717', 'PV714139', 'MZ317934']
# df = multi_sequence_extractor(id_list)
# print(df)

# custom_init = {
#     "0.3": {"M": 0.5, "S": 0.2}
# }

# # Run the generator
# generate_coupled_data(initial_values=custom_init)