import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import itertools
import os
import re
import json

def ensure_numpy(seq):
    """Internal helper to convert strings, lists, or arrays to a NumPy array."""
    if isinstance(seq, str):
        return np.array(list(seq))
    return np.asarray(seq)

def calculate_causal_history(seq_1, seq_2):
    '''
    Constructs the dictionary GX→Y by scanning Y from left to right and recording 
    the corresponding sub-pattern in X each time a bit flip occurs in Y.
    
    Args:
        seq_1 (str/list/np.ndarray): Sequence_Y (Target)
        seq_2 (str/list/np.ndarray): Sequence_X (Source)
    
    Returns:
        set: A set of unique segments from Sequence_X associated with flips in Y.
    '''
    # Convert both to numpy arrays for unified indexing
    s1 = ensure_numpy(seq_1)
    s2 = ensure_numpy(seq_2)

    len_1 = len(s1)
    len_2 = len(s2)

    if len_1 != len_2:
        raise ValueError(f'Sequences must be of equal length. Got {len_1} and {len_2}.')
    
    G = []
    last_position = 0
    k = 1
    
    # Identify bit flips: s1[k] != s1[k-1]
    while k < len_1:
        if s1[k] != s1[k-1]:
            segment = s2[last_position: k+1]
            G.append("".join(segment.astype(str)))
            last_position = k + 1
            k += 1
        k += 1
    
    return set(G)

def sliding_xnor_comparison(p1, p2):
    '''
    Identifies common subsequences between two patterns using XNOR-based sliding.
    Operates on NumPy arrays for bitwise comparison.

    Args:
        p1 (str/list/np.ndarray): The first binary subpattern.
        p2 (str/list/np.ndarray): The second binary subpattern.

    Returns:
        set: Common subsequences (length >= 2).
    '''
    arr1 = ensure_numpy(p1)
    arr2 = ensure_numpy(p2)
    
    n1, n2 = len(arr1), len(arr2)

    if n1 > n2:
        arr1, arr2 = arr2, arr1
        n1, n2 = n2, n1

    found_in_pair = set()

    for shift in range(n2 - n1 + 1):
        # numpy vectorized comparison similar to binary.
        matches = (arr1 == arr2[shift : shift + n1])

        current_match = []
        for i, is_match in enumerate(matches):
            if is_match:
                current_match.append(str(arr2[shift + i]))
            else:
                if len(current_match) >= 2:
                    found_in_pair.add("".join(current_match))
                current_match = []
        
        if len(current_match) >= 2:
            found_in_pair.add("".join(current_match))
    
    return found_in_pair

def generate_pattern_dictionary(G):
    '''
    Constructs the overall pattern dictionary PX->Y from a causal history.
    Iterates through all distinct pairs in G and extracts common sub-patterns.

    Args:
        G (set/list): The causal history dictionary GX->Y.

    Returns:
        set: The final PX->Y pattern dictionary.
    '''
    pattern = set()
    pairs = list(itertools.combinations(G, 2))

    for p1, p2 in pairs:
        extracted = sliding_xnor_comparison(p1, p2)
        pattern.update(extracted)

    # if there is no pattern in patterns set.
    if not pattern:
        return G

    return pattern

def calculate_contribution_analysis(patterns, X, Y):
    '''
    Performs frequency and contribution analysis for extracted causal patterns.
    Quantifies the likelihood that a pattern in X induces a variation in Y.

    Args:
        patterns (set/list): Dictionary of extracted patterns (PX->Y).
        X (str/list/np.ndarray): The causal input sequence.
        Y (str/list/np.ndarray): The output sequence containing bit flips.

    Returns:
        pd.DataFrame: Mapping of patterns to Change, No-change, and Ratio (R_flip).
    '''
    # ensuring both X and Y are of particular dtype
    x_str = "".join(ensure_numpy(X).astype(str))
    y_arr = ensure_numpy(Y)
    
    results = {}

    for pattern in patterns:
        p_str = str(pattern)
        pattern_len = len(p_str)
        
        # find overlapping occurrences in X
        matches = [m.start() for m in re.finditer(f'(?={re.escape(p_str)})', x_str)]
        
        change_cnt = 0
        nochange_cnt = 0
        
        for start_idx in matches:
            has_flip = False
            # check internal transition points for flips in Y
            for offset in range(1, pattern_len):
                k = start_idx + offset
                if k < len(y_arr):
                    if y_arr[k] != y_arr[k-1]:
                        has_flip = True
                        break 
            
            if has_flip:
                change_cnt += 1
            else:
                nochange_cnt += 1

        total = len(matches)
        ratio = change_cnt / total if total > 0 else 0

        results[p_str] = {
            'Change': change_cnt,
            'No-change': nochange_cnt,
            'Ratio': round(ratio, 4)
        }

    df = pd.DataFrame(results).T
    df.index.name = 'Pattern'
    return df

def barchart_analysis(stats, file_name='causal_analysis_chart', direction='x->y'):
    '''
    Generates and saves a bar chart visualizing the contribution ratio (R_flip) 
    for each extracted pattern.

    This method creates a high-resolution visualization of the predictive power 
    of patterns in the PX->Y set. It plots the R_flip ratio on the Y-axis against 
    the binary patterns on the X-axis, allowing for easy identification of 
    perfect causal predictors (where Ratio = 1.0).

    The function automatically handles directory creation, saves the output 
    as a high-DPI PNG file in the 'results' folder, and applies a 'tight' 
    bounding box to ensure labels are not clipped.

    Args:
        stats (pd.DataFrame): A transposed DataFrame containing binary patterns 
            as indices and their corresponding 'Ratio' (R_flip) as a column.
        file_name (str): File name for saving the file.
        direction (str): Causal direction to be considered for barchart analysis.

    Returns:
        None: Displays the plot and saves the file to 'results/causal_analysis_chart.png'.
    '''
    if stats.empty or 'Ratio' not in stats.columns:
        print(f"--- Skipping plot for {direction}: No patterns detected ---")
        return

    fig, ax = plt.subplots(figsize=(15, 6))
    patterns = stats.index 
    indices = np.arange(len(patterns))
    width = 0.6

    ax.bar(indices, stats['Ratio'], width, color='#2ecc71', edgecolor='black')
    
    for i, v in enumerate(stats['Ratio']):
        ax.text(i, v + 0.02, str(v), ha='center', fontweight='bold', fontsize=15)

    ax.set_xlabel(f'Binary Patterns ($P_{direction}$)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Ratio Score ($R_{flip}$)', fontsize=18, fontweight='bold')
    ax.set_xticks(indices)
    ax.set_xticklabels(patterns, rotation=45)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylim(0, 1.1)

    # Apply bold to x-axis tick labels
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    # Apply bold to y-axis tick labels
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    if not os.path.exists('results'):
        os.makedirs('results')

    plt.savefig(f'results/{file_name}.png', bbox_inches='tight', dpi=300)
    plt.show()

def generate_causal_network(stats, file_name='causal_network_graph', target_node='Y'):
    '''
    Constructs an aesthetic radial network graph representing causal proximity.

    This function visualizes the relationship between extracted patterns (PX->Y) 
    and the target variable Y. It utilizes a polar coordinate system where the 
    radial distance from the center node Y is strictly determined by the 
    predictive reliability of the pattern.

    The distance (d) is calculated using the linear transformation:
    d = 10 * (2 - Ratio)

    Under this logic, 'Perfect Predictors' (Ratio = 1.0) are positioned at a 
    distance of 1.0 units from Y, while non-causal patterns (Ratio = 0.0) 
    are placed at a maximum distance of 11.0 units. Nodes are distributed 
    equidistantly in a circular arrangement to prevent label overlap and 
    ensure visual clarity.

    Args:
        stats (pd.DataFrame): A transposed DataFrame where the index contains 
            binary patterns and the 'Ratio' column contains the R_flip values.
        file_name (str): File name for saving the file.
        target_node (str): Target node for network.

    Returns:
        None: Generates a Matplotlib figure, saves it to 'results/causal_network_aesthetic.png', 
            and displays the output.
    '''
    if stats.empty or 'Ratio' not in stats.columns:
        print(f"--- Skipping plot for {target_node}: No patterns detected ---")
        return

    G = nx.DiGraph()
    
    stats_sorted = stats.sort_values(by='Ratio', ascending=False)
    
    patterns = [str(p) for p in stats_sorted.index]
    ratios = stats_sorted['Ratio'].tolist()
    
    pos = {target_node: np.array([0, 0])}
    
    angle_step = 2 * np.pi / len(patterns)
    
    for i, pattern in enumerate(patterns):
        ratio = ratios[i]
        distance = 10 * (2 - ratio)
        
        angle = i * angle_step
        pos[pattern] = np.array([distance * np.cos(angle), distance * np.sin(angle)])
        
        G.add_node(pattern, ratio=ratio)
        G.add_edge(pattern, target_node, ratio=ratio)

    plt.figure(figsize=(12, 12))
    
    nx.draw_networkx_nodes(G, pos, nodelist=[target_node], 
                           node_color='skyblue', node_size=3000, 
                           edgecolors='black', linewidths=2)
    
    nx.draw_networkx_nodes(G, pos, nodelist=patterns, 
                           node_color='#ff9933', node_size=2000, 
                           edgecolors='black', linewidths=1.5)

    for u, v, d in G.edges(data=True):
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            arrowstyle='-|>', 
            arrowsize=30, 
            width=1.5 + (d['ratio'] * 3),
            edge_color='gray',
            alpha=0.6,
            node_size=3000
        )

    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold')
    
    edge_labels = {(u, v): f"{G[u][v]['ratio']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                 font_size=14, label_pos=0.5, font_color='red')

    plt.title('Causal Proximity Network\n' + r'$Distance = 10 \cdot (2 - R_{flip})$', 
              fontsize=18, fontweight='bold', pad=20)
    plt.text(0.95, 0.01, f"Nodes closer to {target_node} = Stronger Causality", 
             transform=plt.gcf().transFigure, ha='right', fontsize=15, style='italic')

    plt.axis('off')
    
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(f'results/{file_name}.png', bbox_inches='tight', dpi=300)
    plt.show()

# # ===================================================
# # old method used
# # ===================================================

# def get_direction_metrics(df):
#     '''
#     Extracts key causal descriptors (N1, L1, R_bar) from a pattern contribution dictionary.

#     This function reduces the high-dimensional pattern-match statistics into three 
#     fundamental quantities required for causal direction identification. These metrics 
#     capture both the deterministic strength and the global statistical tendency of 
#     a specific causal hypothesis (e.g., the direction X -> Y).

#     The quantities computed are:
#     1. N(1) [Deterministic Count]: The total count of 'Perfect Predictors'—patterns 
#        that possess a Ratio of exactly 1.0, indicating they are always followed by 
#        a bit flip in the target sequence.
#     2. L(1) [Maximum Complexity]: The maximum bit-length among all Perfect Predictors. 
#        This quantifies the deepest level of deterministic dependency detected 
#        within the sequence.
#     3. R_bar [Mean Tendency]: The arithmetic mean of the 'Ratio' column across all 
#        patterns in the dictionary, serving as a measure of the global causal 
#        influence strength.

#     Args:
#         df (pd.DataFrame): A DataFrame where the index contains unique binary 
#             patterns and the 'Ratio' column contains the computed R_flip values.

#     Returns:
#         tuple: A triplet of metrics (n_1, l_1, r_bar) where:
#             - n_1 (int): Count of patterns with a perfect ratio.
#             - l_1 (int): Length of the longest perfect pattern.
#             - r_bar (float): The average contribution ratio of all patterns.
#     '''
#     if df.empty:
#         return 0, 0, 0
    
#     df_1 = df[df['Ratio'] == 1.0]
#     n_1 = len(df_1)
#     l_1 = df_1.index.map(len).max() if n_1 > 0 else 0
#     r_bar = df['Ratio'].mean() if not df.empty else 0.0
    
#     return n_1, l_1, r_bar

# def identify_causality(df_xy, df_yx, tau=0.05):
#     '''
#     Identifies the dominant causal direction between two sequences using hierarchical inference rules.

#     This function implements a multi-tiered decision logic to determine whether X causes Y, 
#     Y causes X, or if the relationship is independent/undetermined. The logic prioritizes 
#     deterministic evidence—patterns that perfectly predict a state change—before falling 
#     back to statistical averages.

#     The identification follows a strict hierarchy of rules:
    
#     1. Independence Rule: If the absolute difference between mean ratios (|R̄x->y - R̄y->x|) 
#        is less than the threshold tau, the sequences are considered independent.
#     2. Primary Rule (Deterministic Strength): The direction with the higher count of 
#        'Perfect Predictors' (N_1, where Ratio == 1.0) is identified as causal.
#     3. Secondary Rule (Pattern Complexity): If N_1 is equal, the direction with the 
#        longest perfect pattern (L_1) is identified as causal, as longer patterns 
#        represent more complex deterministic dependencies.
#     4. Tertiary Rule (Mean Tendency): If both N_1 and L_1 are equal, the direction 
#        with the higher overall average ratio (R̄) is selected.
#     5. Tie Case: If all metrics are identical, the direction is marked 'Undetermined'.

#     Args:
#         df_xy (pd.DataFrame): Contribution analysis results for the X -> Y direction.
#         df_yx (pd.DataFrame): Contribution analysis results for the Y -> X direction.
#         tau (float, optional): Threshold for the independence rule. Represents the 
#             minimum significant difference in mean ratios. Defaults to 0.05.

#     Returns:
#         str: A string indicating the detected causal relationship: 
#              "X causes Y", "Y causes X
#     '''
#     nx_y, lx_y, rx_y = get_direction_metrics(df_xy)
#     ny_x, ly_x, ry_x = get_direction_metrics(df_yx)
    
#     if abs(rx_y - ry_x) < tau:
#         return "Independence"
#     if nx_y > ny_x: return "X -> Y"
#     elif ny_x > nx_y: return "Y -> X"
#     if lx_y > ly_x: return "X -> Y"
#     elif ly_x > lx_y: return "Y -> X"
#     if rx_y > ry_x: return "X -> Y"
#     elif ry_x > rx_y: return "Y -> X"

#     return "Undetermined"
# # ===================================================


# # ===================================================
# # New updated method: Entropy based
# # ===================================================

# def binary_entropy(ratio):
#     '''
#     Calculates the Shannon binary entropy for a given success probability.
    
#     The entropy measures the uncertainty associated with a binary random variable 
#     (bit flip vs. no flip). A ratio of 0.0 or 1.0 represents perfect certainty 
#     (0 bits of entropy), while a ratio of 0.5 represents maximum uncertainty 
#     (1 bit of entropy).

#     Args:
#         ratio (float): The R_flip value (probability of a state transition).

#     Returns:
#         float: The calculated entropy in bits.
#     '''
#     if ratio <= 0 or ratio >= 1:
#         return 0.0
#     return -(ratio * np.log2(ratio) + (1 - ratio) * np.log2(1 - ratio))

# def average_entropy(stats):
#     '''
#     Computes the mean binary entropy across a dictionary of causal patterns.
    
#     This serves as an aggregate measure of the "predictive noise" in a 
#     given causal direction. A lower average entropy suggests that the 
#     patterns in the source sequence are more reliable predictors of 
#     state changes in the target sequence.

#     Args:
#         stats (pd.DataFrame): DataFrame containing pattern statistics 
#             with a 'Ratio' column.

#     Returns:
#         float: The arithmetic mean of entropy values for all patterns.
#     '''
#     if stats is None or stats.empty:
#         return 0.0
    
#     entropies = stats['Ratio'].apply(binary_entropy)
#     return float(entropies.mean())

# def identify_causality(stats_x_y, stats_y_x):
#     '''
#     Infers the causal direction by comparing the average entropy of 
#     pattern-response relationships.
    
#     The method follows the principle of Minimum Uncertainty: the direction 
#     with the lower average entropy is identified as the causal path, 
#     as it indicates a more deterministic relationship between the 
#     sequences. If the average entropies are equal, the sequences are 
#     considered independent or the direction is inconclusive.

#     Args:
#         stats_x_y (pd.DataFrame): Pattern statistics for the X -> Y direction.
#         stats_y_x (pd.DataFrame): Pattern statistics for the Y -> X direction.

#     Returns:
#         str: Inferred direction ('X -> Y', 'Y -> X', 'Independence', or 'Undetermined).
#     '''
#     avg_h_xy = average_entropy(stats_x_y)
#     avg_h_yx = average_entropy(stats_y_x)

#     # print(f'''
#     # Average Entropy from X to Y: {avg_h_xy}
#     # Average Entropy from Y to X: {avg_h_yx}
#     #       ''')
    
#     if avg_h_xy < avg_h_yx:
#         return 'X -> Y'
#     elif avg_h_yx < avg_h_xy:
#         return 'Y -> X'
#     elif avg_h_xy == avg_h_yx:
#         return 'Independence'
    
#     return 'Undetermined'
# # ===================================================


# # ===================================================
# # New updated method: Weighted Entropy based
# # ===================================================
def weighted_binary_entropy(ratio, count, seq_len, pattern_len):
    """
    Calculates the binary entropy of a pattern weighted by its frequency.
    
    Args:
        ratio (float): The success probability (R_flip).
        count (int): Total occurrences of this pattern (change + no change).
        seq_len (int): Total length of the sequence.
        pattern_len (int): The length of the specific pattern.
        
    Returns:
        float: The frequency-weighted entropy in bits.
    """
    # 1. Calculate standard binary entropy Hb(r)
    if ratio <= 0 or ratio >= 1:
        entropy = 0.0
    else:
        entropy = -(ratio * np.log2(ratio) + (1 - ratio) * np.log2(1 - ratio))
    
    # 2. Calculate weight: Occurrences / Total possible windows
    total_windows = seq_len - pattern_len + 1
    weight = count / total_windows
    
    # 3. Return weighted entropy
    return weight * entropy

def average_weighted_entropy(stats, seq_len):
    """
    Computes the mean weighted binary entropy across a pattern dictionary.
    
    Args:
        stats (pd.DataFrame): DataFrame with columns ['Pattern', 'Ratio', 'Count'].
            'Count' should be the total occurrences of that pattern.
        seq_len (int): The length of the original sequence used.

    Returns:
        float: The arithmetic mean of the weighted entropy values.
    """
    if stats is None or stats.empty:
        return 0.0
    
    # Calculate weighted entropy for each row
    # We use a lambda to pass the constant sequence length and variable pattern lengths
    weighted_entropies = stats.apply(
        lambda row: weighted_binary_entropy(
            row['Ratio'], 
            row['Change'] + row['No-change'], 
            seq_len, 
            len(str(row.name).strip())
        ), 
        axis=1
    )
    
    return weighted_entropies.mean()

def identify_causality(stats_x_y, stats_y_x, X, Y):
    '''
    Infers the causal direction by comparing the average entropy of 
    pattern-response relationships.
    
    The method follows the principle of Minimum Uncertainty: the direction 
    with the lower average weighted entropy is identified as the causal path, 
    as it indicates a more deterministic relationship between the 
    sequences. If the average weighted entropies are equal, the sequences are 
    considered independent or the direction is inconclusive.

    Args:
        stats_x_y (pd.DataFrame): Pattern statistics for the X -> Y direction.
        stats_y_x (pd.DataFrame): Pattern statistics for the Y -> X direction.

    Returns:
        str: Inferred direction ('X -> Y', 'Y -> X', 'Independence', or 'Undetermined).
    '''
    avg_h_xy = average_weighted_entropy(stats_x_y, len(X))
    avg_h_yx = average_weighted_entropy(stats_y_x, len(Y))

    # print(f'''
    # Average Entropy from X to Y: {avg_h_xy}
    # Average Entropy from Y to X: {avg_h_yx}
    #       ''')
    
    if avg_h_xy < avg_h_yx:
        return 'X -> Y', abs(avg_h_xy-avg_h_yx)
    elif avg_h_yx < avg_h_xy:
        return 'Y -> X', abs(avg_h_xy-avg_h_yx)
    elif avg_h_xy == avg_h_yx:
        return 'Independence', 0.0
    
    return 'Undetermined', abs(avg_h_xy-avg_h_yx)

def run_causal_analysis(seq_x_bin: np.ndarray, seq_y_bin:np.ndarray):
    '''
    Executes the full causal discovery pipeline on a single pair of binary sequences.
    
    This function converts binary arrays to strings, calculates causal history 
    metrics (likelihood of one sequence given the other's history), generates 
    pattern dictionaries for compression/contribution analysis, and provides 
     a final directionality verdict.

    Args:
        seq_x_bin (numpy.ndarray): Binary array representing the first time series (X).
        seq_y_bin (numpy.ndarray): Binary array representing the second time series (Y).

    Returns:
        str: A string indicating the causal verdict (e.g., "Y -> X", "X -> Y", 
             "Independence", or "Undecided").
    '''

    # 1. Calculate Causal History (Converting array to string if needed)
    x_str = "".join(map(str, seq_x_bin))
    y_str = "".join(map(str, seq_y_bin))
    
    Gx_y = calculate_causal_history(y_str, x_str)
    Gy_x = calculate_causal_history(x_str, y_str)
    
    # 2. Generate Pattern Dictionaries
    Px_y = generate_pattern_dictionary(Gx_y)
    Py_x = generate_pattern_dictionary(Gy_x)
    
    # 3. Contribution Analysis
    stats1 = calculate_contribution_analysis(Px_y, x_str, y_str)
    stats2 = calculate_contribution_analysis(Py_x, y_str, x_str)
    
    # 4. Final Verdict
    return identify_causality(stats1, stats2, x_str, y_str)

# # ===================================================


# For synthetic coupling experiment
def discrete(sequence: np.array, n_bins: int):
    '''
    Discretizes a continuous numerical signal into a symbolic integer sequence.

    This function performs uniform binning by dividing the data range into 
    n_bins of equal width. This step is critical for causal pattern 
    extraction, as it transforms floating-point time series into a format 
    compatible with dictionary-based symbolic analysis.

    Args:
        sequence (np.array): The input continuous time-series data.
        n_bins (int): The number of discrete levels (alphabet size) to create.

    Returns:
        np.array: A sequence of integers ranging from 0 to (n_bins - 1).
    '''
    if sequence.size == 0 or n_bins <= 0:
      return np.empty(0)

    minn = np.min(sequence)
    maxx = np.max(sequence)
    data_range = maxx - minn

    width = data_range/n_bins
    bin_sequence = []

    for value in sequence:
      distance_from_min = value - minn
      bin_index = int(distance_from_min / width)

      if bin_index >= n_bins:
        bin_index = n_bins - 1

      bin_sequence.append(bin_index)

    return np.array(bin_sequence)

def generator(length:int=1000, noise_intensity:float=0.01, a:float=0.8, b:float=0.8, phi:float=0):
    '''
    Generates synthetic bivariate time series using a coupled Autoregressive (AR) model.

    The model follows the structure:
    X[t] = a*X[t-1] + phi*Y[t-1] + noise
    Y[t] = b*Y[t-1] + noise

    When phi > 0, Y exerts a causal influence on X. This generator is used to 
    produce ground-truth data for validating causal discovery algorithms.

    

    Args:
        length (int): The number of time steps to generate.
        noise_intensity (float): Scaling factor for the Gaussian noise (e_x, e_y).
        a (float): Self-dependence coefficient for sequence X.
        b (float): Self-dependence coefficient for sequence Y.
        phi (float): Coupling strength representing the causal influence of Y on X.

    Returns:
        tuple: A pair of numpy arrays (X, Y) representing the coupled time series.
    '''
    X = np.zeros(length)
    Y = np.zeros(length)

    e_x = noise_intensity * np.random.normal(0, 1, length)
    e_y = noise_intensity * np.random.normal(0, 1, length)

    for t in range(1, length):
        X[t] = (a * X[t-1]) + (phi * Y[t-1]) + e_x[t]
        Y[t] = (b * Y[t-1]) + e_y[t]

    return X, Y

def coupling_experiment(num_sequences: int = 1000, sequence_length: int = 1000):
    '''
    Executes a large-scale simulation to analyze causal detection sensitivity.

    This method sweeps through various coupling strengths (phi) from 0 to 1. 
    For each phi, it generates multiple pairs of sequences, discretizes them 
    into binary signals, and organizes them for ensemble analysis. This is 
    designed to evaluate at which threshold of phi the causal method accurately 
    identifies 'Y -> X'.

    

    Args:
        num_sequences (int): Number of independent realizations to generate per phi value.
        sequence_length (int): The number of data points in each individual sequence.

    Returns:
        dict: A nested dictionary where keys are phi values and values are 
            dictionaries containing the discretized 'X' and 'Y' sequence matrices.
    '''
    phis = np.arange(0, 1, 0.05)
    result_dict = {}

    for phi in phis:
        X_sequences = np.empty((num_sequences, sequence_length), dtype=int)
        Y_sequences = np.empty((num_sequences, sequence_length), dtype=int)

        for i in range(num_sequences):
            x_raw, y_raw = generator(length=sequence_length, phi=phi)

            x_discrete = discrete(x_raw, 2)
            y_discrete = discrete(y_raw, 2)

            X_sequences[i, :] = x_discrete
            Y_sequences[i, :] = y_discrete

        result_dict[phi] = {
            'X': X_sequences,
            'Y': Y_sequences
        }

    return result_dict

# skew tent maps experiment
def skew_tent_map(x, b):
    '''
    Computes the next iteration of a 1D Skew-Tent Map.
    
    The Skew-Tent map is a piecewise linear chaotic map defined by the 
    skewness parameter 'b'. It maps the unit interval [0, 1] onto itself.
    
    Args:
        x (float): The current state value in the interval [0, 1].
        b (float): The skewness parameter (peak position) in the interval (0, 1).
        
    Returns:
        float: The mapped value T(x).
    '''
    if 0 <= x < b:
        return x / b
    elif b <= x < 1:
        return (1 - x) / (1 - b)
    else:
        return 0 

def generate_coupled_data(b1=0.65, b2=0.47, initial_values=None):
    '''
    Generates time series data for a Master-Slave system of coupled skew-tent maps.
    
    This function simulates a unidirectional coupling where the Master system (Y) 
    evolves independently, and the Slave system (X) is influenced by the Master 
    based on a coupling coefficient 'eta'. Data is generated for multiple 
    coupling strengths, with initial transients removed to ensure the system 
    is on a chaotic attractor.
    
    The simulation follows these governing equations:
    M(t) = T1(M(t-1))
    S(t) = (1 - eta) * T2(S(t-1)) + eta * M(t-1)
    
    Args:
        b1 (float): Skewness parameter for the Master map (T1). Defaults to 0.65.
        b2 (float): Skewness parameter for the Slave map (T2). Defaults to 0.47.
        initial_values (dict, optional): Manual starting points for specific 
            coupling strengths. Format: {"0.1": {"M": val, "S": val}}. 
            If None, values are chosen randomly from U(0, 1).
            
    Returns:
        None: Saves a nested dictionary to 'dataset/coupled_map_data.json' 
            containing 'X' (Slave), 'Y' (Master), and 'initial_conditions'.
            
    Note:
        - Number of trials per eta: 1500
        - Time series length: 1000 (after discarding 500 transient steps).
        - Coupling (eta) range: [0.0, 0.1, ..., 0.9].
    '''
    coupling_coefficients = np.arange(0, 1.0, 0.1)
    num_trials = 1500
    sequence_length = 1000
    transient_cutoff = 500
    total_steps = sequence_length + transient_cutoff

    results = {}

    print(f"{'Coupling (η)':<15} | {'Initial M':<15} | {'Initial S':<15}")

    for eta in coupling_coefficients:
        eta_key = f"{eta:.1f}"
        results[eta_key] = {"X": [], "Y": []}
        
        for trial in range(num_trials):
            if initial_values and eta_key in initial_values:
                m_val = initial_values[eta_key]['M']
                s_val = initial_values[eta_key]['S']
            else:
                m_val = np.random.uniform(0, 1)
                s_val = np.random.uniform(0, 1)

            if trial == 0:
                print(f"{eta_key:<15} | {m_val:<15.6f} | {s_val:<15.6f}")

            m_series = []
            s_series = []

            for t in range(total_steps):
                m_next = skew_tent_map(m_val, b1)
                s_next = (1 - eta) * skew_tent_map(s_val, b2) + eta * m_val
                
                m_val = m_next
                s_val = s_next

                if t >= transient_cutoff:
                    m_series.append(m_val)
                    s_series.append(s_val)

            results[eta_key]["Y"].append(m_series)
            results[eta_key]["X"].append(s_series)

    # Save to JSON
    output_path = 'dataset/coupled_map_data.json'
    with open(output_path, 'w') as f:
        json.dump(results, f)
    
    print(f"\nData generation complete. Saved to: {output_path}")

# # Data extraction
def extract_json_data(file_path, as_numpy=True):
    '''
    Loads and parses the coupled skew-tent map dataset from a JSON file.

    This function reads the nested dictionary structure generated by the 
    coupling simulation. It can optionally convert the time-series lists 
    into NumPy arrays to facilitate faster numerical analysis and vector 
    operations.

    Args:
        file_path (str): The system path to the 'coupled_map_data.json' file.
        as_numpy (bool, optional): If True, converts the 'X' (Slave) and 
            'Y' (Master) lists into NumPy float64 arrays of shape 
            (num_trials, sequence_length). Defaults to True.

    Returns:
        dict: A dictionary where keys are string representations of coupling 
            coefficients (e.g., "0.1"). Each value is a dictionary containing:
            - 'X': Slave time-series data.
            - 'Y': Master time-series data.
            - 'initial_conditions': Metadata regarding the starting values.

    Raises:
        FileNotFoundError: If the specified file_path does not exist on disk.
        json.JSONDecodeError: If the file is not a valid JSON format.

    Example:
        >>> data = extract_json_data('dataset/coupled_map_data.json')
        >>> master_trials = data['0.5']['Y']
        >>> print(type(master_trials))
        <class 'numpy.ndarray'>
    '''
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    print(f"Loading data")
    with open(file_path, 'r') as f:
        data = json.load(f)

    if as_numpy:
        # Convert the lists back to numpy arrays for analysis
        for eta in data:
            data[eta]['X'] = np.array(data[eta]['X'])
            data[eta]['Y'] = np.array(data[eta]['Y'])
            
    print("Load complete.")
    return data
# # ===================================================
# # Checkout demo.py for detailed step-by-step usage
# # ===================================================

# # ===================================================
# # Old-version that works on strings only.
# # ===================================================

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import networkx as nx
# import itertools
# import os
# import re

# def calculate_causal_history(seq_1, seq_2):
#     '''
#     Constructs the dictionary GX→Y by scanning Y from left to right and recording the corresponding sub-
#     pattern in X each time a bit flip occurs in Y
    
#     Args:
#         seq_1 (str): Sequence_Y
#         seq_2 (str): Sequence_X
    
#     Returns:
#         G (list): 
#     '''

#     # length of the two sequences
#     len_1 = len(seq_1)
#     len_2 = len(seq_2)

#     # according to problem definition length of the sequences should be equal.
#     if len_1 != len_2:
#         raise ValueError(f'Strings must be of equal length. Got {len_1} and {len_2}.')
    
#     # unique symbols
#     symbols = str(seq_1)
    
#     # Dictionary 
#     G = []
#     last_position = 0
#     k = 1
#     # main logic
#     while k < len_1:
#         # change in bit
#         if seq_1[k] != seq_1[k-1]:
#             segment = seq_2[last_position: k+1]
#             # print(segment)

#             # if we consider 
#             # if segment not in G:
#             #     G.append(segment)

#             G.append(segment)
#             last_position = k + 1
#             k += 1
            
#         k += 1
    
#     return set(G)

# # =============================
# # Previously used method
# # =============================

# # def causality_inference(Gx_y, Gy_x):
# #     '''
# #     Identifies the causal direction by comparing dictionary cardinalities.

# #     The variable X is inferred to be causal for Y if fewer configurations of X 
# #     are required to explain changes in Y than vice-versa.

# #     Args:
# #         gx_y: The dictionary encoding subpatterns in X temporally aligned 
# #             with bit transitions in Y.
# #         gy_x: The dictionary encoding subpatterns in Y temporally aligned 
# #             with bit transitions in X.

# #     Returns:
# #         The name of the variable identified as the causal agent (e.g., 'X' or 'Y').
# #         Returns 'Inconclusive' if cardinalities are equal.

# #     Notes:
# #         An inference of 'X causes Y' is made if |GX→Y| < |GY→X|.
# #     '''

# #     if len(Gx_y) < len(Gy_x):
# #         return 'X -> Y'
# #     if len(Gx_y) > len(Gy_x):
# #         return 'Y -> X'
# #     else:
# #         return 'Undecided'
    

# def sliding_xnor_comparison(p1, p2):
#     '''
#     Identifies common subsequences between two patterns using XNOR-based sliding.

#     The method performs a bitwise comparison by sliding the shorter pattern 
#     across the longer one. It records matches (where bits are identical) and 
#     extracts regions where two or more consecutive bits align.

#     Args:
#         p1 (str): The first binary subpattern from the causal dictionary.
#         p2 (str): The second binary subpattern from the causal dictionary.

#     Returns:
#         set: A collection of common subsequences extracted from the pair 
#             based on consecutive XNOR matches (length >= 2).
#     '''
#     n1, n2 = len(p1), len(p2)

#     # slide shorter pattern over larger
#     if n1 > n2:
#         p1, p2 = p2, p1
#         n1, n2 = n2, n1

#     found_in_pair = set()

#     for shift in range(n2 - n1 + 1):
#         matches = [1 if p1[i] == p2[shift + i] else 0 for i in range(n1)]

#         current_match = []
#         for i, bit in enumerate(matches):
#             if bit == 1:
#                 current_match.append(p2[shift + i])
#             else:
#                 if len(current_match) >= 2:
#                     found_in_pair.add("".join(current_match))

#                 current_match = []
        
#         if len(current_match) >= 2:
#             found_in_pair.add("".join(current_match))
    
#     return found_in_pair


# def generate_pattern_dictionary(G):
#     '''
#     Constructs the overall pattern dictionary PX->Y from a causal history.

#     This method iterates through all distinct pairs in the provided dictionary 
#     G, extracts common sub-patterns via sliding XNOR comparison, and forms 
#     a union of all results. 

#     Args:
#         G (list): The causal history dictionary (e.g., GX->Y) containing 
#             subpatterns of the source sequence.

#     Returns:
#         set: The final pattern dictionary containing unique extracted 
#             subsequences identified as potential causal agents.

#     Notes: 
#         To avoid redundancy, patterns identical to the original segments in G are excluded from the final set (commented lines below).
#     '''
#     pattern = set()
#     pairs = list(itertools.combinations(G, 2))
#     # print(pairs)

#     for p1, p2 in pairs:
#         extracted = sliding_xnor_comparison(p1, p2)
#         pattern.update(extracted)
    
#     # the patterns from source are excluded
#     # final_pattern = {p for p in pattern if p not in G}
#     # print(final_pattern)

#     return pattern

# # # Old non-overlapping code
# # def calculate_contribution_analysis(patterns, X, Y):
# #     '''
# #     Performs frequency and contribution analysis for extracted causal patterns.

# #     This method quantifies the likelihood that a specific pattern in X induces 
# #     a variation in Y. It slides each pattern through sequence X, and for every 
# #     occurrence, it checks if a corresponding bit flip (yk != yk-1) occurs in Y 
# #     at the pattern's terminal index.

# #     The contribution is expressed as the ratio:
# #     R_flip = (Number of flips in Y when pattern occurs) / (Total occurrences in X)

# #     Args:
# #         patterns (set/list): The dictionary of extracted common patterns (PX->Y).
# #         X (str): The causal input binary sequence.
# #         Y (str): The output binary sequence containing bit flips.

# #     Returns:
# #         dict: A dictionary where each pattern is mapped to its 'Change' count, 
# #             'No-change' count, and the computed 'Ratio' (R_flip).
# #     '''
# #     results = {}

# #     for pattern in patterns:
# #         pattern_len = len(pattern)
# #         change_cnt = 0
# #         nochange_cnt = 0
        
# #         # We iterate through every possible starting position
# #         # len(X) - pattern_len + 1 ensures the window fits
# #         for i in range(len(X) - pattern_len + 1):
# #             window = X[i : i + pattern_len]
            
# #             if window == pattern:
# #                 # The terminal index k is the end of the pattern match
# #                 k = i + pattern_len - 1

# #                 # Check for bit flip in Y at index k compared to k-1
# #                 if k > 0 and Y[k] != Y[k-1]:
# #                     change_cnt += 1
# #                 else:
# #                     nochange_cnt += 1

# #         # Ratio calculation (R_flip)
# #         total = change_cnt + nochange_cnt
# #         ratio = change_cnt / total if total > 0 else 0

# #         results[pattern] = {
# #             'Change': change_cnt,
# #             'No-change': nochange_cnt,
# #             'Ratio': round(ratio, 3)
# #         }

# #     # Format as a sorted DataFrame
# #     df = pd.DataFrame(results).T
# #     df.index.name = 'Pattern'
# #     return df.sort_values(by='Ratio', ascending=False)


# def calculate_contribution_analysis(patterns, X, Y):
#     '''
#     Performs frequency and contribution analysis for extracted causal patterns.

#     This method quantifies the likelihood that a specific pattern in X induces 
#     a variation in Y. It slides each pattern through sequence X, and for every 
#     occurrence, it checks if a corresponding bit flip (yk != yk-1) occurs in Y 
#     at the pattern's terminal index.

#     The contribution is expressed as the ratio:
#     R_flip = (Number of flips in Y when pattern occurs) / (Total occurrences in X)

#     Args:
#         patterns (set/list): The dictionary of extracted common patterns (PX->Y).
#         X (str): The causal input binary sequence.
#         Y (str): The output binary sequence containing bit flips.

#     Returns:
#         dict: A dictionary where each pattern is mapped to its 'Change' count, 
#             'No-change' count, and the computed 'Ratio' (R_flip).
#     '''
#     results = {}

#     for pattern in patterns:
#         pattern_len = len(pattern)
#         # Find all overlapping occurrences
#         matches = [m.start() for m in re.finditer(f'(?={re.escape(pattern)})', X)]
        
#         change_cnt = 0
#         nochange_cnt = 0
        
#         for start_idx in matches:
#             # Check for ANY bit flip in the target within the pattern's window
#             has_flip = False
            
#             # A pattern of length L has L-1 internal transition points
#             for offset in range(1, pattern_len):
#                 k = start_idx + offset
#                 if k < len(Y):
#                     if Y[k] != Y[k-1]:
#                         has_flip = True
#                         break # Flip found, count as 'Change' and move to next match
            
#             if has_flip:
#                 change_cnt += 1
#             else:
#                 nochange_cnt += 1

#         total = len(matches)
#         ratio = change_cnt / total if total > 0 else 0

#         results[pattern] = {
#             'Change': change_cnt,
#             'No-change': nochange_cnt,
#             'Ratio': round(ratio, 4)
#         }

#     df = pd.DataFrame(results).T
#     df.index.name = 'Pattern'
#     return df


# def barchart_analysis(stats, file_name='causal_analysis_chart', direction='x->y'):
#     '''
#     Generates and saves a bar chart visualizing the contribution ratio (R_flip) 
#     for each extracted pattern.

#     This method creates a high-resolution visualization of the predictive power 
#     of patterns in the PX->Y set. It plots the R_flip ratio on the Y-axis against 
#     the binary patterns on the X-axis, allowing for easy identification of 
#     perfect causal predictors (where Ratio = 1.0).

#     The function automatically handles directory creation, saves the output 
#     as a high-DPI PNG file in the 'results' folder, and applies a 'tight' 
#     bounding box to ensure labels are not clipped.

#     Args:
#         stats (pd.DataFrame): A transposed DataFrame containing binary patterns 
#             as indices and their corresponding 'Ratio' (R_flip) as a column.
#         file_name (str): File name for saving the file.
#         direction (str): Causal direction to be considered for barchart analysis.

#     Returns:
#         None: Displays the plot and saves the file to 'results/causal_analysis_chart.png'.
#     '''
#     fig, ax = plt.subplots(figsize=(15, 6))

#     patterns = stats.index 
#     x = np.arange(len(patterns))
#     width = 0.6

#     ax.bar(
#         x, 
#         stats['Ratio'], 
#         width,
#         label='Ratio ($R_{flip}$)',
#         color='#2ecc71',
#         edgecolor='black'
#     )
    
#     for i, v in enumerate(stats['Ratio']):
#         ax.text(i, v + 0.02, str(v), ha='center', fontweight='bold')

#     ax.set_xlabel(f'Binary Patterns ($P_{direction}$)', fontsize=14, fontweight='bold', labelpad=10)
#     ax.set_ylabel('Ratio Score ($R_{flip}$)', fontsize=14, fontweight='bold', labelpad=10)
#     ax.set_title('Contribution Analysis: Pattern Predictive Power', fontsize=16, fontweight='bold')
#     ax.set_xticks(x)
#     ax.set_xticklabels(patterns, rotation=45)
#     ax.set_ylim(0, 1.1)

#     ax.tick_params(axis='x', labelsize=12)
#     ax.tick_params(axis='y', labelsize=12)
#     # ax.legend()

#     if not os.path.exists('results'):
#         os.makedirs('results')

#     plt.savefig(f'results/{file_name}.png', bbox_inches='tight', dpi=300)

#     plt.tight_layout()
#     plt.show()


# def generate_causal_network(stats, file_name='causal_network_graph', target_node='Y'):
#     '''
#     Constructs an aesthetic radial network graph representing causal proximity.

#     This function visualizes the relationship between extracted patterns (PX->Y) 
#     and the target variable Y. It utilizes a polar coordinate system where the 
#     radial distance from the center node Y is strictly determined by the 
#     predictive reliability of the pattern.

#     The distance (d) is calculated using the linear transformation:
#     d = 10 * (2 - Ratio)

#     Under this logic, 'Perfect Predictors' (Ratio = 1.0) are positioned at a 
#     distance of 1.0 units from Y, while non-causal patterns (Ratio = 0.0) 
#     are placed at a maximum distance of 11.0 units. Nodes are distributed 
#     equidistantly in a circular arrangement to prevent label overlap and 
#     ensure visual clarity.

#     Args:
#         stats (pd.DataFrame): A transposed DataFrame where the index contains 
#             binary patterns and the 'Ratio' column contains the R_flip values.
#         file_name (str): File name for saving the file.
#         target_node (str): Target node for network.

#     Returns:
#         None: Generates a Matplotlib figure, saves it to 'results/causal_network_aesthetic.png', 
#             and displays the output.
#     '''
#     G = nx.DiGraph()
    
#     stats_sorted = stats.sort_values(by='Ratio', ascending=False)
    
#     patterns = stats_sorted.index.tolist()
#     ratios = stats_sorted['Ratio'].tolist()
    
#     pos = {target_node: np.array([0, 0])}
    
#     angle_step = 2 * np.pi / len(patterns)
    
#     for i, pattern in enumerate(patterns):
#         ratio = ratios[i]
#         distance = 10 * (2 - ratio)
        
#         angle = i * angle_step
#         pos[pattern] = np.array([distance * np.cos(angle), distance * np.sin(angle)])
        
#         G.add_node(pattern, ratio=ratio)
#         G.add_edge(pattern, target_node, ratio=ratio)

#     plt.figure(figsize=(12, 12))
    
#     nx.draw_networkx_nodes(G, pos, nodelist=[target_node], 
#                            node_color='skyblue', node_size=3000, 
#                            edgecolors='black', linewidths=2)
    
#     nx.draw_networkx_nodes(G, pos, nodelist=patterns, 
#                            node_color='#ff9933', node_size=2000, 
#                            edgecolors='black', linewidths=1.5)

#     for u, v, d in G.edges(data=True):
#         nx.draw_networkx_edges(
#             G, pos, edgelist=[(u, v)],
#             arrowstyle='-|>', 
#             arrowsize=30, 
#             width=1.5 + (d['ratio'] * 3),
#             edge_color='gray',
#             alpha=0.6,
#             node_size=3000
#         )

#     nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold')
    
#     edge_labels = {(u, v): f"{G[u][v]['ratio']:.2f}" for u, v in G.edges()}
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
#                                  font_size=10, label_pos=0.5, font_color='red')

#     plt.title('Causal Proximity Network\n' + r'$Distance = 10 \cdot (2 - R_{flip})$', 
#               fontsize=18, fontweight='bold', pad=20)
#     plt.text(0.95, 0.01, f"Nodes closer to {target_node} = Stronger Causality", 
#              transform=plt.gcf().transFigure, ha='right', fontsize=12, style='italic')

#     plt.axis('off')
    
#     if not os.path.exists('results'):
#         os.makedirs('results')
#     plt.savefig(f'results/{file_name}.png', bbox_inches='tight', dpi=300)
#     plt.show()


# # Updated causality 
# def get_direction_metrics(df):
#     '''
#     Extracts key causal descriptors (N1, L1, R_bar) from a pattern contribution dictionary.

#     This function reduces the high-dimensional pattern-match statistics into three 
#     fundamental quantities required for causal direction identification. These metrics 
#     capture both the deterministic strength and the global statistical tendency of 
#     a specific causal hypothesis (e.g., the direction X -> Y).

#     The quantities computed are:
#     1. N(1) [Deterministic Count]: The total count of 'Perfect Predictors'—patterns 
#        that possess a Ratio of exactly 1.0, indicating they are always followed by 
#        a bit flip in the target sequence.
#     2. L(1) [Maximum Complexity]: The maximum bit-length among all Perfect Predictors. 
#        This quantifies the deepest level of deterministic dependency detected 
#        within the sequence.
#     3. R_bar [Mean Tendency]: The arithmetic mean of the 'Ratio' column across all 
#        patterns in the dictionary, serving as a measure of the global causal 
#        influence strength.

#     Args:
#         df (pd.DataFrame): A DataFrame where the index contains unique binary 
#             patterns and the 'Ratio' column contains the computed R_flip values.

#     Returns:
#         tuple: A triplet of metrics (n_1, l_1, r_bar) where:
#             - n_1 (int): Count of patterns with a perfect ratio.
#             - l_1 (int): Length of the longest perfect pattern.
#             - r_bar (float): The average contribution ratio of all patterns.
#     '''
#     # 1. N(1): Number of patterns with ratio exactly 1.0
#     df_1 = df[df['Ratio'] == 1.0]
#     n_1 = len(df_1)
    
#     # 2. L(1): Maximum length among patterns with ratio 1.0
#     # If no patterns have ratio 1, length is 0
#     l_1 = df_1.index.map(len).max() if n_1 > 0 else 0
    
#     # 3. R_bar: Average ratio over all patterns
#     r_bar = df['Ratio'].mean() if not df.empty else 0.0
    
#     return n_1, l_1, r_bar


# def identify_causality(df_xy, df_yx, tau=0.05):
#     '''
#     Identifies the dominant causal direction between two sequences using hierarchical inference rules.

#     This function implements a multi-tiered decision logic to determine whether X causes Y, 
#     Y causes X, or if the relationship is independent/undetermined. The logic prioritizes 
#     deterministic evidence—patterns that perfectly predict a state change—before falling 
#     back to statistical averages.

#     The identification follows a strict hierarchy of rules:
    
#     1. Independence Rule: If the absolute difference between mean ratios (|R̄x->y - R̄y->x|) 
#        is less than the threshold tau, the sequences are considered independent.
#     2. Primary Rule (Deterministic Strength): The direction with the higher count of 
#        'Perfect Predictors' (N_1, where Ratio == 1.0) is identified as causal.
#     3. Secondary Rule (Pattern Complexity): If N_1 is equal, the direction with the 
#        longest perfect pattern (L_1) is identified as causal, as longer patterns 
#        represent more complex deterministic dependencies.
#     4. Tertiary Rule (Mean Tendency): If both N_1 and L_1 are equal, the direction 
#        with the higher overall average ratio (R̄) is selected.
#     5. Tie Case: If all metrics are identical, the direction is marked 'Undetermined'.

#     Args:
#         df_xy (pd.DataFrame): Contribution analysis results for the X -> Y direction.
#         df_yx (pd.DataFrame): Contribution analysis results for the Y -> X direction.
#         tau (float, optional): Threshold for the independence rule. Represents the 
#             minimum significant difference in mean ratios. Defaults to 0.05.

#     Returns:
#         str: A string indicating the detected causal relationship: 
#              "X causes Y", "Y causes X", "No causal influence detected", or "Undetermined".
#     '''
#     # Extract metrics for both directions
#     nx_y, lx_y, rx_y = get_direction_metrics(df_xy)
#     ny_x, ly_x, ry_x = get_direction_metrics(df_yx)
    
#     # --- 1. Independence Rule ---
#     if abs(rx_y - ry_x) < tau:
#         return "No causal influence detected (Independence)"

#     # --- 2. Primary Decision Rule (Deterministic Strength) ---
#     if nx_y > ny_x:
#         return "X causes Y"
#     elif ny_x > nx_y:
#         return "Y causes X"

#     # --- 3. Secondary Decision Rule (Pattern Length) ---
#     if lx_y > ly_x:
#         return "X causes Y"
#     elif ly_x > lx_y:
#         return "Y causes X"

#     # --- 4. Tertiary Decision Rule (Mean Tendency) ---
#     if rx_y > ry_x:
#         return "X causes Y"
#     elif ry_x > rx_y:
#         return "Y causes X"

#     # --- 5. Tie Case ---
#     return "Undetermined"

# # ===================================================
# # Checkout demo.py for detailed step-by-step usage
# # ===================================================


