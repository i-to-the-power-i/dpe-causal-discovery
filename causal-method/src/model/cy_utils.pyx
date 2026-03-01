# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
import pandas as pd
import cython
from typing import List, Set, Any
import json

cnp.import_array()

def _ensure_uint8(seq: Any):
    """Internal helper to get a read-only byte view."""
    if isinstance(seq, str):
        return seq.encode('ascii')
    elif isinstance(seq, list):
        return np.array(seq, dtype=np.uint8)
    return np.asanyarray(seq, dtype=np.uint8)

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
    cdef const cnp.uint8_t[:] s1 = _ensure_uint8(seq_1)
    cdef const cnp.uint8_t[:] s2 = _ensure_uint8(seq_2)
    
    cdef int len_1 = s1.shape[0]
    cdef int last_position = 0
    cdef int k = 1
    cdef list G = []
    
    # if len_1 != s2.shape[0]:
    #     raise ValueError(f"Sequences must be of equal length.")
    
    while k < len_1:
        if s1[k] != s1[k-1]:
            # byte slicing
            G.append(bytes(s2[last_position : k+1]).decode('ascii'))
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
    cdef const cnp.uint8_t[:] v1 = _ensure_uint8(p1)
    cdef const cnp.uint8_t[:] v2 = _ensure_uint8(p2)
    cdef int n1 = v1.shape[0]
    cdef int n2 = v2.shape[0]
    
    if n1 > n2:
        v1, v2 = v2, v1
        n1, n2 = n2, n1

    found_in_pair = set()
    cdef int shift, i
    cdef list current_match
    
    for shift in range(n2 - n1 + 1):
        current_match = []
        for i in range(n1):
            if v1[i] == v2[shift + i]:
                current_match.append(chr(v1[i]))
            else:
                if len(current_match) >= 2:
                    found_in_pair.add("".join(current_match))
                current_match = []
        
        if len(current_match) >= 2:
            found_in_pair.add("".join(current_match))
    
    return found_in_pair

def generate_pattern_dictionary(G_input):
    '''
    Constructs the overall pattern dictionary PX->Y from a causal history.
    Iterates through all distinct pairs in G and extracts common sub-patterns.

    Args:
        G (set/list): The causal history dictionary GX->Y.

    Returns:
        set: The final PX->Y pattern dictionary.
    '''
    G = list(G_input)
    patterns = set()
    cdef int n = len(G)
    cdef int i, j
    
    for i in range(n):
        for j in range(i + 1, n):
            extracted = sliding_xnor_comparison(G[i], G[j])
            patterns.update(extracted)
    
    # if there is no pattern in patterns set.
    if not patterns:
        return G_input

    return patterns

def calculate_contribution_analysis(patterns, X_in, Y_in):
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
    cdef const cnp.uint8_t[:] X = _ensure_uint8(X_in)
    cdef const cnp.uint8_t[:] Y = _ensure_uint8(Y_in)
    cdef const cnp.uint8_t[:] p
    
    cdef int x_len = X.shape[0]
    cdef int y_len = Y.shape[0]
    cdef int p_len, i, j, k, change_cnt, match_count
    cdef bint has_flip, is_match
    
    results = {}

    for pattern in patterns:
        p = _ensure_uint8(pattern)
        p_len = p.shape[0]
        change_cnt = 0
        match_count = 0
        
        # searching for pattern
        for i in range(x_len - p_len + 1):
            is_match = True
            for j in range(p_len):
                if X[i + j] != p[j]:
                    is_match = False
                    break
            
            if is_match:
                match_count += 1
                has_flip = False
                for k in range(i + 1, i + p_len):
                    if k < y_len and Y[k] != Y[k-1]:
                        has_flip = True
                        break
                
                if has_flip:
                    change_cnt += 1

        ratio = <float>change_cnt / match_count if match_count > 0 else 0.0
        results[pattern] = {
            'Change': change_cnt,
            'No-change': match_count - change_cnt,
            'Ratio': round(ratio, 4)
        }

    df = pd.DataFrame(results).T
    df.index.name = 'Pattern'
    return df

def discrete(sequence: np.ndarray, n_bins: int = 2):
    '''
    Discretizes a continuous signal into symbolic integer bins.

    This function uses uniform binning to map a range of continuous values 
    to a set of discrete states. This transformation is a prerequisite for 
    symbolic information-theoretic analysis (like Entropy or Mutual Information).

    

    Args:
        sequence (np.ndarray): The input numerical time-series data.
        n_bins (int): The number of discrete levels to create.

    Returns:
        np.ndarray: An array of integers in the range [0, n_bins - 1].
    '''
    if len(sequence) == 0:
        return np.empty(0, dtype=int)

    seq_min = sequence.min()
    seq_max = sequence.max()
    data_range = seq_max - seq_min

    if data_range == 0:
        return np.zeros(len(sequence), dtype=int)

    bin_width = data_range / n_bins
    indices = ((sequence - seq_min) / bin_width).astype(int)
    
    return np.clip(indices, 0, n_bins - 1)

def generator(length: int = 1000, noise_intensity: float = 0.01, 
              a: float = 0.8, b: float = 0.8, phi: float = 0.0):
    '''
    Generates synthetic bivariate time series using a coupled Autoregressive (AR) model.

    The relationship is defined by:
    - Y[t] = b * Y[t-1] + noise (Independent process)
    - X[t] = a * X[t-1] + phi * Y[t-1] + noise (Coupled process)

    The parameter 'phi' controls the unidirectional causal influence from Y to X.

    

    Args:
        length (int): Total number of time steps.
        noise_intensity (float): Standard deviation of the Gaussian noise.
        a (float): Auto-dependence coefficient for X.
        b (float): Auto-dependence coefficient for Y.
        phi (float): Coupling strength (Y -> X).

    Returns:
        tuple[np.ndarray, np.ndarray]: Generated time series (X, Y).
    '''
    X = np.zeros(length)
    Y = np.zeros(length)

    ex = np.random.normal(0, noise_intensity, length)
    ey = np.random.normal(0, noise_intensity, length)

    for t in range(1, length):
        Y[t] = (b * Y[t-1]) + ey[t]
        X[t] = (a * X[t-1]) + (phi * Y[t-1]) + ex[t]

    return X, Y

def coupling_experiment(num_sequences: int = 2000, sequence_length: int = 1500, transients: int = 500):
    '''
    Performs an ensemble simulation across a range of coupling strengths (phi).
    The first `transients` steps are discarded to ensure the system has reached 
    a steady state. Returns raw continuous data.

    Args:
        num_sequences (int): Number of independent realizations per phi.
        sequence_length (int): Final number of time steps per realization (after transient removal).
        transients (int): Number of initial steps to discard.

    Returns:
        dict: A mapping of phi values to dictionaries containing raw 'X' and 'Y' arrays.
    '''
    phis = np.arange(0, 1, 0.05)
    results = {}
    
    total_length = sequence_length + transients

    for phi in phis:
        X_data = np.empty((num_sequences, sequence_length), dtype=float)
        Y_data = np.empty((num_sequences, sequence_length), dtype=float)

        for i in range(num_sequences):
            x_raw, y_raw = generator(total_length, 0.01, 0.8, 0.8, phi)
            
            X_data[i, :] = x_raw[transients:]
            Y_data[i, :] = y_raw[transients:]

        results[phi] = {'X': X_data, 'Y': Y_data}

    return results

# # ===================================================
# # Updated method based on weighted binary entropy.
# # ===================================================
# Rflip
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
    cdef float entropy, weight, total_windows
    
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
        float: Causal strength.
    '''
    cdef float avg_h_xy, avg_h_yx
    
    avg_h_xy = average_weighted_entropy(stats_x_y, len(X))
    avg_h_yx = average_weighted_entropy(stats_y_x, len(Y))
    
    if avg_h_xy < avg_h_yx:
        return 'X -> Y', avg_h_xy, avg_h_yx
    elif avg_h_yx < avg_h_xy:
        return 'Y -> X', avg_h_xy, avg_h_yx
    elif avg_h_xy == avg_h_yx:
        return 'Independence', avg_h_xy, avg_h_yx
    
    return 'Undetermined', avg_h_xy, avg_h_yx

# # ===================================================

def run_causal_analysis(seq_x_bin, seq_y_bin):
    """
    Executes the full causal discovery pipeline on a single pair of binary sequences.
    
    This Cythonized version uses typed memoryviews/arrays and float precision 
    to reduce Python overhead during the pipeline execution.
    """
    # 1. Efficiently convert float arrays to strings for the logic gate
    # We use a join with a list comprehension for the bridge to Python-based logic
    cdef str x_str = "".join([str(int(i)) for i in seq_x_bin])
    cdef str y_str = "".join([str(int(i)) for i in seq_y_bin])
    
    # 2. Calculate Causal History
    # Assuming these functions are defined as 'def' or 'cpdef' in your cy_utils
    Gx_y = calculate_causal_history(y_str, x_str)
    Gy_x = calculate_causal_history(x_str, y_str)
    
    # 3. Generate Pattern Dictionaries
    Px_y = generate_pattern_dictionary(Gx_y)
    Py_x = generate_pattern_dictionary(Gy_x)
    
    # 4. Contribution Analysis
    # We pass the strings and the dictionaries to the analysis functions
    stats1 = calculate_contribution_analysis(Px_y, x_str, y_str)
    stats2 = calculate_contribution_analysis(Py_x, y_str, x_str)
    
    # 5. Final Verdict
    # Returns the final directionality verdict
    return identify_causality(stats1, stats2, x_str, y_str)
# # ===================================================

cdef float c_skew_tent_map(x, b):
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
    if 0.0 <= x < b:
        return x / b
    elif b <= x < 1.0:
        return (1.0 - x) / (1.0 - b)
    else:
        return 0.0

# def generate_coupled_data(b1=0.65, b2=0.47, initial_values=None):
#     '''
#     Generates time series data for a Master-Slave system of coupled skew-tent maps.
    
#     This function simulates a unidirectional coupling where the Master system (Y) 
#     evolves independently, and the Slave system (X) is influenced by the Master 
#     based on a coupling coefficient 'eta'. Data is generated for multiple 
#     coupling strengths, with initial transients removed to ensure the system 
#     is on a chaotic attractor.
    
#     The simulation follows these governing equations:
#     M(t) = T1(M(t-1))
#     S(t) = (1 - eta) * T2(S(t-1)) + eta * M(t)
    
#     Args:
#         b1 (float): Skewness parameter for the Master map (T1). Defaults to 0.65.
#         b2 (float): Skewness parameter for the Slave map (T2). Defaults to 0.47.
#         initial_values (dict, optional): Manual starting points for specific 
#             coupling strengths. Format: {"0.1": {"M": val, "S": val}}. 
#             If None, values are chosen randomly from U(0, 1).
            
#     Returns:
#         None: Saves a nested dictionary to 'dataset/coupled_map_data.json' 
#             containing 'X' (Slave), 'Y' (Master), and 'initial_conditions'.
            
#     Note:
#         - Number of trials per eta: 2000
#         - Time series length: 1500 (after discarding 500 transient steps).
#         - Coupling (eta) range: [0.0, 0.1, ..., 0.9].
#     '''
#     coupling_coefficients = np.arange(0, 1.0, 0.1)
#     cdef int num_trials = 2000
#     cdef int seq_len = 1500
#     cdef int transient = 500
#     cdef int total_steps = seq_len + transient
    
#     cdef int trial, t
#     cdef float eta, m_val, s_val, m_next, s_next
    
#     results = {}

#     print(f"{'Coupling (η)':<15} | {'Initial M':<15} | {'Initial S':<15}")

#     for eta in coupling_coefficients:
#         eta_key = f"{eta:.1f}"
#         results[eta_key] = {"X": [], "Y": [], "initial_conditions": []}
        
#         for trial in range(num_trials):
#             if initial_values and eta_key in initial_values:
#                 m_val = initial_values[eta_key]['M']
#                 s_val = initial_values[eta_key]['S']
#             else:
#                 m_val = np.random.uniform(0, 1)
#                 s_val = np.random.uniform(0, 1)

#             results[eta_key]["initial_conditions"].append({"M0": m_val, "S0": s_val})

#             if trial == 0:
#                 print(f"{eta_key:<15} | {m_val:<15.6f} | {s_val:<15.6f}")

#             m_series = np.empty(seq_len-transient, dtype=np.float64)
#             s_series = np.empty(seq_len-transient, dtype=np.float64)

#             for t in range(total_steps):
#                 m_next = c_skew_tent_map(m_val, b1)
#                 s_next = (1.0 - eta) * c_skew_tent_map(s_val, b2) + eta * m_next
                
#                 m_val = m_next
#                 s_val = s_next

#                 if t >= transient:
#                     m_series[t - transient] = m_val
#                     s_series[t - transient] = s_val

#             results[eta_key]["Y"].append(m_series.tolist())
#             results[eta_key]["X"].append(s_series.tolist())

#     output_path = 'dataset/coupled_map_data.json'
#     with open(output_path, 'w') as f:
#         json.dump(results, f)
    
#     print(f"\nData generation complete. Saved to: {output_path}")

# # Updated 1D coupled skew-tent maps
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
    S(t) = (1 - eta) * T2(S(t-1)) + eta * M(t)
    
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
        - Number of trials per eta: 2000
        - Time series length: 1500 (after discarding 500 transient steps).
        - Coupling (eta) range: [0.0, 0.1, ..., 0.9].
    '''
    coupling_coefficients = np.arange(0, 1.0, 0.1)
    cdef int num_trials = 2000
    cdef int seq_len = 1500        
    cdef int transient = 500       
    cdef int total_steps = seq_len + transient 
    
    cdef int trial, t
    cdef float eta, m_init, s_init
    
    results = {}

    print(f"{'Coupling (η)':<15} | {'Initial M':<15} | {'Initial S':<15}")

    for eta in coupling_coefficients:
        eta_key = f"{eta:.1f}"
        results[eta_key] = {"X": [], "Y": [], "initial_conditions": []}
        
        for trial in range(num_trials):
            if initial_values and eta_key in initial_values:
                m_init = initial_values[eta_key]['M']
                s_init = initial_values[eta_key]['S']
            else:
                m_init = np.random.uniform(0, 1)
                s_init = np.random.uniform(0, 1)

            results[eta_key]["initial_conditions"].append({"M0": m_init, "S0": s_init})

            if trial == 0:
                print(f"{eta_key:<15} | {m_init:<15.6f} | {s_init:<15.6f}")

            m_series = np.zeros(total_steps, dtype=np.float64)
            s_series = np.zeros(total_steps, dtype=np.float64)
            
            m_series[0] = m_init
            s_series[0] = s_init

            for t in range(1, total_steps):
                m_series[t] = c_skew_tent_map(m_series[t-1], b1)
                
                m_val_current = m_series[t]
                s_series[t] = (1.0 - eta) * c_skew_tent_map(s_series[t-1], b2) + eta * m_val_current

            results[eta_key]["Y"].append(m_series[transient:].tolist())
            results[eta_key]["X"].append(s_series[transient:].tolist())

    return results

# # ===================================================
# # Old-version that works on strings only.
# # ===================================================

# # cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
# import numpy as np
# cimport numpy as cnp
# import pandas as pd
# import cython
# from typing import Any, List

# def calculate_causal_history(seq_1, seq_2):
#     cdef int len_1 = len(seq_1)
#     cdef int len_2 = len(seq_2)
    
#     if len_1 != len_2:
#         raise ValueError(f"Strings must be of equal length. Got {len_1} and {len_2}.")
    
#     # Using a list to collect segments; strings are immutable in Python/Cython 
#     # so we still interact with Python objects here, but the loop is C-speed.
#     cdef list G = []
#     cdef int last_position = 0
#     cdef int k = 1
    
#     while k < len_1:
#         if seq_1[k] != seq_1[k-1]:
#             G.append(seq_2[last_position : k+1])
#             last_position = k + 1
#             k += 1
#         k += 1
    
#     return set(G)


# def sliding_xnor_comparison(p1, p2):
#     cdef int n1 = len(p1)
#     cdef int n2 = len(p2)
#     cdef str temp_p
#     cdef int temp_n

#     if n1 > n2:
#         p1, p2 = p2, p1
#         n1, n2 = n2, n1

#     found_in_pair = set()
#     cdef int shift, i
#     cdef list current_match
    
#     for shift in range(n2 - n1 + 1):
#         current_match = []
#         for i in range(n1):
#             if p1[i] == p2[shift + i]:
#                 current_match.append(p2[shift + i])
#             else:
#                 if len(current_match) >= 2:
#                     found_in_pair.add("".join(current_match))
#                 current_match = []
        
#         if len(current_match) >= 2:
#             found_in_pair.add("".join(current_match))
    
#     return found_in_pair


# def generate_pattern_dictionary(g):
#     cdef list G = list(g)
#     cdef set patterns = set()
#     cdef int n = len(G)
#     cdef int i, j
#     cdef str p1, p2
#     cdef set extracted

#     # Manual nested loop is faster in Cython than itertools.combinations
#     for i in range(n):
#         p1 = G[i]
#         for j in range(i + 1, n):
#             p2 = G[j]
            
#             # Call the previously optimized sliding comparison
#             extracted = sliding_xnor_comparison(p1, p2)
            
#             # Update the master set
#             if extracted:
#                 patterns.update(extracted)
    
#     return patterns

# # # Old-version with non-overlapping
# # def calculate_contribution_analysis(patterns, X, Y):
# #     # This function speeds up the sliding window match and bit-flip check
# #     results = {}
# #     cdef int len_X = len(X)
# #     cdef int pattern_len, i, k, change_cnt, nochange_cnt
# #     cdef str pattern, window
    
# #     for pattern in patterns:
# #         pattern_len = len(pattern)
# #         change_cnt = 0
# #         nochange_cnt = 0
# #         i = 0
        
# #         while i <= len_X - pattern_len:
# #             # Slicing is still a Python operation, but the logic flow is C-speed
# #             window = X[i : i + pattern_len]
            
# #             if window == pattern:
# #                 k = i + pattern_len - 1
# #                 if k > 0 and Y[k] != Y[k-1]:
# #                     change_cnt += 1
# #                 else:
# #                     nochange_cnt += 1
# #             #     i += pattern_len
# #             # else:
# #             i += 1

# #         total = change_cnt + nochange_cnt
# #         ratio = float(change_cnt) / total if total > 0 else 0.0
# #         results[pattern] = {from typing import Any, List
# #             'Change': change_cnt, 
# #             'No-change': nochange_cnt, 
# #             'Ratio': round(ratio, 3)
# #         }
# #     return pd.DataFrame(results).T.sort_values(by='Ratio', ascending=False)

# # # New-version with overlapping
# def calculate_contribution_analysis(patterns, str X_in, str Y_in):
#     # Convert input strings to bytes for memoryview access
#     cdef bytes X_bytes = X_in.encode('ascii')
#     cdef bytes Y_bytes = Y_in.encode('ascii')
    
#     cdef const unsigned char[:] X = X_bytes # type: ignore
#     cdef const unsigned char[:] Y = Y_bytes # type: ignore
    
#     # Declare all C-level variables at the top
#     cdef int x_len = len(X)
#     cdef int y_len = len(Y)
#     cdef int p_len, i, j, k, match_count, change_cnt
#     cdef int is_match, has_flip
#     cdef float ratio
#     cdef bytes p_bytes
#     cdef const unsigned char[:] p
    
#     results = {}

#     for pattern in patterns:
#         # Prepare the current pattern as a memoryview
#         p_bytes = pattern.encode('ascii')
#         p = p_bytes # type: ignore
#         p_len = len(p)
        
#         match_count = 0
#         change_cnt = 0
        
#         # Sliding window through X (replaces re.finditer)
#         for i in range(x_len - p_len + 1):
#             # Check if pattern matches at current index i
#             is_match = 1
#             for j in range(p_len):
#                 if X[i + j] != p[j]: # type: ignore
#                     is_match = 0
#                     break
            
#             if is_match == 1:
#                 match_count += 1
#                 has_flip = 0
                
#                 # Check for bit flips in Y within the window
#                 # Y[k] != Y[k-1]
#                 for k in range(i + 1, i + p_len):
#                     if k < y_len:
#                         if Y[k] != Y[k - 1]: # type: ignore
#                             has_flip = 1
#                             break
                
#                 if has_flip == 1:
#                     change_cnt += 1

#         # Calculate metrics for this pattern
#         ratio = 0.0
#         if match_count > 0:
#             ratio = <float>change_cnt / <float>match_count

#         results[pattern] = {
#             'Change': change_cnt,
#             'No-change': match_count - change_cnt,
#             'Ratio': round(ratio, 4)
#         }

#     df = pd.DataFrame(results).T
#     df.index.name = 'Pattern'
#     return df