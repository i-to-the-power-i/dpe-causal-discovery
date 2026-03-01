# src/model/main.py
import sys
import os

# Import from siblings in the same package
from model.cy_utils import (
    calculate_causal_history, 
    generate_pattern_dictionary, 
    calculate_contribution_analysis
)
from model.utils import (
    identify_causality, 
    barchart_analysis, 
    generate_causal_network
)

def main():
    """Main entry point for the analyze-causality command."""
    print("--- Causal Pattern Analysis Pipeline ---")
    
    # # Example data
    # y = "000001000010000000000100001000"
    # x = "011101111010011001110101101001"

    y = input('Enter sequence Y\n')
    x = input('Enter sequence X\n')

    # 1. Evaluate Direction
    gx_y = calculate_causal_history(y, x)
    gy_x = calculate_causal_history(x, y)

    print(f'''Final Dictionaries
    Gx->y: {gx_y}
    Gy->x: {gy_x}
    ''')

    # 2. Pattern Extraction
    px_y = generate_pattern_dictionary(gx_y)
    py_x = generate_pattern_dictionary(gy_x)

    print("\nFinal Pattern Dictionary PX->Y:")
    print(sorted(list(px_y), key=len))
    print("\nFinal Pattern Dictionary PY->X:")
    print(sorted(list(py_x), key=len))
    
    # 3. Contribution Analysis
    stats1 = calculate_contribution_analysis(px_y, x, y)
    stats2 = calculate_contribution_analysis(py_x, y, x)
    print("\nPattern Statistics:")
    print(stats1)
    print(stats2)


    # 4. Visualization
    # Ensure results directory exists relative to current working directory
    if not os.path.exists('results'):
        os.makedirs('results')
        
    barchart_analysis(stats1, 'causal_analysis_chart_xy', direction='x->y')
    generate_causal_network(stats1, 'causal_network_graph_xy', target_node='Y')

    barchart_analysis(stats2, 'causal_analysis_chart_yx', direction='y->x')
    generate_causal_network(stats2, 'causal_network_graph_yx', target_node='X')

    direction, _= identify_causality(stats1, stats2, x, y)
    print(f"Detected Causal Direction: {direction}")
    
if __name__ == "__main__":
    main()