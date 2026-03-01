# Dictionary Based Pattern Entropy Method for Causal Inference.

This implements a causal inference framework designed to extract binary patterns from time-series data and quantify their predictive power using the $R_{flip}$ ratio. It includes **Cython-optimized** computational kernels for high-performance analysis and advanced **NetworkX** visualizations.

## Features

* **Causal History Extraction**: Identifies sub-patterns in sequence $X$ that correspond to bit-flips in sequence $Y$.
* **Cython Optimization**: High-speed pattern matching and contribution analysis.
* **Aesthetic Network Mapping**: Visualizes causal strength.
* **Automated Reporting**: Generates high-resolution (300 DPI) charts saved automatically to `/results`.

---

## Getting Started

You can set up the project using one of the two methods below:

1. **Clone via Terminal:**
   `git clone https://github.com/i-to-the-power-i/dpe-causal-discovery.git`
    This method requires manual structuring of the files and folders according to the `Project Structure`.
   
3. **Download Manual Bundle:**
   [Download DPE.zip from Google Drive](https://drive.google.com/file/d/1-HfHr1tWR3A9-pT3lZ85Rf5eJzThILQr/view?usp=sharing)
   The `DPE.zip` folder contains all the files, datasets and documentation in `Project Structure` (**Most Preferred**).

---

## Project Structure

```text
DPE/
├──causal-method/
    ├── environment.yml        # Conda environment set-up
    ├── pyproject.toml         # Build system & CLI metadata
    ├── setup.py               # Cython compilation script
    ├── requirements.txt       # Project dependencies
    ├── README.md              # Documentation
    ├── docs/                  # Complete Documentation with examples
    └── src/
        └── model/             # Core Package
            ├── __init__.py    # Package initializer
            ├── main.py        # CLI Entry point logic
            ├── cy_utils.pyx   # Computational kernels
            └── utils.pyx      # Visuals and Inference logic
            └── demo.py        # Examples and updated network graph
    └── results/               # Automatically generated .png output
└──experiments/            # Experiments
        └── ETCPy/             # ETCPy package fork from CCMs
        └── results/

```
---

## Installation & Setup

### 1. Prerequisites
Ensure you have a C compiler installed (GCC on Linux/Mac or MSVC on Windows).


### 2. Standard Installation (via pyproject.toml)
From the **root directory** (`DPE/`), install the project in editable mode. This pulls dependencies from `requirements.txt` and registers the `analyze-causality` command:

```bash
# Navigate to project root
cd causal-method

# Create the conda environment
conda env create -f environment.yml

# Install project and dependencies
pip install -e .

# Cython files .pyx Compilation
python setup.py build_ext -inplace

# Return to the root directory
cd ..

```
From the **root directory** `DPE/`, navigate to `/experiments/ETCPY`:

```bash
# Navigate to the ETCPy directory
cd experiments/ETCPy/

# Install the package without build isolation
pip install -e . --no-build-isolation

# Return to the experiments directory
cd ..

```

---

## 📖 Usage Guide

### 1. Command Line Interface (Recommended)

After installation, you can run the full analysis pipeline from any directory by simply typing:

```bash
analyze-causality

```

### 2. Programmatic Usage

If you are importing the modules into your own scripts, use the package-based imports refer `demo.py.`

---

## Visualizations Explained

### Contribution Analysis (Bar Chart)

Shows the  ratio Rflip​ for every extracted pattern.

* **Y-Axis**: Ratio Score (0.0 to 1.0).
* **Interpretation**: A score of **1.0** indicates a "Perfect Predictor"—every time this pattern appears in X, a flip occurs in Y.

### Causal Proximity Network (Graph)

A radial graph where Y is the center.

* **Aesthetics**: Closer nodes represent stronger causal influence.

---

## Maintenance

To remove build artifacts and intermediate C files:

```bash
rm -rf build/
find . -name "*.so" -delete
find . -name "*.c" -delete

```
