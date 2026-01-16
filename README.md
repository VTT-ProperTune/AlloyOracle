# **Deep Learning Accelerated Multi-Criteria Screening of Chromium-Based A2+B2 Superalloys Across 11 Elements**

This repository contains datasets and scripts to reproduce selected datasets and results of the article 'Accelerated discovery of Cr-based A2+B2 superalloys across 11 elements with a deep-learning CALPHAD surrogate'.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## Prerequisites

A `requirements.txt` file is included for installing the required Python packages (excluding `tc-python`, which must be installed after Thermo-Calc).

### System Requirements
- Python 3.9.17
- Thermo-Calc version 2025b

### Python Packages
Install the required packages using:
```
pip install -r requirements.txt
```

### Thermo-Calc Setup
1. Install Thermo-Calc (tested with version 2025b).
2. Install `tc-python` (`tc-python==2025.2.30`) using the Python wheel provided with Thermo-Calc 2025b.

> **Note:** As Thermo-Calc licenses do not permit the open release of large calculated datasets, only a 100-composition 'dummy' version of the full set of feasible compositions (roughly 15,000 compositions) is provided. Therefore, to reproduce some of the figures in the manuscript, a license to the commercial TCHEA7 database is needed.

## Repository Structure

- **`data_generation/`**: Scripts for generating and processing CALPHAD datasets
- **`surrogate_modeling/`**: Code for training surrogate models
- **`screening/`**: Screening pipeline for candidate compositions
- **`analyze_feasible_candidates/`**: Analysis and visualization of screening results
- **`analyze_experimental_compositions/`**: Analysis of experimental data
- **`helper_functions/`**: Utility functions for calculations (yield strength, VEC, misfit volumes)
- **`shared/`**: Shared utilities and common functions

## Workflow

Follow these steps to reproduce the datasets and results:

### 1. Generate Composition Pool for Generating Training Data
Run the **`data_generation/training_pool_generator.py`** to create composition pool `compositions_b.h5`.

### 2. Generate Dataset B
Run the **`data_generation/calphad_data_generator.py`** to reproduce Dataset B `dataset_b.h5` using `compositions_b.h5` as input.

**Note:** Due to CALPHAD convergence issues we iteratively checked which compositions were missing from the dataset, and repeated calculations for these compositions until all 357,096 compositions were calculated in at least one temperature. In the end, our `dataset_b.h5` included 99.98% (3,570,288) of the CALPHAD data points.

### 3. Postprocess and Split Dataset B
Rung the **`data_generation/postprocess_and_split_dataset.py`** to process and split `dataset_b.h5` into development and test sets. This creates `dataset_b_dev.h5` and `dataset_b_test.h5`.

### 4. Train Surrogate Model
Run **`surrogate_modeling/main.py`** to create the surrogate model with optimized hyperparameters. Move the generated `model/` folder to the `screening/` folder.

### 5. Generate Composition Pool
Create the composition candidate pool for screening with **`data_generation/screening_pool_generator.py`**. Move the generated `composition_pool_screening.h5` to the `screening/` folder.

### 6. Perform Screening
Execute the screening step with **`screening/main.py`** to identify feasible candidates. This produces `feasible_candidates.xlsx`.

### 7. Analyze Results
-`analyze_feasible_candidates/`: analyze the feasible compositions (alloy families, VEC-strength scatter plots).
-`analyze_experimental_compositions`: contains the scripts to visualize DFT-derived properties (strength, D-parameters).

## Usage Tips

- Ensure Thermo-Calc is properly configured before running data generation scripts.
- When working with CALPHAD data, be aware of potential convergence issues and missing data points.
- Check file paths and dependencies when moving generated files between folders

## License

This project is licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See the LICENSE file for details.

## Citation

- TBA