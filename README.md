# **Deep Learning Accelerated Multi-Criteria Screening of Chromium-Based A2+B2 Superalloys Across 11 Elements**

This repository contains datasets and scripts to reproduce selected datasets and results of the article 'Accelerated discovery of Cr-based A2+B2 superalloys across 11 elements with a deep-learning CALPHAD surrogate'.

---

## **Prerequisites**

A `requirements.txt` file is included for installing the required Python packages (excluding `tc-python`, which must be installed after Thermo-Calc).

1. **Install Thermo-Calc** (tested with version **2025b**).
2. **Install Python packages** from `requirements.txt` (Python **3.9.17**).
3. **Install `tc_python`** (`tc-python==2025.2.30`) using the Python wheel provided with Thermo-Calc 2025b.

> Note: As Thermo-Calc license do not permit the open release of large calculated datasets, 
> only a 100-composition 'dummy' version of the full set of feasible compositions (roughly 15,000 compositions).
> Therefore, to reproduce some of the figures in the manuscript, a license to the commercial TCHEA7 database is needed.

---

## **Description of folders**
- **`generate_surrogate_model`**: contains the scripts to generate Thermo-Calc data and the CALPHAD surrogate model.
- **`analyze_feasible_compositions`**: contains the scripts to analyze the feasible compositions (alloy families, VEC-strength scatter plots).
- **`analyze_experimental_compositions`**: contains the scripts to visualize DFT-derived properties (strength, D-parameters).
- **`helper_functions`**: functions to calculate edge-slip disclocation strength, VEC, and misfit volumes.

---

## **License**
This project is licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See the LICENSE file.

---

## **Citation**
- TBA
