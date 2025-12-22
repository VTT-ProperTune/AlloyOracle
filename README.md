# **Deep Learning Accelerated Multi-Criteria Screening of Chromium-Based A2+B2 Superalloys Across 11 Elements**

This repository contains datasets and scripts to reproduce selected datasets and results of the article.

---

## **TODO**
- Add license information.
- Add citation information.

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

## **Description of the scripts**  
*(Run the scripts in this order to create datasets `B_dev` and `B_test`)*

- **`composition_pool_generator.py`**: Generates the compositions (inputs for Dataset B).
  
- **`calphad_data_generator.py`**: Runs CALPHAD calculations for the given composition set at multiple temperatures using asynchronous parallel processing (outputs for Dataset B).

- **`postprocess_and_split_dataset.py`**: Postprocesses and splits the generated Dataset B into `B_dev` and `B_test` as described below.

## **Description of the datasets**

- **`dataset_a_dev_compositions.h5`**: Compositions of dataset A_dev.

- **`dataset_a_test_compositions.h5`**: Compositions of dataset A_test.

- **`compositions_validated_feasible.h5`**: Compositions of the criteria-fulfilling alloys.

---

## **Surrogate Modeling**

### **Hyperparameter Optimization**
Using `postprocess_and_split_dataset.split_by_temperature_ratio` method, `Dataset B (B_dev)` is divided into (with `random_seed = 1234`):

- **`B_dev_temp` (85%)**  
  Used to train the DNN model after splitting into training and early stopping subsets (`random_seed = 2468`):  
  → `B_dev_training` (85%)  
  → `B_dev_early_stopping` (15%)  
  *Random seed for TensorFlow is set before training with `tf.random.set_seed(2468)`.*

- **`B_dev_validation` (15%)**  
  Used to evaluate model performance and select the best hyperparameters.

---

### **Training the Final Screening Model**
Using the temperature-based split in `split_by_temperature.py`, `Dataset B (B_dev)` is divided into (with `random_seed = 2468`):  
→ `B_dev_training` (85%)  
→ `B_dev_early_stopping` (15%)  
*TensorFlow random seed set with `tf.random.set_seed(2468)`.*  
`B_test` is used for final evaluation.

---

### **Composition Preselection for CALPHAD Validation**
- Performed using **MiniBatchKMeans** (`batch_size = 16384`, `random_seed = 1234`) on standardized surrogate-predicted compositions.
- If fewer than 20,000 compositions are selected, **incremental furthest search** is applied until the target count is reached.
- These compositions are evaluated using CALPHAD, and criteria evaluation is repeated.
- Nominal compositions of validated candidates fulfilling the defined criteria are provided in:  
  **`compositions_validated_feasible.h5`**

---

## **License**
- TBA

---

## **Citation**
- TBA
