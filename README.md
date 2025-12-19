This repository contain datasets and scripts related to publication

TODO: add in citation format
*Deep learning accelerated multi-criteria screening of chromium-based A2+B2 superalloys across 11 elements*

Summary of the study, criteria.


Description of the files

composition_pool_generator.py
- Generates the compositions (inputs in Dataset B)
calphad_data_generator.py
- Runs CALPHAD calculations for given composition set in multiple temperatuers (outputs for Dataset B)
postprocess_and_split_dataset.py
- Postprocesses and splits the generated dataset B into B_dev and B_test as described below.

Data splitting:
After CALPHAD data is generated its postprocessed and split in postprocess_and_split_dataset.py (random_seed = 42)
Dataset_B:
	-> B_dev (85%)
	-> B_test (15%)
________________________________
Surrogate modeling:
Hyperparameter optimization:
________________________________
Using the temperature-based split, Dataset B (B_dev) is divided into (random_seed = 1234):
-> B_dev_temp (85%): used to train the DNN model after splitting into training and early stopping subsets (random_seed = 2468):
	-> B_dev_training (85%)
	-> B_dev_early_stopping (15%)
	** Random seed for tensorflow is set before creating and training the model with tf.random.set_seed(2468)
-> B_dev_validation (15%): used to evaluate the model performance with current hyperparameters and to select the best hyperparameters after the optimization completes.
________________________________
Training the final screening model using optimized hyperparameters:
________________________________
Using the temperature-based split provided in split_by_temperature.py, Dataset B (B_dev) is divided into (random_seed = 2468):
-> B_dev_training (85%)
-> B_dev_early_stopping (15%)
** Random seed for tensorflow is set before creating and training the model with tf.random.set_seed(2468)
B_test is used for evaluation after the final model is trained.

Composition preselection for CALPHAD validation is made using MiniBatchKMeans (batch_size=16384, random_seed=1234) algorithm on standardized surrogate predicted matrix compositions.
The resulting number of compositions with this approach can be slightly less than 20,000; incremental furthest search was used to pick compositions until the number of compositions reached 20,000.
These compositions were evaluated using CALPHAD and the criteria evaluation was repeated; the nominal compositions of the validated candidates that fulfill the defined criteria are provided in compositions_validated_feasible.h5
________________________________


Prerequisites
________________________________
-python==3.9.17
-Thermo-Calc installation (tested using version 2025b)
-tc_python (tc-python==2025.2.30, using Python wheel file provided with Thermo-Calc 2025b)
-scikit-learn=1.2.2
-numpy==1.26.4
-pandas=1.4.4

