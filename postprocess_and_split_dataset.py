"""
Author: Mikko Tahkola mikko.tahkola@vtt.fi, VTT 2025
"""

import numpy as np
import pandas as pd
from postprocess_outputs import postprocess_calphad_data
   
 
def split_by_temperature_ratio(df, temp_col='T_input', ratio=0.85, random_state=42):
    """
    Splits the DataFrame into two subsets based on the given ratio,
    preserving the distribution of temperature groups.
    
    Parameters:
    - df: pandas DataFrame
    - temp_col: column name for temperature grouping
    - ratio: fraction for the first subset (e.g., 0.8 for 80%)
    - random_state: seed for reproducibility
    
    Returns:
    - subset1 (ratio portion), subset2 (remaining portion)
    """
    np.random.seed(random_state)
    subset1 = pd.DataFrame()
    subset2 = pd.DataFrame()
    
    # Group by temperature
    for _, group in df.groupby(temp_col):
        # Shuffle group
        shuffled = group.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Compute split index
        split_idx = int(len(shuffled) * ratio)
        
        # Split into two parts
        part1 = shuffled.iloc[:split_idx]
        part2 = shuffled.iloc[split_idx:]
        
        # Append to subsets
        subset1 = pd.concat([subset1, part1], ignore_index=True)
        subset2 = pd.concat([subset2, part2], ignore_index=True)
    
    return subset1, subset2  

def main():
    ############################################
    
    # Configuration
    config = {}
    dev_split = 0.85
    config['element_labels'] = ["Cr","Al","Fe","Ni","Ti","V","Mo","W","Mn","Si","Co"]
    config['inputs'] = ['T_input'] + config['element_labels']
    config['outputs'] = ["LIQUID", "A2", "B2", "OTHER_PHASES"]
    for el in config['element_labels']:
        config['outputs'].append(f'{el}_in_A2')
    config['tc'] = {
        'tc_phases': ['LIQUID', 'BCC_B2#1', 'BCC_B2#2', 'BCC_B2#3', 'BCC_B2#4', 'OTHER_PHASES'],
        'bcc_site_fraction_absdiff_cutoff': 1e-3,
        'max_b2_index': 4
        }
      
    ############################################
    # Load the CALPHAD dataset
    df = pd.read_hdf("data_compositions_dataset_b.h5", key='df')
    df = df.drop_duplicates()
    
    ############################################
    # Post-process dataset - calculate A2 and B2 phase fractions and A2 composition
    df_postprocessed = postprocess_calphad_data(df, config['element_labels'], config['tc']['tc_phases'], config['tc']['bcc_site_fraction_absdiff_cutoff'])
    for col in df_postprocessed:
        if col not in df:
            df[col] = df_postprocessed[col]
    
    # For the surrogate modeling, we only need the defined input and output variables
    df = df[config['inputs']+config['outputs']]
    
    ############################################
    # Split the dataset
    dev_split = 0.85 # Percentage of data to split to development dataset
    df_dev, df_test = split_by_temperature_ratio(df, temp_col='T_input', ratio=dev_split)
    
    df_dev = df_dev.drop_duplicates()
    df_test = df_test.drop_duplicates()
    
    print(f'Development dataset size: {df_dev.shape[0]} samples {(df_dev.shape[0]/df.shape[0]):.2%}')
    print(f'Test dataset size: {df_test.shape[0]} samples {(df_test.shape[0]/df.shape[0]):.2%}')
    
    df_dev.to_hdf('dataset_b_dev.h5', key='df', mode='w')
    df_test.to_hdf('dataset_b_test.h5', key='df', mode='w')
    
if __name__=="__main__":
    main()
    
# Configuration
config = {}
dev_split = 0.85
config['element_labels'] = ["Cr","Al","Fe","Ni","Ti","V","Mo","W","Mn","Si","Co"]
config['inputs'] = ['T_input'] + config['element_labels']
config['outputs'] = ["LIQUID", "A2", "B2", "OTHER_PHASES"]
for el in config['element_labels']:
    config['outputs'].append(f'{el}_in_A2')
config['tc'] = {
    'tc_phases': ['LIQUID', 'BCC_B2#1', 'BCC_B2#2', 'BCC_B2#3', 'BCC_B2#4', 'OTHER_PHASES'],
    'bcc_site_fraction_absdiff_cutoff': 1e-3,
    'max_b2_index': 4
    }
  
############################################
# Load the CALPHAD dataset
df = pd.read_hdf("data_compositions_dataset_b.h5", key='df')
df = df.drop_duplicates()

############################################
# Post-process dataset - calculate A2 and B2 phase fractions and A2 composition
df_postprocessed = postprocess_calphad_data(df, config['element_labels'], config['tc']['tc_phases'], config['tc']['bcc_site_fraction_absdiff_cutoff'])
for col in df_postprocessed:
    if col not in df:
        df[col] = df_postprocessed[col]

# For the surrogate modeling, we only need the defined input and output variables
df = df[config['inputs']+config['outputs']]

############################################
# Split the dataset
dev_split = 0.85 # Percentage of data to split to development dataset
df_dev, df_test = split_by_temperature_ratio(df, temp_col='T_input', ratio=dev_split)

df_dev = df_dev.drop_duplicates()
df_test = df_test.drop_duplicates()

print(f'Development dataset size: {df_dev.shape[0]} samples {(df_dev.shape[0]/df.shape[0]):.2%}')
print(f'Test dataset size: {df_test.shape[0]} samples {(df_test.shape[0]/df.shape[0]):.2%}')

df_dev.to_hdf('dataset_b_dev.h5', key='df', mode='w')
df_test.to_hdf('dataset_b_test.h5', key='df', mode='w')