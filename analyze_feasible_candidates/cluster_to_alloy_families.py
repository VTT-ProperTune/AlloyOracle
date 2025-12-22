import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
mpl.rcParams['font.family'] = 'Times New Roman'
          
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
       
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the Excel file
#df = pd.read_excel("results_multiple_thresholds_with_nA2phases1/compositions_and_clusters_8_all_th05_with_nA2phases1.xlsx")
df = pd.read_excel("feasible_compositions_dummy.xlsx")

n_clusters = 3
random_seed = 9 # For the K-means clustering

use_nominal_clustering = True # otherwise use A2 composition for clustering

output_ending = "nominal"

# Load data

# Extract relevant columns for clustering
#bcc1_columns = [col for col in df.columns if col.endswith("_in_A2")]

nominal_columns = [col for col in df.columns[1:12]]
print("nominal_columns:", nominal_columns)
X_nominal = df[nominal_columns].values

# Choose number of clusters (you can change this)

df_exp = pd.read_excel("experimental_compositions.xlsx", 'Sheet1')

use_renaming = False

print("df_exp.columns:", df_exp.columns)
if use_renaming:
   # --- Renaming Logic Start ---
   # Rename columns X_BCC_B2#1_<element> to <element>_in_A2
   rename_map = {}
   #for col in df_exp.columns:
   #    #if "X_BCC_B2#1_" in col:
   #    #    # Remove the prefix and add the suffix
   #    #    element = col.replace("X_BCC_B2#1_", "")
   #    #    new_name = f"{element}_in_A2"
   #    #    rename_map[col] = new_name

   if rename_map:
       df_exp.rename(columns=rename_map, inplace=True)
       print(f"Renamed {len(rename_map)} columns.")
       print("New columns:", df_exp.columns)

#for col in bcc1_columns:
#    if col not in df_exp.columns:
#       df_exp[col] = df_exp['Cr'] * 0 

for col in nominal_columns:
    if col not in df_exp.columns:
       df_exp[col] = df_exp['Cr'] * 0 

X_nominal_exp = df_exp[nominal_columns].values
  
# Fit KMeans)
kmeans = KMeans(n_clusters=n_clusters, random_state=42 + random_seed)

cluster_labels = 0
X = X_nominal
km = kmeans.fit(X_nominal) #+ 1

# Add the cluster labels to the DataFrame

df["kmeans_cluster"] = km.labels_ + 1


new_exp_clusters = kmeans.predict(X_nominal_exp) + 1

df_exp["predicted_cluster"] = new_exp_clusters

present_labels = np.unique(new_exp_clusters) 
expected_labels = np.arange(n_clusters) + 1

# Check if all expected labels are present
c = df["kmeans_cluster"].to_numpy()  # or .values
c_new = c.copy()
c_new[c == 1] = 3
c_new[c == 3] = 1
df["kmeans_cluster"] = c_new

c = df_exp["predicted_cluster"].to_numpy()  # or .values
c_new = c.copy()
c_new[c == 1] = 3
c_new[c == 3] = 1
df_exp["predicted_cluster"] = c_new

fname = f"feasible_compositions_in_{n_clusters}_alloy_clusters.xlsx"
df.to_excel( fname, index=False)
print(f"Screened compositions are clustered and saved to {fname}!")
