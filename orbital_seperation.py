import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as st
from pathlib import Path
from tdpy.verbosity import print
from pairwise_model import adjacent_mutual_hill_radii

plt.rcParams.update({'font.size': 8})

filename = Path(__file__).with_name("only_transit_data_28062023.csv")
print(f"Reading from {filename}...")
datafile = pd.read_csv(filename, skiprows=202)

### K Calculation ###

semi_major_axes = datafile["pl_orbsmax"]  # [AU]

pair_positions, hill_radii = adjacent_mutual_hill_radii(datafile)
axis_gaps = (
    semi_major_axes.iloc[pair_positions + 1].to_numpy()
    - semi_major_axes.iloc[pair_positions].to_numpy()
)
K_data = pd.Series(axis_gaps / hill_radii)
K_data = K_data[K_data > 0]
logK = np.log10(K_data)

print("Mean of logK of NASA is", np.mean(logK), "and standard deviation is", np.std(logK))

res1 = st.shapiro(logK)
print("Normality test logK result for dataset is", res1.statistic)

### End ###

### Figure ###

plt.figure(2, figsize=(3.5, 4.5))
plt.hist(logK, density=True, bins=50, histtype="step", color="red")
plt.xlabel("$log_{10}K$")
plt.ylabel("Number of Exoplanet Pairs")
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=3)
plt.minorticks_on()
plt.tight_layout()
plt.show()

### End ###

