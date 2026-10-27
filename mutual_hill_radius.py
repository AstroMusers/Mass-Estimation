import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from tdpy.verbosity import print
from pairwise_model import adjacent_mutual_hill_radii

plt.rcParams.update({'font.size': 8})

filename = Path(__file__).with_name("only_transit_data_28062023.csv")
print(f"Reading from {filename}...")
datafile = pd.read_csv(filename, skiprows=202)

_, hill_radii = adjacent_mutual_hill_radii(datafile)
hill_data = pd.Series(hill_radii)
hill_data = hill_data.dropna()
hill_data = np.log10(hill_data)

plt.figure(3, figsize=(3.5,4.5))
plt.hist(hill_data, bins=50, histtype="step", color="0.3")
plt.xlabel(r"$\log_{10} R_{H_{i,i+1}}$ [AU]")
plt.ylabel("Number of Exoplanet Pairs", labelpad=0.1)
plt.xlim(right=-0.5)
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=3)
plt.minorticks_on()
plt.show()



