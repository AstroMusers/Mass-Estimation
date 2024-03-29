import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update({'font.size': 8})

filename = "only_transit_data_28062023.csv"
datafile = pd.read_csv(filename, skiprows=177)

d_list = []
for i in range(0,len(datafile)-1):
    if datafile["disc_instrument"][i] == "TESS CCD Array" or datafile["disc_instrument"][i] == "Kepler CCD Array":
        if datafile["hostname"][i] == datafile["hostname"][i+1]:
            P_i = datafile["pl_orbper"][i+1]/datafile["pl_orbper"][i]
            D = 2*((P_i**(2/3) - 1)/(P_i**(2/3)+1))
            d_list.append(D)
d_data = pd.Series(d_list)

wanted_data = d_data[d_data > 0]
w = np.log10(wanted_data)

plt.figure(1, figsize=(3.5, 4.5))
plt.hist(w, bins=50, histtype="step", color="0.3")
plt.xlabel("$log_{10}D$")
plt.ylabel("Number of Exoplanet Pairs")
plt.xlim(-1.1, 0.5)
plt.xticks()
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=3)
plt.minorticks_on()
plt.show()