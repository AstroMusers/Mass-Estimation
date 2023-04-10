import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns

plt.rcParams.update({'font.size': 8})

filename = "a_total.csv"
datafile = pd.read_csv(filename, skiprows=60)

d_list = []
passed = 0
for i in range(len(datafile)-1):
    if datafile["hostname"][i] == datafile["hostname"][i+1]:
        P_i = datafile["pl_orbper"][i+1]/datafile["pl_orbper"][i]
        D = 2*((P_i**(2/3) - 1)/(P_i**(2/3)+1))
        d_list.append(D)
    else:
        passed += 1
d_data = pd.Series(d_list)
wanted_data = d_data[d_data > 0]
w = np.log10(wanted_data)

w.to_csv("logD.csv", index=False, header=False)

plt.figure(1, figsize=(3.5, 4.5))
plt.hist(w, bins=50, histtype="step", color="0.3")
plt.xlabel("logD")
plt.ylabel("Number of Exoplanet Pairs")
plt.xlim(-1.1, 0.5)
plt.xticks()
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=3)
plt.minorticks_on()
plt.show()

print(f"{passed} lines in the a_total.csv has been passed")

