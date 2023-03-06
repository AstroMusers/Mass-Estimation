import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns

plt.rcParams.update({'font.size': 8})

filename = "a_total.csv"
datafile = pd.read_csv(filename, skiprows=60)

d_list = []
for i in range(0,len(datafile)-1):
    if datafile["disc_instrument"][i] == "TESS CCD Array" or datafile["disc_instrument"][i] == "Kepler CCD Array":
        if datafile["hostname"][i] == datafile["hostname"][i+1]:
            P_i = datafile["pl_orbper"][i+1]/datafile["pl_orbper"][i]
            D = 2*((P_i**(2/3) - 1)/(P_i**(2/3)+1))
            d_list.append(D)
d_data = pd.Series(d_list)

wanted_data = d_data[d_data > 0]
w = np.log(wanted_data)

[mean, std] = stats.norm.fit(w)
x = np.linspace(np.min(w), np.max(w), 871)

plt.figure(1, figsize=(3.5, 4.5))
plt.hist(w, density=True, bins=50, histtype="step", color="0.3")
plt.plot(x, stats.norm.pdf(x, mean, std), color="0.2", linestyle="dashdot", linewidth=1)
plt.xlabel("Orbital Spacing")
plt.ylabel("Density")
plt.xlim(-2.5, 1)
plt.xticks(range(-2,2))
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=3)
plt.minorticks_on()
plt.show()