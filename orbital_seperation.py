import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as st

plt.rcParams.update({'font.size': 8})

filename = "only_transit_data_28062023.csv"
datafile = pd.read_csv(filename, skiprows=202)

### K Calculation ###

planet_masses = datafile["pl_bmasse"] 
star_masses = datafile["st_mass"] 
semi_major_axes = datafile["pl_orbsmax"] 

K_list = []
for i in range(0, len(datafile)-1):
    M = 0
    index = 0
    for r in datafile["hostname"]:
        if r == datafile["hostname"][i]:
            M += datafile["pl_bmasse"][index]
        index += 1
    M += datafile["st_mass"][i]* (0.33261191609863575142462501642665*(10**6))
    if datafile["hostname"][i] == datafile["hostname"][i+1]:
        hill_radius = ((planet_masses[i] + planet_masses[i+1])/(3*M))**(1/3) * ((semi_major_axes[i+1] + semi_major_axes[i])/2)
        K = (semi_major_axes[i+1] - semi_major_axes[i])/hill_radius
        K_list.append(K)
K_data = pd.Series(K_list)
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

