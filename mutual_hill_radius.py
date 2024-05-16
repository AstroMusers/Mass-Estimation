import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update({'font.size': 8})

filename = "only_transit_data_28062023.csv"
datafile = pd.read_csv(filename, skiprows=202)

planet_masses = datafile["pl_bmasse"] 
star_masses = datafile["st_mass"] 
semi_major_axes = datafile["pl_orbsmax"] 

M_total_system_list = []
hill_list = []
for i in range(0, len(datafile)-1):
    M = 0
    index = 0
    for r in datafile["hostname"]:
        if r == datafile["hostname"][i]:
            M += datafile["pl_bmasse"][index]
        index += 1
    M += datafile["st_mass"][i] * (0.33261191609863575142462501642665*(10**6))
    if datafile["hostname"][i] == datafile["hostname"][i+1]:
        hill_radius = ((planet_masses[i] + planet_masses[i+1])/(3*M))**(1/3) * ((semi_major_axes[i+1] + semi_major_axes[i])/2)
        hill_list.append(hill_radius)
hill_data = pd.Series(hill_list)
hill_data = hill_data.dropna()
hill_data = np.log10(hill_data)

plt.figure(3, figsize=(3.5,4.5))
plt.hist(hill_data, bins=50, histtype="step", color="0.3")
plt.xlabel("$\log_{10} R_{H_{i,i+1}}$ [AU]")
plt.ylabel("Number of Exoplanet Pairs", labelpad=0.1)
plt.xlim(right=-0.5)
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=3)
plt.minorticks_on()
plt.show()



