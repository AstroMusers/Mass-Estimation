import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import astropy.constants as const
import astropy.units as u
from scipy import stats

plt.rcParams.update({'font.size': 8})

filename = "a_total.csv"
datafile = pd.read_csv(filename, skiprows=60)

planet_masses = datafile["pl_bmasse"] 
star_masses = datafile["st_mass"] 
semi_major_axes = datafile["pl_orbsmax"] 

M_total_system_list = []
hill_list = []
for i in range(0,len(datafile)-1):
    M = 0
    index = 0
    for r in datafile["hostname"]:
        if r == datafile["hostname"][i]:
            M += datafile["pl_bmasse"][index]
        index += 1
    M += datafile["st_mass"][i]* (0.33261191609863575142462501642665*(10**(6)))
    if datafile["hostname"][i] == datafile["hostname"][i+1]:
        hill_radius = ((planet_masses[i] + planet_masses[i+1])/(3*M))**(1/3) * ((semi_major_axes[i+1] + semi_major_axes[i])/2)
        hill_list.append(hill_radius)
hill_data = pd.Series(hill_list)
hill_data = hill_data.dropna()
hill_data = np.log(hill_data)

plt.figure(3, figsize=(3.5,4.5))
plt.hist(hill_data, bins=100, histtype="step", color="grey")
plt.xlabel("Mutual Hill Sphere Radius (AU)")
plt.ylabel("Density")
plt.xlim(right=0)
plt.show()



