import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update({'font.size': 8})

filename = "a_total.csv"
datafile = pd.read_csv(filename, skiprows=60)

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

planet_masses_list = []
for i in range (0,len(datafile)-1):
    if datafile["pl_rade"][i]<1.5:
        m = (0.441 + 0.615*datafile["pl_rade"][i])*(datafile["pl_rade"][i])**3
    elif 1.5<datafile["pl_rade"][i]<4:
        m = 2.69*(datafile["pl_rade"][i])**(0.93)
    else:
        m = 3*(datafile["pl_rade"][i])
    planet_masses_list.append(m)
planet_masses = pd.Series(planet_masses_list)

K12_list = []
for i in range(0, len(datafile)-2):
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
        K12_list.append(K)
K12_data = pd.Series(K12_list)
K12_data = K12_data[K12_data > 0]
logK12 = np.log10(K12_data)

plt.figure(2, figsize=(3.5,4.5))
plt.hist(logK, density=True, bins=50, histtype="step", color="red")
plt.hist(logK12, density=True, bins=50, histtype="step", color="blue")
plt.xlabel("logK")
plt.ylabel("Number of Exoplanet Pairs")
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=3)
plt.minorticks_on()
plt.show()

