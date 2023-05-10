import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import astropy.constants as ast

plt.rcParams.update({'font.size': 8})

filename = "a_total.csv"
datafile = pd.read_csv(filename, skiprows=60)

el1 = 0
el2 = 0
flux_ratio_list = []
radius_list = []
mass_list = []
for i in range(len(datafile)-1):
    if datafile["discoverymethod"][i] == "Transit" or datafile["discoverymethod"][i] == "Radial Velocity":
        el1 += 1
        if pd.notna(datafile["st_teff"][i]) == True and pd.notna(datafile["st_rad"][i]) == True and pd.notna(datafile["pl_orbsmax"][i]) == True:
            el2 += 1
            flux_ratio = (datafile["st_teff"][i]/5772)**4 * (datafile["st_rad"][i])**2 * (1/datafile["pl_orbsmax"][i])**2
            # unit is solar radius^2 * au^-2
            flux_ratio_list.append(flux_ratio)
            radius_list.append(datafile["pl_rade"][i])
            mass_list.append(datafile["pl_bmasse"][i])
flux_ratio_list = np.log10(flux_ratio_list)
radius_list = np.log10(radius_list)
mass_list = np.log10(mass_list)

x1 = np.linspace(min(flux_ratio_list), max(flux_ratio_list), len(flux_ratio_list))
y1 = np.linspace(1, 1, len(flux_ratio_list))
x2 = np.linspace(min(mass_list), max(mass_list), len(mass_list))
y2 = np.linspace(1, 1, len(mass_list))

plt.figure(1,figsize=(7.2, 4.5))
plt.scatter(flux_ratio_list, mass_list, c=radius_list, cmap="hot")
plt.colorbar(label="log$_{10}(R_{pla} / R_{\U00002295}$)")
plt.xlabel("log$_{10}(F_{pla} / F_{\U00002295}$)")
plt.ylabel("log$_{10}(M_{pla} / M_{\U00002295}$)")
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=3)
plt.minorticks_on()
plt.plot(x1,y1,color="0.3", linestyle='dashed', linewidth=1)
plt.plot(y2,x2,color="0.3", linestyle='dashed', linewidth=1)
plt.show()


