import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as st
import astropy.constants as ast

#pl_bmasse 1239,1574,1621 are empty

plt.rcParams.update({'font.size': 8})

filename = "only_transit_data_28062023.csv"
datafile = pd.read_csv(filename, skiprows=202)

mu_tilde_list = []
for i in range(len(datafile)-1):
    if datafile["hostname"][i] == datafile["hostname"][i+1]:
        mass_sum = (datafile["pl_bmasse"][i] + datafile["pl_bmasse"][i+1]) * ast.M_earth.value #in kg unit
        star_mass = datafile["st_mass"][i] * ast.M_sun.value #in kg unit
        mu_tilde = mass_sum/star_mass #unitless
        mu_tilde_list.append(mu_tilde)
    else:
        pass
mu_tilde_array = np.array(mu_tilde_list)
log10_mu_tilde_array = np.log10(mu_tilde_array)

nan_places = np.argwhere(np.isnan(mu_tilde_array))

print("Mean of mu_tilde", np.nanmean(log10_mu_tilde_array))
print("Standard Deviation of mu_tilde", np.nanvar(log10_mu_tilde_array))

plt.figure("mu_tilde_distribution")
plt.hist(log10_mu_tilde_array, 100)
plt.xticks()
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=3)
plt.minorticks_on()
plt.xlabel("log10_mu_tilde")
plt.ylabel("Number of Exoplanet Pairs")
plt.show()
