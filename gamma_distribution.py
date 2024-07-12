import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as st
import astropy.constants as ast

#pl_bmasse 1239,1574,1621 are empty

plt.rcParams.update({'font.size': 8})

filename = "only_transit_data_28062023.csv"
datafile = pd.read_csv(filename, skiprows=202)

gamma_list = []
for i in range(len(datafile)-1):
    if datafile["hostname"][i] == datafile["hostname"][i+1]:
        mass1 = datafile["pl_bmasse"][i]
        mass2 = datafile["pl_bmasse"][i+1]
        gamma = np.min((mass1,mass2))/np.max((mass1,mass2))
        gamma_list.append(gamma)
    else:
        pass
gamma_array = np.array(gamma_list)

nan_places = np.argwhere(np.isnan(gamma_array))

print("Mean of mu_tilde", np.nanmean(gamma_array))
print("Standard Deviation of mu_tilde", np.nanvar(gamma_array))

plt.figure(1)
plt.hist(gamma_array, 100)
plt.xticks()
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=3)
plt.minorticks_on()
plt.xlabel(fr"$\gamma$")
plt.ylabel("Number of Exoplanet Pairs")
plt.savefig("gamma_distribution.pdf")
plt.show()

a = st.uniform.rvs(0,1,len(gamma_array))
plt.figure(2)
plt.hist(a, 100)
plt.xticks()
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=3)
plt.minorticks_on()
plt.xlabel(fr"$\gamma$")
plt.ylabel("Number of Exoplanet Pairs")
plt.show()