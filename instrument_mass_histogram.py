import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats as scp
import astropy.constants as const
from pathlib import Path
from tdpy.verbosity import print
from pairwise_model import log_mu_tilde_from_log_K, pair_mass_fractions

plt.rcParams.update({'font.size': 8})

filename = Path(__file__).with_name("only_transit_data_28062023.csv")
print(f"Reading from {filename}...")
datafile = pd.read_csv(filename, skiprows=202)

b = 0
masses_our_work = [0] * 1745
logD_filename = Path(__file__).with_name("logD.csv")
print(f"Reading from {logD_filename}...")
logD_dataframe = pd.read_csv(logD_filename)
logD_series = logD_dataframe.squeeze()
logD = logD_series.values.tolist()
while b != 1000:

    #K-calculation
    logK = scp.norm.rvs(size=len(logD), loc=1.32, scale=0.31)

    #gamma-calculation
    gamma = []
    a = 0
    while a != len(logD):
        calc_data_gamma = scp.uniform.rvs()
        if 0 < calc_data_gamma < 1:
            gamma.append(calc_data_gamma)
            a += 1

    #our work - mass tilde
    mu_tilde = []
    for i in range(len(logD)):
        mu_log = log_mu_tilde_from_log_K(logD[i], logK[i])
        mu_data = 10**mu_log
        mu_tilde.append(mu_data)
    mu_tilde_data = pd.Series(mu_tilde)
    mu_tilde_log = np.log10(mu_tilde_data)
    mu_tilde_log = mu_tilde_log.dropna()

    minimum_fractions, maximum_fractions = pair_mass_fractions(gamma)
    planet_list = np.column_stack((minimum_fractions, maximum_fractions)).ravel().tolist()
    planet_list.append(np.mean(planet_list))
    planet_list_log = np.log10(planet_list)
    planet_list_log = pd.Series(planet_list_log)
    planet_list_log = planet_list_log.dropna()

    planet_mass = []
    for i in range(len(datafile)-1):
        mass = (planet_list[i]*datafile["st_mass"][i]*const.M_sun.value)/const.M_earth.value  # [M_earth]
        planet_mass.append(mass)
    masses_our_work = [sum(x) for x in zip(masses_our_work, planet_mass)]
    b += 1

final_masses = []
for i in range(len(masses_our_work)):
    final_masses.append(masses_our_work[i]/1000)
final_masses = np.log10(final_masses)
final_masses = pd.Series(final_masses)
final_masses = final_masses.dropna()

# NEA mass data
e_mu_data = []
e_mass_data = []
for i in range(len(datafile) - 1):
    mass = datafile["pl_bmasse"][i]  # [M_earth]
    e_mu = (datafile["pl_bmasse"][i]) / (datafile["st_mass"][i])
    e_mu_data.append(e_mu)
    e_mass_data.append(mass)
e_mass_log = np.log10(e_mass_data)
e_mass_log = pd.Series(e_mass_log)
e_mass_log = e_mass_log.dropna()

datafile = datafile.fillna(0)  # some planets do not have mass values, they are replaced with 0s

# classsification of masses wrt discovery facility and discovery method, t=transiting, nt=nontransiting
t_tess = []
t_kepler = []
t_other = []
t_total = []
nt_tess = []
nt_kepler = []
nt_other = []
nt_total = []
nan_count = 0
for i in range(len(datafile) - 1):
    if datafile["discoverymethod"][i] == "Transit":
        if datafile["disc_facility"][i] == "Transiting Exoplanet Survey Satellite (TESS)" and datafile["pl_bmasse"][
                i] != 0:
            t_tess.append(datafile["pl_bmasse"][i])
            t_total.append(datafile["pl_bmasse"][i])
        elif datafile["disc_facility"][i] == "Kepler" and datafile["pl_bmasse"][i] != 0:
            t_kepler.append(datafile["pl_bmasse"][i])
            t_total.append(datafile["pl_bmasse"][i])
        else:
            if datafile["pl_bmasse"][i] != 0:
                t_other.append(datafile["pl_bmasse"][i])
                t_total.append(datafile["pl_bmasse"][i])
    else:
        if datafile["disc_facility"][i] == "Transiting Exoplanet Survey Satellite (TESS)" and datafile["pl_bmasse"][
                i] != 0:
            nt_tess.append(datafile["pl_bmasse"][i])
            nt_total.append(datafile["pl_bmasse"][i])
        elif datafile["disc_facility"][i] == "Kepler" and datafile["pl_bmasse"][i] != 0:
            nt_kepler.append(datafile["pl_bmasse"][i])
            nt_total.append(datafile["pl_bmasse"][i])
        else:
            if datafile["pl_bmasse"][i] != 0:
                nt_other.append(datafile["pl_bmasse"][i])
                nt_total.append(datafile["pl_bmasse"][i])
    if datafile["pl_bmasse"][i] == 0:
        nan_count += 1
t_tess_log = np.log10(t_tess)
t_kepler_log = np.log10(t_kepler)
t_other_log = np.log10(t_other)
t_total_log = np.log10(t_total)
nt_tess_log = np.log10(nt_tess)
nt_kepler_log = np.log10(nt_kepler)
nt_other_log = np.log10(nt_other)
nt_total_log = np.log10(nt_total)

plt.figure(1, figsize=(7.2, 4.5))
plt.hist(t_tess_log, color="orange", label="Transiting - TESS", histtype="step", bins=50)
plt.hist(t_kepler_log, color="violet", label="Transiting - Kepler", histtype="step", bins=50)
plt.hist(t_other_log, color="blue", label="Transiting - Other", histtype="step", bins=50)
plt.hist(t_total_log, color="black", label="Transiting - Total", histtype="step", bins=50)
plt.hist(nt_total_log, color="red", label="Non-transiting - Total", histtype="step", bins=50)
plt.hist(final_masses, color="lime", label="Our Work", histtype="step", bins=50)
plt.legend()
plt.xlabel(r"$log_{10}M/M_{\oplus}$")
plt.ylabel("Number of Planets")
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=4)
plt.minorticks_on()
plt.show()
