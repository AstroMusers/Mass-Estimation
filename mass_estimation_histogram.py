import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats as scp
import astropy.constants as const

plt.rcParams.update({'font.size': 8})

filename = "only_transit_data_28062023.csv"
datafile = pd.read_csv(filename, skiprows=177)

b = 0
masses_our_work = [0] * 1745
while b != 1000:

    #K-calculation
    logK = scp.norm.rvs(size=872, loc=1.32, scale=0.31)

    #gamma-calculation
    gamma = []
    a = 0
    while a != 872:
        calc_data_gamma = scp.uniform.rvs()
        if 0 < calc_data_gamma < 1:
            gamma.append(calc_data_gamma)
            a += 1

    logD_dataframe = pd.read_csv("logD.csv")
    logD_series = logD_dataframe.squeeze()
    logD = logD_series.values.tolist()

    #our work - mass tilde
    mu_tilde = []
    for i in range(len(logD)):
        mu_log = (3*(logD[i] - logK[i])) + np.log10(3)
        mu_data = 10**mu_log
        mu_tilde.append(mu_data)
    mu_tilde_data = pd.Series(mu_tilde)
    mu_tilde_log = np.log10(mu_tilde_data)
    mu_tilde_log = mu_tilde_log.dropna()

    planet_list = []
    for i in range(len(logD)):
        planet_min_mu = gamma[i] * (1+gamma[i])**(-1) * mu_tilde[i]
        planet_max_mu = (1+gamma[i])**(-1) * mu_tilde[i]
        planet_list.append(planet_min_mu)
        planet_list.append(planet_max_mu)
    planet_list.append(np.mean(planet_list))
    planet_list_log = np.log10(planet_list)
    planet_list_log = pd.Series(planet_list_log)
    planet_list_log = planet_list_log.dropna()

    planet_mass = []
    for i in range(len(datafile)-1):
        mass = (planet_list[i]*datafile["st_mass"][i]*const.M_sun.value)/const.M_earth.value
        planet_mass.append(mass)
    masses_our_work = [sum(x) for x in zip(masses_our_work, planet_mass)]
    b += 1

final_masses = []
for i in range(len(masses_our_work)):
    final_masses.append(masses_our_work[i]/1000)
final_masses = np.log10(final_masses)
final_masses = pd.Series(final_masses)
final_masses = final_masses.dropna()

# Otegi et al. mass
t_mu_data = []
t_mass_data = []
for i in range(len(datafile) - 1):
    if datafile["pl_dens"][i] > 3.3:
        mass = 0.9 * ((datafile["pl_rade"][i]) ** 3.45)
    else:
        mass = 1.74 * ((datafile["pl_rade"][i]) ** 1.58)
    mu = mass / (datafile["st_mass"][i])
    t_mass_data.append(mass)
    t_mu_data.append(mu)
t_mass_log = np.log10(t_mass_data)
t_mass_log = pd.Series(t_mass_log)
t_mass_log = t_mass_log.dropna()

# NEA mass data
e_mu_data = []
e_mass_data = []
for i in range(len(datafile) - 1):
    mass = datafile["pl_bmasse"][i]
    e_mu = (datafile["pl_bmasse"][i]) / (datafile["st_mass"][i])
    e_mu_data.append(e_mu)
    e_mass_data.append(mass)
e_mass_log = np.log10(e_mass_data)
e_mass_log = pd.Series(e_mass_log)
e_mass_log = e_mass_log.dropna()

plt.figure(1, figsize=(7.2, 4.5))
# experimental mass plot
plt.hist(e_mass_log, color="red", label="NASA Exoplanet Archive, 2023", histtype="step", bins=50)
# our mass plot
plt.hist(final_masses, label="This Work", color="lime", histtype="step", bins=50)
# otegi mass plot
plt.hist(t_mass_log, color="blue", label="Otegi et al. 2020", histtype="step", bins=50)
plt.legend()
plt.xlabel("$log_{10}M/M_{\oplus}$")
plt.ylabel("Number of Planets")
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=4)
plt.minorticks_on()
plt.show()

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
plt.xlabel("$log_{10}M/M_{\oplus}$")
plt.ylabel("Number of Planets")
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=4)
plt.minorticks_on()
plt.show()
