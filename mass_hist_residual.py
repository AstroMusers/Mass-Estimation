import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats as scp
import astropy.constants as const
import matplotlib.patches as ptc

plt.rcParams.update({'font.size': 8})

filename = "only_transit_data_28062023.csv"
datafile = pd.read_csv(filename, skiprows=177)

logD_dataframe = pd.read_csv("logD.csv")
logD_series = logD_dataframe.squeeze()
logD = logD_series.values.tolist()

#K-calculation
logK_a = scp.norm.rvs(size=(len(logD),1000), loc=1.32, scale=0.31)
logK = np.mean(logK_a, axis=1)

#gamma-calculation
gamma = []
a = 0
while a != len(logD):
    calc_data_gamma = scp.norm.rvs(loc=1, scale=0.3)
    if 0 < calc_data_gamma < 1:
        gamma.append(calc_data_gamma)
        a += 1

#thiswork work - mass tilde
mu_tilde = []
for i in range(len(logD)):
    mu_log = (3*(logD[i] - logK[i])) + np.log10(3)
    mu_data = 10**mu_log
    mu_tilde.append(mu_data)
mu_tilde_data = pd.Series(mu_tilde)

planet_list = []
for i in range(len(logD)):
    planet_min_mu = gamma[i] * (1+gamma[i])**(-1) * mu_tilde[i]
    planet_max_mu = (1+gamma[i])**(-1) * mu_tilde[i]
    planet_list.append(planet_min_mu)
    planet_list.append(planet_max_mu)
planet_list.append(np.mean(planet_list))

planet_mass = []
for i in range(len(datafile)-1):
    mass = (planet_list[i]*datafile["st_mass"][i]*const.M_sun.value)/const.M_earth.value
    planet_mass.append(mass)
planet_mass_log = np.log10(planet_mass)
planet_mass_log = pd.Series(planet_mass_log)
planet_mass_log = planet_mass_log.dropna()

#Otegi et al. mass
t_mu_data = []
t_mass_data = []
for i in range(len(datafile)-1):
    if datafile["pl_dens"][i] > 3.3:
        mass = 0.9*((datafile["pl_rade"][i])**3.45)
    else:
        mass = 1.74*((datafile["pl_rade"][i])**1.58)
    mu = mass/(datafile["st_mass"][i])
    t_mass_data.append(mass)
    t_mu_data.append(mu)
t_mu_series = pd.Series(t_mu_data)
t_mu_log = np.log10(t_mu_series)
t_mu_log = t_mu_log.dropna()
t_mass_log = np.log10(t_mass_data)
t_mass_log = pd.Series(t_mass_log)
t_mass_log = t_mass_log.dropna()

#NEA mass data
e_mu_data = []
e_mass_data = []
for i in range(len(datafile)-1):
    mass = datafile["pl_bmasse"][i]
    e_mu = (datafile["pl_bmasse"][i])/(datafile["st_mass"][i])
    e_mu_data.append(e_mu)
    e_mass_data.append(mass)
e_mu_series = pd.Series(e_mu_data)
e_mu_log = np.log10(e_mu_series)
e_mu_log = e_mu_log.dropna()
e_mass_log = np.log10(e_mass_data)
e_mass_log = pd.Series(e_mass_log)
e_mass_log = e_mass_log.dropna()

#malhotra mass data
malhotra_log_m_msun_data = scp.norm.rvs(size=(174500), loc=0.64, scale=1.21)
malhotra_log_m_msun_data2 = scp.norm.rvs(size=(1745), loc=0.64, scale=1.21)
#malhotra_log_m_msun_data = np.mean(malhotra_log_m_msun_data, axis=1)

diff_data_otegi = []
diff_data_thiswork = []
diff_data_malhotra = []
for i in range(len(e_mu_series)):
    diff_otegi = t_mass_data[i] - e_mass_data[i]
    diff_thiswork = planet_mass[i] - e_mass_data[i]
    diff_malhotra = malhotra_log_m_msun_data[i] - e_mass_data[i]
    diff_data_otegi.append(diff_otegi)
    diff_data_thiswork.append(diff_thiswork)
    diff_data_malhotra.append(diff_malhotra)
diff_series_otegi = pd.Series(diff_data_otegi)
diff_series_thiswork = pd.Series(diff_data_thiswork)
diff_series_malhotra = pd.Series(diff_data_malhotra)


plt.figure(1, figsize=(7.2, 4.5))
#nasa exoplanet archive
plt.hist(e_mass_log, color="0.3", label="NASA Exoplanet Archive, 2023", bins=100, histtype="step", density=True)
#this work
plt.hist(planet_mass_log, label="This Work", color="lime", bins=100, histtype="step", density=True)
#otegi
plt.hist(t_mass_log, color="blue", label="Otegi et al. 2020", bins=100, histtype="step", density=True)
#malhotra
plt.hist(malhotra_log_m_msun_data, color="red", label="Malhotra, 2015", bins=100, histtype="step", density=True)
plt.hist(malhotra_log_m_msun_data2, bins=100, histtype="step", density=True)
plt.legend()
plt.xlabel("log$_{10}$ $M/M_{\oplus}$")
plt.ylabel("PDF(log$_{10}$ $M/M_{\oplus}$)")
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=4)
plt.minorticks_on()

fig = plt.figure(2, figsize=(7.2, 4.5))
sub2 = plt.subplot(2, 1, 1)
sub2.scatter(e_mu_series, diff_series_otegi, marker="o", label="Otegi et al., 2020", color="blue", s=4)
sub2.scatter(e_mu_series, diff_series_thiswork, marker="^", label="This Work", color="lime", s=4)
sub2.scatter(e_mu_series, diff_series_malhotra,marker="x", label="y = Malhotra, 2015", color="red", s=4)
sub2.set_xlim(-1, 20)
sub2.set_ylim(-10, 10)

sub3 = plt.subplot(2, 1, 2)
sub3.scatter(e_mu_series, diff_series_otegi, marker="o", label="y = Otegi et al. 2020 ", color="blue", s=4)
sub3.scatter(e_mu_series, diff_series_thiswork, marker="^", label="y = This Work", color="lime", s=4)
sub3.scatter(e_mu_series, diff_series_malhotra,marker="x", label="y = Malhotra, 2015", color="red", s=4)
sub3.set_xlim(-3,100)
sub3.set_ylim(-100,75)
sub3.fill_between((-1, 20), -10, 10, facecolor='orange', alpha=0.2)
con1 = ptc.ConnectionPatch(xyA=(-1, -10), coordsA=sub2.transData, xyB=(-1, 0), coordsB=sub3.transData, color='orange')
fig.add_artist(con1)
con2 = ptc.ConnectionPatch(xyA=(20, -10), coordsA=sub2.transData, xyB=(20, 0), coordsB=sub3.transData, color='orange')
fig.add_artist(con2)
fig.supylabel("Residual of x and y")
fig.supxlabel("x = NASA Exoplanet Archive, 2023")
plt.legend()

plt.figure(3, figsize=(7.2, 4.5))
plt.scatter(e_mu_series, diff_series_malhotra,marker="o", label="y = Malhotra, 2015", color="blue", s=4)
plt.ylabel("Residual of x and y")
plt.xlabel("x = NASA Exoplanet Archive, 2023 \u03BC$_{i}$")
plt.legend()
plt.show()
