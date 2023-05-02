import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats as scp
import astropy.constants as const
import matplotlib.patches as ptc

plt.rcParams.update({'font.size': 8})

#K-calculation
K = scp.norm.rvs(size=1128, loc=1.32, scale=0.31)
K_log = np.log10(K)

#gamma-calculation
gamma = []
a = 0
while a != 1128:
    calc_data_gamma = scp.norm.rvs(loc=1, scale=0.3)
    if 0 < calc_data_gamma < 1:
        gamma.append(calc_data_gamma)
        a += 1

logD_dataframe = pd.read_csv("logD.csv")
logD_series = logD_dataframe.squeeze()
logD = logD_series.values.tolist()

#malhotra mass
mu_tilde = []
for i in range(len(K)):
    mu_log = (3*(logD[i] - K_log[i])) + np.log10(3)
    mu_data = np.exp(mu_log)
    mu_tilde.append(mu_data)
mu_tilde_data = pd.Series(mu_tilde)
mu_tilde_log = np.log10(mu_tilde_data)
mu_tilde_log = mu_tilde_log.dropna()

planet_list = []
for i in range(len(gamma)):
    planet_min_mu = gamma[i] * (1+gamma[i])**(-1) * mu_tilde[i] * (const.M_earth.value/const.M_sun.value)
    planet_max_mu = (1+gamma[i])**(-1) * mu_tilde[i] * (const.M_earth.value/const.M_sun.value)
    planet_list.append(planet_min_mu)
    planet_list.append(planet_max_mu)
planet_list_log = np.log10(planet_list)

datafile = pd.read_csv("a_total.csv", skiprows=60)

#Otegi et al. mass
t_mu_data = []
otegi_tilde_list = []
for i in range(len(datafile)-1):
    if datafile["pl_dens"][i] > 3.3:
        mass = 0.9*((datafile["pl_rade"][i])**3.45)*const.M_earth.value
    else:
        mass = 1.74*((datafile["pl_rade"][i])**1.58)*const.M_earth.value
    mu = mass/(datafile["st_mass"][i]*const.M_sun.value)
    t_mu_data.append(mu)
    if datafile["hostname"][i] == datafile["hostname"][i + 1]:
        mu_tilde = t_mu_data[i] + t_mu_data[i - 1]
        otegi_tilde_list.append(mu_tilde)
t_mu_series = pd.Series(t_mu_data)
t_mu_log = np.log10(t_mu_series)
t_mu_log = t_mu_log.dropna()
otegi_tilde = pd.Series(otegi_tilde_list)
otegi_tilde_log = np.log10(otegi_tilde)
otegi_tilde_log = otegi_tilde_log.dropna()

#empirical mass data
e_mu_data = []
empirical_tilde_list = []
for i in range(len(datafile)-1):
    e_mu = (datafile["pl_bmasse"][i]*const.M_earth.value)/(datafile["st_mass"][i]*const.M_sun.value)
    e_mu_data.append(e_mu)
    if datafile["hostname"][i] == datafile["hostname"][i + 1]:
        mu_tilde = e_mu_data[i] + e_mu_data[i - 1]
        empirical_tilde_list.append(mu_tilde)
e_mu_series = pd.Series(e_mu_data)
e_mu_log = np.log10(e_mu_series)
e_mu_log = e_mu_log.dropna()
e_tilde = pd.Series(empirical_tilde_list)
e_tilde_log = np.log10(e_tilde)
e_tilde_log = e_tilde_log.dropna()

#malhotra old eq mass
past_eq_mass_list = []
malhotra_tilde_list = []
n_of_systems = 0
for i in range (len(datafile)-1):
    if datafile["pl_rade"][i]<1.5:
        m = (0.441 + 0.615*datafile["pl_rade"][i])*(datafile["pl_rade"][i])**3
    elif 1.5<datafile["pl_rade"][i]<4:
        m = 2.69*(datafile["pl_rade"][i])**(0.93)
    else:
        m = 3*(datafile["pl_rade"][i])
    past_eq_mass = (m*const.M_earth.value)/(datafile["st_mass"][i]*const.M_sun.value)
    past_eq_mass_list.append(past_eq_mass)
    if datafile["hostname"][i] == datafile["hostname"][i+1]:
        mu_tilde = past_eq_mass_list[i] + past_eq_mass_list[i-1]
        malhotra_tilde_list.append(mu_tilde)
    else:
        n_of_systems += 1
planet_masses = pd.Series(past_eq_mass_list)
planet_masses_log = np.log10(planet_masses)
planet_masses_log = planet_masses_log.dropna()
malhotrae12_tilde = pd.Series(malhotra_tilde_list)
malhotrae12_tilde_log = np.log10(malhotrae12_tilde)
malhotrae12_tilde_log = malhotrae12_tilde_log.dropna()

diff_data_t = []
diff_data_old = []
diff_data_malh = []
for i in range(len(e_mu_series)):
    diff_t = t_mu_series[i] - e_mu_series[i]
    diff_old = planet_masses[i] - e_mu_series[i]
    diff_malh = planet_list[i] - e_mu_series[i]
    diff_data_t.append(diff_t)
    diff_data_old.append(diff_old)
    diff_data_malh.append(diff_malh)
diff_series_t = pd.Series(diff_data_t)
diff_series_old = pd.Series(diff_data_old)
diff_series_malh = pd.Series(diff_data_malh)


[mean_fit_pla, std_fit_pla] = scp.norm.fit(planet_list_log)
[mean_fit_tmass, std_fit_tmass] = scp.norm.fit(t_mu_log)
[mean_fit_emass, std_fit_emass] = scp.norm.fit(e_mu_log)
[mean_fit_old, std_fit_old] = scp.norm.fit(planet_masses_log)

x = np.linspace(np.min(planet_list_log), np.max(planet_list_log), 2256)
tmass_lin = np.linspace(np.min(t_mu_log), np.max(t_mu_log), 2061)
emass_lin = np.linspace(np.min(e_mu_log), np.max(e_mu_log), 2061)
old_lin = np.linspace(np.min(planet_masses_log), np.max(planet_masses_log), 2189)

plt.figure(1, figsize=(7.2, 4.5))
#experimental mass plot
plt.plot(emass_lin, scp.norm.pdf(emass_lin, loc=mean_fit_emass, scale=std_fit_emass), color="0.3", label="NASA Exoplanet Archive")
#malhotra mass plot
plt.plot(x, scp.norm.pdf(x, loc=mean_fit_pla, scale=std_fit_pla), label="Malhotra", color="lime", linestyle="dashdot")
#new article mass plot
plt.plot(tmass_lin, scp.norm.pdf(tmass_lin, loc=mean_fit_tmass, scale=std_fit_tmass), color="blue", label="Otegi et al.", linestyle='dotted')
#old mass plot
plt.plot(old_lin, scp.norm.pdf(old_lin, loc=mean_fit_old, scale=std_fit_old), color="red", label="Weiss & Marcy, Wu & Lithwick", linestyle="dashed")
plt.legend()
plt.xlabel("log\u03BC$_{i}$")
plt.ylabel("PDF(log\u03BC$_{i}$)")
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=4)
plt.minorticks_on()

fig = plt.figure(2, figsize=(7.2, 4.5))

sub2 = plt.subplot(2, 1, 1)
sub2.scatter(e_mu_series, diff_series_t, marker="o", label="Otegi et al. ", color="blue", s=12)
sub2.scatter(e_mu_series, diff_series_old, marker="*", label="Weiss & Marcy, Wu & Lithwick", color="red", s=12)
sub2.scatter(e_mu_series, diff_series_malh, marker="^", label="Malhotra", color="lime", s=12)
sub2.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=4)
sub2.set_xlim(-0.0001, 0.002107)
sub2.set_ylim(-0.002, 0.00054)

sub3 = plt.subplot(2, 1, 2)
sub3.scatter(e_mu_series, diff_series_t, marker="o", label="y = Otegi et al. ", color="blue", s=12)
sub3.scatter(e_mu_series, diff_series_old, marker="*", label="y = Weiss & Marcy, Wu & Lithwick", color="red", s=12)
sub3.scatter(e_mu_series, diff_series_malh, marker="^", label="y = Malhotra", color="lime", s=12)
sub3.set_xlim(-0.00067,0.02091)
sub3.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=4)
sub3.minorticks_on()
sub3.fill_between((-0.000212, 0.002107), -0.005, 0.005, facecolor='orange', alpha=0.2)

con1 = ptc.ConnectionPatch(xyA=(-0.0001, -0.002), coordsA=sub2.transData, xyB=(-0.0001, -0.00073), coordsB=sub3.transData, color='orange')
fig.add_artist(con1)

con2 = ptc.ConnectionPatch(xyA=(0.002107, -0.002), coordsA=sub2.transData, xyB=(0.002107, -0.00073), coordsB=sub3.transData, color='orange')
fig.add_artist(con2)

fig.supylabel("Residual of x and y")
fig.supxlabel("x = NASA Exoplanet Archive \u03BC$_{i}$")
plt.legend()

fig2 = plt.figure(3, figsize=(7.2, 4.5))

sub21= plt.subplot(2, 2, 1)
sub21.hist(t_mu_log, density=True, bins=50, histtype="step", color="0.3")
sub21.set_title("Otegi et al.", fontsize=8)
sub22 = plt.subplot(2, 2, 2)
sub22.hist(e_mu_log, density=True, bins=50, histtype="step", color="0.3")
sub22.set_title("NASA Exoplanet Archive", fontsize=8)
sub23 = plt.subplot(2, 2, 3)
sub23.hist(planet_masses_log, density=True, bins=50, histtype="step", color="0.3")
sub23.set_xlabel("Weiss & Marcy, Wu & Lithwick")
sub23.set_xlim(left=-6)
sub24 = plt.subplot(2, 2, 4)
sub24.hist(planet_list_log, density=True, bins=50, histtype="step", color="0.3")
sub24.set_xlabel("Malhotra")
sub24.set_xlim(left= -7.5)

fig2.supylabel("Number of Planets")
fig2.supxlabel("log\u03BC$_{i}$")



fig4 = plt.figure(4, figsize=(7.2, 4.5))
sub41 = plt.subplot(2, 2, 1)
sub41.hist(otegi_tilde_log, density=True, bins=50, histtype="step", color="0.3")
sub41.set_title("Otegi et al. ", fontsize=8)
sub42 = plt.subplot(2, 2, 2)
sub42.hist(e_tilde_log, density=True, bins=50, histtype="step", color="0.3")
sub42.set_title("NASA Exoplanet Archive", fontsize=8)
sub43 = plt.subplot(2, 2, 3)
sub43.hist(malhotrae12_tilde_log, density=True, bins=75, histtype="step", color="0.3")
sub43.set_xlabel("Weiss & Marcy , Wu & Lithwick")
sub43.set_xlim(left= -5.5)
sub44 = plt.subplot(2, 2, 4)
sub44.hist(mu_tilde_log, density=True, bins=50, histtype="step", color="0.3")
sub44.set_xlabel("Malhotra")
sub44.set_xlim(left= -1.5)

fig4.supylabel("Number of Planets")
fig4.supxlabel("log$\~{\u03BC}_{i}$")
plt.show()
