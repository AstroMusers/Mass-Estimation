import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats as scp
import astropy.constants as const
import matplotlib.patches as ptc

plt.rcParams.update({'font.size': 8})

filename = "only_transit_data_28062023.csv"
datafile = pd.read_csv(filename, skiprows=177)

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

#weiss_wu eq mass
past_eq_mass_list = []
weiss_wu_mass = []
for i in range (len(datafile)-1):
    if datafile["pl_rade"][i]<1.5:
        m = (0.441 + 0.615*datafile["pl_rade"][i])*(datafile["pl_rade"][i])**3
    elif 1.5<datafile["pl_rade"][i]<4:
        m = 2.69*(datafile["pl_rade"][i])**(0.93)
    else:
        m = 3*(datafile["pl_rade"][i])
    past_eq_mass = (m)/(datafile["st_mass"][i])
    past_eq_mass_list.append(past_eq_mass)
    weiss_wu_mass.append(m)
planet_masses = pd.Series(past_eq_mass_list)
planet_masses_log = np.log10(planet_masses)
planet_masses_log = planet_masses_log.dropna()
weiss_wu_mass_log = np.log10(weiss_wu_mass)
weiss_wu_mass_log = pd.Series(weiss_wu_mass_log)
weiss_wu_mass_log = weiss_wu_mass_log.dropna()


diff_data_otegi = []
diff_data_weisswu = []
diff_data_our = []
for i in range(len(e_mu_series)):
    diff_otegi = t_mass_data[i] - e_mass_data[i]
    diff_weisswu = weiss_wu_mass[i] - e_mass_data[i]
    diff_our = planet_mass[i] - e_mass_data[i]
    diff_data_otegi.append(diff_otegi)
    diff_data_weisswu.append(diff_weisswu)
    diff_data_our.append(diff_our)
diff_series_otegi = pd.Series(diff_data_otegi)
diff_series_weisswu = pd.Series(diff_data_weisswu)
diff_series_our = pd.Series(diff_data_our)


[mean_fit_pla, std_fit_pla] = scp.norm.fit(planet_mass_log)
[mean_fit_tmass, std_fit_tmass] = scp.norm.fit(t_mass_log)
[mean_fit_emass, std_fit_emass] = scp.norm.fit(e_mass_log)
[mean_fit_old, std_fit_old] = scp.norm.fit(weiss_wu_mass_log)

x = np.linspace(np.min(planet_mass_log), np.max(planet_mass_log), 1745)
tmass_lin = np.linspace(np.min(t_mass_log), np.max(t_mass_log), 1744)
emass_lin = np.linspace(np.min(e_mass_log), np.max(e_mass_log), 1741)
old_lin = np.linspace(np.min(weiss_wu_mass_log), np.max(weiss_wu_mass_log), 1744)

plt.figure(1, figsize=(7.2, 4.5))
#experimental mass plot
plt.plot(emass_lin, scp.norm.pdf(emass_lin, loc=mean_fit_emass, scale=std_fit_emass), color="0.3", label="NASA Exoplanet Archive, 2023")
#our mass plot
plt.plot(x, scp.norm.pdf(x, loc=mean_fit_pla, scale=std_fit_pla), label="This Work", color="lime", linestyle="dashdot")
#otegi mass plot
plt.plot(tmass_lin, scp.norm.pdf(tmass_lin, loc=mean_fit_tmass, scale=std_fit_tmass), color="blue", label="Otegi et al. 2020", linestyle='dotted')
#weiss-wu mass plot
plt.plot(old_lin, scp.norm.pdf(old_lin, loc=mean_fit_old, scale=std_fit_old), color="red", label="will change (weiss-wu)", linestyle="dashed")
plt.legend()
plt.xlabel("log$_{10}$ $M/M_{\oplus}$")
plt.ylabel("PDF(log$_{10}$ $M/M_{\oplus}$)")
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=4)
plt.minorticks_on()

fig = plt.figure(2, figsize=(7.2, 4.5))
sub2 = plt.subplot(2, 1, 1)
sub2.scatter(e_mu_series, diff_series_otegi, marker="o", label="Otegi et al., 2020", color="blue", s=12)
sub2.scatter(e_mu_series, diff_series_weisswu, marker="*", label="will change (weiss-wu)", color="red", s=12)
sub2.scatter(e_mu_series, diff_series_our, marker="^", label="This Work", color="lime", s=12)
sub2.set_xlim(-1, 22)
sub2.set_ylim(-25, 25)

sub3 = plt.subplot(2, 1, 2)
sub3.scatter(e_mu_series, diff_series_otegi, marker="o", label="y = Otegi et al. 2020 ", color="blue", s=12)
sub3.scatter(e_mu_series, diff_series_weisswu, marker="*", label="y = will change (weiss-wu)", color="red", s=12)
sub3.scatter(e_mu_series, diff_series_our, marker="^", label="y = This Work", color="lime", s=12)
sub3.set_xlim(-3,100)
sub3.set_ylim(-100,75)
sub3.fill_between((-1, 22), -25, 25, facecolor='orange', alpha=0.2)

con1 = ptc.ConnectionPatch(xyA=(-1, -25), coordsA=sub2.transData, xyB=(-1, 0), coordsB=sub3.transData, color='orange')
fig.add_artist(con1)

con2 = ptc.ConnectionPatch(xyA=(22, -25), coordsA=sub2.transData, xyB=(22, 0), coordsB=sub3.transData, color='orange')
fig.add_artist(con2)

fig.supylabel("Residual of x and y")
fig.supxlabel("x = NASA Exoplanet Archive, 2023 \u03BC$_{i}$")
plt.legend()

fig2 = plt.figure(3, figsize=(7.2, 4.5))

sub21= plt.subplot(2, 2, 1)
sub21.hist(t_mu_log, density=True, bins=50, histtype="step", color="0.3")
sub21.set_title("Otegi et al. 2020", fontsize=8)
sub21.set_xlim(-1,3)
sub21.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=4)
sub21.minorticks_on()
sub22 = plt.subplot(2, 2, 2)
sub22.hist(e_mu_log, density=True, bins=50, histtype="step", color="0.3")
sub22.set_title("NASA Exoplanet Archive, 2023", fontsize=8)
sub22.set_xlim(-1,3)
sub22.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=4)
sub22.minorticks_on()
sub23 = plt.subplot(2, 2, 3)
sub23.hist(planet_masses_log, density=True, bins=50, histtype="step", color="0.3")
sub23.set_xlabel("will change (weiss-wu)")
sub23.set_xlim(left=-.5)
sub23.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=4)
sub23.minorticks_on()
sub24 = plt.subplot(2, 2, 4)
sub24.hist(planet_list_log, density=True, bins=50, histtype="step", color="0.3")
sub24.set_xlabel("This Work")
sub24.set_xlim(left= -9)
sub24.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=4)
sub24.minorticks_on()
fig2.supylabel("Number of Planets")
fig2.supxlabel("log$_{10}$\u03BC$_{i}$")

plt.show()
