import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats as scp
import seaborn as sns
import astropy.constants as const

plt.rcParams.update({'font.size': 9})

#K-calculation

K = scp.norm.rvs(size=1129, loc=1.32, scale=0.31)
print(len(K))

#gamma-calculation

gamma = []
a = 0
while a != 1129:
    calc_data_gamma = scp.norm.rvs(loc=1, scale=0.3)
    if 0 < calc_data_gamma < 1:
        gamma.append(calc_data_gamma)
        a += 1

#Gamma and K plot lines

# plt.figure(1)
# plt.subplot(1, 2, 1)
# plt.hist(K, bins=30, density=True)
# plt.xlabel("K")
# plt.subplot(1, 2, 2)
# plt.hist(gamma, bins=30, density=True)
# plt.xlabel("Gamma")
# plt.show()

logD_dataframe = pd.read_csv("logD.csv")
logD_series = logD_dataframe.squeeze()
logD = logD_series.values.tolist()

K_log = np.log(K)

mu_tilde = []
for i in range(len(K)):
    mu_log = (3*(logD[i] - K_log[i])) + np.log(3)
    mu_data = np.exp(mu_log)
    mu_tilde.append(mu_data)
mu_tilde_data = pd.Series(mu_tilde)
mu_tilde_data.to_csv("mu_tilde.csv")

planet_list = []
for i in range(len(gamma)):
    planet_min_mu = gamma[i] * (1+gamma[i])**(-1) * mu_tilde[i]
    planet_max_mu = (1+gamma[i])**(-1) * mu_tilde[i]
    planet_list.append(planet_min_mu)
    planet_list.append(planet_max_mu)

np.savetxt("planet_mu.csv", planet_list, delimiter=",", fmt="%s")

planet_list_log = np.log(planet_list)

datafile = pd.read_csv("a_total.csv", skiprows=60)

t_mu_data = []
for i in range (len(datafile)):
    if datafile["pl_dens"][i] > 3.3:
        mass = 0.9*((datafile["pl_rade"][i])**3.45)*const.M_earth.value
    else:
        mass = 1.74*((datafile["pl_rade"][i])**1.58)*const.M_earth.value
    mu = mass/(datafile["st_mass"][i]*const.M_sun.value)
    t_mu_data.append(mu)
t_mu_series = pd.Series(t_mu_data)
t_mu_log = np.log(t_mu_series)
t_mu_log = t_mu_log.dropna()

e_mu_data = []
for i in range(len(datafile)):
    e_mu = (datafile["pl_bmasse"][i]*const.M_earth.value)/(datafile["st_mass"][i]*const.M_sun.value)
    e_mu_data.append(e_mu)
e_mu_series = pd.Series(e_mu_data)
e_mu_log = np.log(e_mu_series)
e_mu_log = e_mu_log.dropna()

diff_data = []
for i in range(len(e_mu_series)):
    diff = t_mu_series[i] - e_mu_series[i]
    diff_data.append(diff)
diff_series = pd.Series(diff_data)
diff_abs = np.abs(diff_series)
diff_log = np.log(diff_abs)
diff_log = diff_log.dropna()

[mean_fit_pla, std_fit_pla] = scp.norm.fit(planet_list_log)
[mean_fit, std_fit] = scp.norm.fit(K_log)
[mean_fit_tmass, std_fit_tmass] = scp.norm.fit(t_mu_log)
[mean_fit_emass, std_fit_emass] = scp.norm.fit(e_mu_log)

x = np.linspace(np.min(planet_list_log), np.max(planet_list_log), 2256)
z = np.linspace(np.min(K_log), np.max(K_log), 1129)
tmass_lin = np.linspace(np.min(t_mu_log), np.max(t_mu_log), 2061)
emass_lin = np.linspace(np.min(e_mu_log), np.max(e_mu_log), 2061)

plt.figure(1)
plt.subplot(2,1,1)
#experimental mass plot
plt.plot(emass_lin, scp.norm.pdf(emass_lin, loc=mean_fit_emass, scale=std_fit_emass), color=sns.color_palette('Set2')[0], label="Exp Mu")
#malhotra mass plot
plt.plot(x, scp.norm.pdf(x, loc=mean_fit_pla, scale=std_fit_pla), color=sns.color_palette('Set2')[2], label="Malhotra Mu")
#new article mass plot
plt.plot(tmass_lin, scp.norm.pdf(tmass_lin, loc=mean_fit_tmass, scale=std_fit_tmass), color=sns.color_palette('Set2')[6], label="New Article Mu")
plt.title("Prob Density of Mu")
plt.legend()
plt.xlabel("log_mu")
plt.ylabel("PDF(log_mu)")
plt.subplot(2,1,2)
plt.scatter(e_mu_series,diff_series,color=sns.color_palette('Set2')[0],s=7, label="Theoretical - Experimental")
plt.xlabel("Experimental Mu")
plt.ylabel("Difference of Data")
plt.legend()
plt.show()
