import random
import astropy.constants as ast
import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner
import scipy.stats as scp
import pandas as pd


def log_prior(theta):
    gamma, logK, logD = theta
    sigmagamma = 0.3
    sigmalogK = 0.31
    sigmalogD = 0.2357
    mugamma = 0.5
    mulogK = 1.32
    mulogD = -0.3004
    a = -0.5 * ((gamma - mugamma) / sigmagamma) ** 2
    b = -0.5 * ((logK - mulogK) / sigmalogK) ** 2
    c = -0.5 * ((logD - mulogD) / sigmalogD) ** 2
    if gamma <= 0 or gamma >= 1 or logD >= np.log(2):
        return -np.inf
    return a + b + c

def log_likelihood(theta):
    gamma, logK, logD = theta
    mu_tilde = (10 ** (3 * (-0.28246259424187536 - logK) + np.log10(3)))
    model = (1 + gamma) ** (-1) * mu_tilde * ast.M_sun.value / ast.M_earth.value
    return model

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + np.sum(np.log(log_likelihood(theta)))

ndim = 3
nwalkers = 50

#K-calculation
logK = scp.norm.rvs(size=50, loc=1.32, scale=0.31)
logK = np.transpose(np.array(logK))

#gamma-calculation
gamma = []
a = 0
while a != 50:
    calc_data_gamma = scp.norm.rvs(1,0.3)
    if 0 < calc_data_gamma < 1:
        gamma.append(calc_data_gamma)
        a += 1

gamma = np.transpose(np.array(gamma))

#logD data
logD_dataframe = pd.read_csv("logD.csv")
logD_series = logD_dataframe.squeeze()
logD = logD_series.values.tolist()

logD_choice = random.choices(logD,k=50)
logD_choice = np.transpose(np.array(logD_choice))

pos = np.transpose(np.vstack((gamma, logK, logD_choice)))

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
state_1 = sampler.run_mcmc(pos, 1000, progress=True)
sampler.reset()
sampler.run_mcmc(state_1, 5000, progress=True)
"""
sampler2 = emcee.EnsembleSampler(nwalkers, ndim, log_prior)
state_2 = sampler2.run_mcmc(pos, 1000, progress=True)
sampler.reset()
sampler.run_mcmc(state_2, 5000, progress=True)
"""
flat_samples = sampler.get_chain(flat=True)
#flat_samples2 = sampler2.get_chain(flat=True)

mu_tilde = 10**(3*(flat_samples[:,2] - flat_samples[:,1]) + np.log10(3))
max_mu = (1-flat_samples[:,0])**(-1) * mu_tilde

plt.hist(flat_samples[:, 0], bins=50, density=True, alpha=0.5, color="blue", label="gamma")
plt.hist(flat_samples[:, 1], bins=50, density=True, alpha=0.5, color="green", label="logK")
plt.hist(flat_samples[:, 2], bins=50, density=True, alpha=0.5, color="red", label="logD")
plt.xlabel("Parameter value")
plt.ylabel("Probability Density")
plt.legend()
plt.show()

plt.hist(np.log(max_mu), bins=50, density=True, alpha=0.5, color="orange", label="max_mu")
plt.show()

labels = ["gamma", "logK", "logD"]
truths = [1, 1.32, -0.3004]
fig = corner.corner(flat_samples, labels=labels, truths=truths)
#fig2 = corner.corner(flat_samples2, labels=labels, truths=truths)
plt.show()

