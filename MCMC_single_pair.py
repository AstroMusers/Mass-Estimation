import emcee
import matplotlib.pyplot as plt
import scipy.stats as scp
from scipy.special import expit as sigmoid, log_expit
import numpy as np
import astropy.constants as ast
import corner
import pandas as pd
from pathlib import Path
from pairwise_model import log_K, log_prior, combine_log_posterior, orbital_log_likelihood

catalog_path = Path(__file__).with_name("logD.csv")
print(f"Reading from {catalog_path}...")
datafile = pd.read_csv(catalog_path)
logD_value = datafile.iloc[0, 0]

def log_likelihood(theta,
                   # observed orbital separation for the system
                   osep_observed=logD_value):

    tilde, gamma = theta

    loglikelihood = orbital_log_likelihood(tilde, osep_observed)
    return loglikelihood


def log_posterior(theta):
    return combine_log_posterior(theta, log_prior, log_likelihood)

ndim = 2
nwalkers = 1000

##### mutilde and gamma samples #####

log_mu_tilde = scp.norm.rvs(size=nwalkers, loc=-4.31921754762478, scale=0.28778378882844535)
log_mu_tilde = np.transpose(np.array(log_mu_tilde))

gamma = []
a = 0
while a != nwalkers:
    calc_data_gamma = scp.norm.rvs(1, 0.3)
    if 0 < calc_data_gamma < 1:
        gamma.append(calc_data_gamma)
        a += 1
gamma = np.transpose(np.array(gamma))

##### end #####

pos = np.transpose(np.vstack((log_mu_tilde, gamma)))

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
state_1 = sampler.run_mcmc(pos, 50, progress=True)
sampler.reset()
sampler.run_mcmc(state_1, 250, progress=True)
flat_samples = sampler.get_chain(flat=True)

flat_samples = flat_samples.T

mu_tilde = 10**(flat_samples[0])
st_mass = 0.5 * ast.M_sun.value / ast.M_earth.value #solar mass turned to earth mass units
masses = (1 + flat_samples[1])**(-1) * mu_tilde * st_mass
K = 10**log_K(flat_samples[0], logD_value)

flat_samples = np.vstack((np.vstack((flat_samples,masses)),K))
corner_samples = flat_samples.T

labels = ["log_mutilde", "gamma", "upper_mass", "orbital_seperation"]
truths = [-4.31921754762478, 0.5, np.nanmean(masses), np.nanmean(K)]
fig = corner.corner(corner_samples, labels=labels, truths=truths)
plt.show()

plt.figure(2)
plt.scatter(K, masses, color="Blue", s=7)
plt.xlabel("K")
plt.ylabel(r"Upper Masses (M$_\oplus$)")
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=4)
plt.minorticks_on()
plt.show()
