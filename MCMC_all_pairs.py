import emcee
import matplotlib.pyplot as plt
import scipy.stats as scp
from scipy.special import expit as sigmoid, log_expit
import numpy as np
import corner
import pandas as pd
from pathlib import Path
from pairwise_model import log_K, log_prior, combine_log_posterior, orbital_log_likelihood
from tdpy.verbosity import print, retr_boolverb

catalog_path = Path(__file__).with_name("logD.csv")
print(f"Reading from {catalog_path}...")
logD = pd.read_csv(catalog_path)

def log_likelihood(theta,
                   # observed orbital separation for the system
                   osep_observed=logD):

    log_mutilde, gamma = theta

    loglikelihood = orbital_log_likelihood(log_mutilde, osep_observed)
    with np.errstate(over='ignore'):
        return float(np.sum(np.asarray(loglikelihood)))


def log_posterior(theta):
    return combine_log_posterior(theta, log_prior, log_likelihood)

ndim = 2
nwalkers = 1000

##### log_mutilde and gamma samples #####

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
state_1 = sampler.run_mcmc(pos, 50, progress=retr_boolverb())
sampler.reset()
sampler.run_mcmc(state_1, 250, progress=retr_boolverb())
flat_samples = sampler.get_chain(flat=True)

labels = ["log10_mutilde", "gamma"]
truths = [-4.31921754762478, 0.5]
fig = corner.corner(flat_samples, labels=labels, truths=truths)
plt.show()

flat_samples = flat_samples.T

mu_tilde = 10**(flat_samples[0])
mu_upper_mass = (1 + flat_samples[1])**(-1) * mu_tilde

plt.figure(2)
plt.scatter(mu_tilde, mu_upper_mass, color="Blue", s=7)
plt.xlabel("mu_tilde")
plt.ylabel("mu_upper_mass")
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=4)
plt.minorticks_on()

plt.figure(3)
plt.hist(mu_upper_mass,100, color="Blue")
plt.xlabel("mu_upper")
plt.ylabel("Number of Planets")
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=4)
plt.minorticks_on()
plt.show()
