import emcee
import matplotlib.pyplot as plt
import scipy.stats as scp
import numpy as np
import astropy.constants as ast
import corner
import pandas as pd

datafile = pd.read_csv("logD.csv")
logD_value = datafile.iloc[0][0]

def log_K(log_mu_tilde,
          # observed orbital separation for the system
          log_osep_observed):
    logK = log_osep_observed - (1/3) * log_mu_tilde + (1/3) * np.log10(3.)
    return logK

def sigmoid(x):
    function = np.exp(x)/(np.exp(x) + 1)
    return function

def log_prior(theta):
    gamma, log_mutilde = theta
    mean_mutilde = -4.31921754762478
    std_mutilde = 0.28778378882844535
    mean_gamma = 0.5
    std_gamma = 0.3
    norm_gamma = 1. / (gamma * std_gamma * np.sqrt(2. * np.pi))
    norm_mutilde = 1. / (10. ** (log_mutilde) * std_mutilde * np.sqrt(2. * np.pi))

    log_mutilde_dist = norm_mutilde * np.exp(-(log_mutilde - mean_mutilde) ** 2. / (2. * std_mutilde ** 2.))
    gamma_dist = norm_gamma * np.exp(-(np.log10(gamma) - mean_gamma) ** 2. / (2. * std_gamma ** 2.))

    if gamma_dist <= 0 or gamma_dist >= 1 or gamma != float:
        return -np.inf
    else:
        return log_mutilde_dist + gamma_dist

def log_likelihood(theta,
                   # observed orbital separation for the system
                   osep_observed=logD_value):

    gamma, tilde = theta

    logK = log_K(tilde, osep_observed)
    K = 10. ** (logK)
    K0 = np.sqrt(12)

    likelihood = 1. - sigmoid(K - K0)
    loglikelihood = np.log(likelihood)
    return loglikelihood


def log_posterior(theta):
    logprior = log_prior(theta)
    if not np.isfinite(logprior):
        return -np.inf
    else:
        loglikelihood = log_likelihood(theta)
        logposterior = logprior + loglikelihood
        return logposterior

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
plt.ylabel("Upper Masses (M$_\oplus$)")
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=4)
plt.minorticks_on()
plt.show()
