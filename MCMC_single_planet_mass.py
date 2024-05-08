import emcee
import matplotlib.pyplot as plt
import scipy.stats as scp
import numpy as np
import astropy.constants as ast
import corner
import pandas as pd

## st_mass = 0.5 solar mass

datafile = pd.read_csv("logD.csv")
logD_value = datafile.iloc[0][0]


def log_prior(theta):
    
    gamma, tilde = theta
    
    meanD = -0.30041
    stdD = 0.2357
    meanK = 1.2886
    stdK = 0.2388
    meanG = 0.5
    stdG = 0.3
    normD = 1/(10**(logD) * stdD * np.sqrt(2*np.pi))
    normG = 1/(gamma * stdG * np.sqrt(2*np.pi))
    normK = 1/(10**(logK) * stdK * np.sqrt(2*np.pi))
    a = normK * np.exp(-(logK-meanK)**2 / (2 * stdK**2))
    b = normG * np.exp(-(np.log10(gamma)-meanG)**2 / (2 * stdG**2))
    c = normD * np.exp(-(logD-meanD)**2 / (2 * stdD**2))
    if b <= 0 or b >= 1 or gamma != float:
        return -np.inf
    else:
        return a + b + c

def log_likelihood(theta, 
                   # observed orbital separation for the system
                   osep_observed=None, \
                   ):
    gamma, tilde = theta
    
    #mu_tilde = (10 ** (3 * (logD_value - logK) + np.log10(3)))
    #mu_upper = (1 + gamma) ** (-1) * mu_tilde

    #model = mu_upper * ast.M_sun.value / ast.M_earth.value
    logK = my_func(gamma, tilde, osep_observed)
    
    likelihood = 1. - sigmoid(K-K0)
    log_likelihood = np.log(likelihood)

    return log_likelihood


def log_posterior(theta):

    log_prior = log_prior(theta)
    
    if not np.isfinite(log_prior):
        return -np.inf
    else:
        log_likelihood = log_likelihood(theta, data=data)
        log_posterior = log_prior + log_likelihood
        return log_posterior

ndim = 3
nwalkers = 2000

##### logK, logD and gamma samples #####

logD = scp.norm.rvs(size=nwalkers, loc=-0.30041, scale=0.2357)
logD = np.transpose(np.array(logD))

logK = scp.norm.rvs(size=nwalkers, loc=1.32, scale=0.31)
logK = np.transpose(np.array(logK))

gamma = []
a = 0
while a != nwalkers:
    calc_data_gamma = scp.norm.rvs(1,0.3)
    if 0 < calc_data_gamma < 1:
        gamma.append(calc_data_gamma)
        a += 1
gamma = np.transpose(np.array(gamma))

##### end #####

pos = np.transpose(np.vstack((logK, gamma, logD)))

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
state_1 = sampler.run_mcmc(pos, 50, progress=True)
sampler.reset()
sampler.run_mcmc(state_1, 250, progress=True)
flat_samples = sampler.get_chain(flat=True)

labels = ["logK", "gamma", "logD"]
truths = [1.2886, 0.5, -0.30041]
fig = corner.corner(flat_samples, labels=labels, truths=truths)
plt.show()

flat_samples = flat_samples.T

masses = (1 + flat_samples[1]) ** (-1) * (10 ** (3 * (logD_value - flat_samples[0]) + np.log10(3))) * ast.M_sun.value / ast.M_earth.value

plt.figure(2)
plt.scatter(flat_samples[0], masses, color="Blue", s=7)
plt.xlabel("K")
plt.ylabel("$M/M_{\oplus}$, 500000 samples")
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=4)
plt.minorticks_on()
plt.show()



