import emcee
import matplotlib.pyplot as plt
import scipy.stats as scp
import numpy as np
import astropy.constants as ast
import corner
import seaborn as sns

## logD = -0.28246259424187536
## st_mass = 0.5 solar mass

def log_prior(theta):
    logK, gamma = theta
    meanK = 1.2886
    stdK = 0.2388
    meanG = 0.5
    stdG = 0.3
    normG = 1/(gamma * stdG * np.sqrt(2*np.pi))
    normK = 1/(10**(logK) * stdK * np.sqrt(2*np.pi))
    a = normK * np.exp(-(logK-meanK)**2 / (2 * stdK**2))
    b = normG * np.exp(-(np.log10(gamma)-meanG)**2 / (2 * stdG**2))
    if b <= 0 or b >= 1 or gamma != float:
        return -np.inf
    else:
        return a + b

def log_likelihood(theta):
    logK, gamma = theta
    mu_tilde = (10 ** (3 * (-0.28246259424187536 - logK) + np.log10(3)))
    model = (1 + gamma) ** (-1) * mu_tilde * ast.M_sun.value / ast.M_earth.value
    return model

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

ndim = 2
nwalkers = 1000

##### logK and gamma samples #####

logK = scp.norm.rvs(size=1000, loc=1.32, scale=0.31)
logK = np.transpose(np.array(logK))

plt.hist(logK)
plt.show()

gamma = []
a = 0
while a != 1000:
    calc_data_gamma = scp.norm.rvs(1,0.3)
    if 0 < calc_data_gamma < 1:
        gamma.append(calc_data_gamma)
        a += 1
gamma = np.transpose(np.array(gamma))

plt.hist(gamma)
plt.show()

##### end #####

pos = np.transpose(np.vstack((logK, gamma)))

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
state_1 = sampler.run_mcmc(pos, 100, progress=True)
sampler.reset()
sampler.run_mcmc(state_1, 500, progress=True)
flat_samples = sampler.get_chain(flat=True)

labels = ["logK", "gamma"]
truths = [1.2886, 0.5]
fig = corner.corner(flat_samples, labels=labels, truths=truths)
plt.show()

flat_samples = flat_samples.T

masses = (1 + flat_samples[1]) ** (-1) * (10 ** (3 * (-0.28246259424187536 - flat_samples[0]) + np.log10(3))) * ast.M_sun.value / ast.M_earth.value

plt.figure(2)
plt.scatter(flat_samples[0], masses, color="Blue", s=7)
plt.xlabel("K")
plt.ylabel("$M/M_{\oplus}$, 500000 samples")
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=4)
plt.minorticks_on()
plt.show()



