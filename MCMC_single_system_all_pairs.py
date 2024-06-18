import emcee
import matplotlib.pyplot as plt
import scipy.stats as scp
import numpy as np
import astropy.constants as ast
import corner
import pandas as pd

filename = "only_transit_data_28062023.csv"
datafile = pd.read_csv(filename, skiprows=202)

number_of_systems = 5
a = 0
i = 0
passed_systems = 0

ndim = 2
nwalkers = 100
burn_steps = 50
num_steps = 250

gamma_array = np.zeros((num_steps * nwalkers,))
log_mu_tilde_array = np.zeros((num_steps * nwalkers,))
mass_array = np.zeros((num_steps * nwalkers,))
K_array = np.zeros((num_steps * nwalkers,))
mean_K_list = []
mean_masses_list = []
which_systems = []
planet_number = []

while a != number_of_systems:

    d_list = []
    row_number = 0
    for k in range(len(datafile)):
        if datafile["hostname"][k] == datafile["hostname"][i] and datafile["hostname"][k] == datafile["hostname"][k+1]:
            if datafile["pl_orbper"][k + 1] < datafile["pl_orbper"][k]:
                P_i = datafile["pl_orbper"][k] / datafile["pl_orbper"][k+1]
            else:
                P_i = datafile["pl_orbper"][k + 1] / datafile["pl_orbper"][k]
            D = 2 * (((P_i ** (2 / 3)) - 1) / ((P_i ** (2 / 3)) + 1))
            d_list.append(D) #list of orbital seperations
            row_number += 1

    row_number += 1 #to look at the next system at next loop
    logD_array = np.log(np.array(d_list))

    if len(logD_array) == 0:
        passed_systems += 1

    for j in range(len(logD_array)):

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
            mean_mutilde = -4.3041661629482
            std_mutilde = 0.30148352295199915
            mean_gamma = 1
            std_gamma = 0.3

            log_mutilde_dist = scp.norm.logpdf(log_mutilde, mean_mutilde, std_mutilde)
            gamma_dist = scp.norm.logpdf(gamma, mean_gamma, std_gamma)

            total_log_prior = log_mutilde_dist + gamma_dist

            if gamma <= 0 or gamma >= 1:
                return -np.inf

            return total_log_prior

        def log_likelihood(theta,
                           # observed orbital separation for the system
                           osep_observed=logD_array[j]):

            gamma, log_mutilde = theta

            logK = log_K(log_mutilde, osep_observed)
            K = 10. ** (logK)
            K0 = np.sqrt(12)

            likelihood = 1. - sigmoid(K - K0)
            loglikelihood = np.log10(likelihood)
            return loglikelihood


        def log_posterior(theta):
            logprior = log_prior(theta)
            if not np.isfinite(logprior):
                return -np.inf
            else:
                loglikelihood = log_likelihood(theta)
                logposterior = logprior + loglikelihood
                return logposterior

        ##### mutilde and gamma samples #####

        log_mu_tilde = scp.norm.rvs(size=nwalkers, loc=-4.3041661629482, scale=0.30148352295199915)
        log_mu_tilde = np.transpose(np.array(log_mu_tilde))

        gamma = []
        b = 0
        while b != nwalkers:
            calc_data_gamma = scp.norm.rvs(1, 0.3)
            if 0 < calc_data_gamma < 1:
                gamma.append(calc_data_gamma)
                b += 1
        gamma = np.transpose(np.array(gamma))

        ##### end #####

        pos = np.transpose(np.vstack((gamma,log_mu_tilde)))

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
        state_1 = sampler.run_mcmc(pos, burn_steps, progress=True)
        sampler.reset()
        sampler.run_mcmc(state_1, num_steps, progress=True)
        flat_samples = sampler.get_chain(flat=True)

        flat_samples = flat_samples.T

        mu_tilde = 10**(flat_samples[1])
        st_mass = datafile["st_mass"][i] * ast.M_sun.value / ast.M_earth.value #solar mass turned to earth mass units
        masses = (1 + flat_samples[0])**(-1) * mu_tilde * st_mass #result is in earth mass units
        K = 10**log_K(flat_samples[1], logD_array[j])

        flat_samples = np.vstack((np.vstack((flat_samples,masses)),K))
        corner_samples = flat_samples.T

        mean_masses = np.nanmean(masses)
        mean_K = np.nanmean(K)

        gamma_array = np.vstack((gamma_array,flat_samples[0]))
        log_mu_tilde_array = np.vstack((log_mu_tilde_array, flat_samples[1]))
        mass_array = np.vstack((mass_array, masses))
        K_array = np.vstack((K_array, K))
        mean_K_list.append(mean_K)
        mean_masses_list.append(mean_masses)

        labels = ["gamma", "log_mutilde", "upper_mass", "dynamical_seperation (K)"]
        truths = [0.5, -4.3041661629482, np.nanmean(masses), np.nanmean(K)]
        fig = corner.corner(corner_samples, labels=labels, truths=truths)
        print(f"System Hostname = {datafile["hostname"][i]}")
        plt.show()
        which_systems.append(datafile["hostname"][i])
        planet_number.append(j + 1)  ##first planet, second planet, etc.

    a += 1
    i += row_number

##### #####

print(f"Passed System Count = {passed_systems}")
print(f"Systems With More Than One Planet = {which_systems}")
print(f"Planet Indexes = {planet_number}")

### total mass figure ###

mass_array_list = mass_array.tolist()
mass_array_merged = sum(mass_array_list, [])
mass_array_merged = mass_array_merged[250000:]
log_mass = np.log10(mass_array_merged)

plt.figure("Total Mass")
plt.hist(log_mass, bins=100)
plt.xlabel("Mass [Earth Mass]")
plt.ylabel("Occurance Number")

### violin plot ###

mass_array = mass_array[1:]
categories = np.array(which_systems)
subcategories = np.array(planet_number)
data = pd.DataFrame({'Category': categories, 'Value': mass_array_list[1:], 'Subcategory': subcategories})

unique_categories = data['Category'].unique()
unique_subcategories = data['Subcategory'].unique()

fig, ax = plt.subplots()
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_subcategories)))

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


for i, category in enumerate(unique_categories):
    for j, subcategory in enumerate(unique_subcategories):
        subset = data[(data['Category'] == category) & (data['Subcategory'] == subcategory)]
        subset_values = subset['Value'].values

        if len(subset_values) > 0:
            subset_values = np.concatenate(subset_values).astype(np.float64)

            try:
                parts = ax.violinplot(subset_values, positions=[i + j * 0.15 - 0.15], vert=False, showmeans=False,
                                      showmedians=False, showextrema=False)
            except ValueError:
                pass

            for pc in parts['bodies']:
                pc.set_facecolor(colors[j])
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)

            quartile1, medians, quartile3 = np.percentile(subset_values, [25, 50, 75])
            whiskers = adjacent_values(np.sort(subset_values), quartile1, quartile3)

            ax.scatter([medians], [i + j * 0.15 - 0.15], marker='o', color='white', s=25, zorder=3)
            ax.hlines(i + j * 0.15 - 0.15, quartile1, quartile3, color='k', linestyle='-', lw=5)
            ax.hlines(i + j * 0.15 - 0.15, whiskers[0], whiskers[1], color='k', linestyle='-', lw=1)

# Customize the plot
ax.set_yticks(np.arange(len(unique_categories)))
ax.set_yticklabels(unique_categories)
ax.set_xlabel('Maximum Mass [Earth Mass]')
ax.set_ylabel('The Star of The System')

# Add a legend
for j, subcategory in enumerate(unique_subcategories):
    ax.scatter([], [], color=colors[j], label=f'Interval of Planets {subcategory} - {subcategory+1}')
ax.legend()
plt.show()