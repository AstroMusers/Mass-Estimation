#m0 ilk gezegenin kütle dağılımı olsun (tick)
#bütün indisleri 0 1 2 diye değiştir (tick)
#maksimum gezegen kütlesine bak ve hangi sistemde olduğunu kaydet (tick)
#https://github.com/tdaylan/tdpy/blob/6c9337f828204e36265a968f3e454278f4f109d2/tdpy/util.py#L354 (tick)
#turn to the sigmoid function and make kmax = 40 (tick)
#1 dünyadan 10 dünyaya kadar dene (tick)
#sigmoidin merkezi minimum, kmin (tick)
#ötegezegen dosyasındaki gezegenlerin gamma plot yap (tick)
import random
import emcee
import matplotlib.pyplot as plt
import scipy.stats as scp
from scipy.special import expit as sigmoid, log_expit
import numpy as np
import astropy.constants as ast
import corner
import pandas as pd
from pathlib import Path
from tdpy import samp_powr
from pairwise_model import adjacent_orbital_separations, reconstruct_neighbor_masses, orbital_separation, log_K

filename = Path(__file__).with_name("only_transit_data_28062023.csv")
print(f"Reading from {filename}...")
datafile = pd.read_csv(filename, skiprows=202)

number_of_systems = 5
a = 0
row_counter = 0
passed_systems = 0
lower_limit = 1
upper_limit = 317.8
gamma_mean = 1
gamma_std = 0.3

while a != number_of_systems and row_counter < len(datafile):
    st_mass = datafile["st_mass"].iloc[row_counter] * ast.M_sun.value / ast.M_earth.value  # [M_earth]

    #### order decision ####

    order = 2

    #order = scp.powerlaw.rvs(a=0.4, loc=0, scale=1)
    #print(order)

    ########################

    separations, row_number = adjacent_orbital_separations(datafile, row_counter)
    logD_array = np.log10(separations)

    if len(logD_array) == 0:
        passed_systems += 1
    else:

        def log_prior(theta):
            total_log_prior = 0

            m0 = theta[0]
            if not np.isfinite(m0) or m0 > 317.8 or m0 < 1:
                return -np.inf
            m0_dist = np.log10(order-1) - np.log10(lower_limit) - order*np.log10(m0) + order*np.log10(lower_limit)
            #m0_dist = scp.powerlaw.logpdf(a=order, loc=lower_limit, scale=interval_width, x=m0)
            total_log_prior += m0_dist

            for i in range(1, len(theta)):
                gamma = theta[i]
                if not np.isfinite(gamma) or gamma <= 0 or gamma >= 1:
                    return -np.inf
                gamma_dist = scp.norm.logpdf(gamma, gamma_mean, gamma_std)
                total_log_prior += gamma_dist

            return total_log_prior

        def log_likelihood(theta, osep_observed, return_directions=False):
            m0 = theta[0]
            gamma = theta[1:]
            if np.ma.is_masked(osep_observed):
                raise ValueError('Observed separations must not contain masked values')
            observations = np.asarray(osep_observed)
            if not len(gamma) or observations.shape != (len(gamma),):
                raise ValueError('Observed separations must be one-dimensional with one value per neighboring mass ratio')
            if np.isnan(observations).any():
                raise ValueError('Observed separations must not contain NaN')
            total_loglikelihood = 0

            mass_decide = random.randint(0, 1)
            number_values = []
            for i in range(len(gamma) - 1):
                number = random.randint(0, 1)
                number_values.append(number)
            directions = (mass_decide, *number_values)
            masses = reconstruct_neighbor_masses(m0, gamma, directions)[:, None]
            mu_tilde_distribution = (masses[:-1] + masses[1:]) / st_mass
            log_mu_tilde = np.log10(mu_tilde_distribution)

            for i, observed_separation in enumerate(observations):
                K = orbital_separation(log_mu_tilde[i], observed_separation)
                if K > 40:
                    return (-np.inf, directions) if return_directions else -np.inf
                else:
                    K0 = np.sqrt(12)
                    loglikelihood = log_expit(K - K0) / np.log(10.0)
                    total_loglikelihood += loglikelihood

            return (total_loglikelihood, directions) if return_directions else total_loglikelihood

        def log_posterior(theta, return_directions=False):
            logprior = log_prior(theta)
            if not np.isfinite(logprior):
                return (-np.inf, (0,) * len(logD_array)) if return_directions else -np.inf
            loglikelihood, directions = log_likelihood(theta, logD_array, return_directions=True)
            logposterior = logprior + loglikelihood
            return (logposterior, directions) if return_directions else logposterior

        ##### mutilde and gamma samples #####

        ndim = len(logD_array) + 1
        nwalkers = 5000
        burn_steps = 5
        num_steps = 25

        labels = []
        truths = []

        pos_array = np.zeros(nwalkers)

        m0 = samp_powr(nwalkers, lower_limit, upper_limit, order)
        pos_array = np.vstack((pos_array,m0))
        labels.append(fr"$\log M_{0}$")
        truths.append(np.mean(np.log10(m0)))

        for i in range(len(logD_array)):
            gamma = []
            b = 0
            while b != nwalkers:
                calc_data_gamma = scp.norm.rvs(gamma_mean, gamma_std)
                if 0 < calc_data_gamma < 1:
                    gamma.append(calc_data_gamma)
                    b += 1
            gamma = np.array(gamma)
            pos_array = np.vstack((pos_array,gamma))
            labels.append(fr"$\gamma_{i}$")
            truths.append(np.mean(gamma))

        pos = np.transpose(pos_array[1:])

        ##### end #####

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(True,),
                        blobs_dtype=[('mass_directions', np.int8, (len(logD_array),))])
        state_1 = sampler.run_mcmc(pos, burn_steps, progress=True)
        sampler.reset()
        sampler.run_mcmc(state_1, num_steps, progress=True)
        flat_samples = sampler.get_chain(flat=True)

        flat_samples = flat_samples.T

        m0 = flat_samples[0]
        gamma = flat_samples[1:]
        directions = sampler.get_blobs(flat=True)['mass_directions'].T
        masses = reconstruct_neighbor_masses(m0, gamma, directions)

        flat_samples_corner = flat_samples
        calculated_masses = masses[1:]
        for i in range(len(calculated_masses)):
            labels.append(r"$\log M$" + fr"$_{i+1}$" + r" [$\log M_{\oplus}$]")
            truths.append(np.mean(np.log10(calculated_masses)[i]))
        flat_samples_corner = np.vstack((flat_samples_corner, np.log10(calculated_masses)))

        mu_tilde_distribution = []
        for i in range(len(masses) - 1):
            mu_tilde = (masses[i] + masses[i+1])/st_mass
            log_mutilde = np.log10(mu_tilde)
            mu_tilde_distribution.append(mu_tilde)
            labels.append(r"$\log\tilde{\mu}$" + fr"$_{i}$")
            truths.append(np.mean(log_mutilde))
        mu_tilde_distribution = np.array(mu_tilde_distribution)
        log_mu_tilde = np.log10(mu_tilde_distribution)
        flat_samples_corner = np.vstack((flat_samples_corner, log_mu_tilde))

        """number_dist = []
        for i in range(num_steps*nwalkers):
            number = random.randint(0,1)
            number_dist.append(number)
        number_dist = np.array(number_dist)
        labels.append("Random Number Distribution")
        truths.append(np.mean(number_dist))
        flat_samples_corner = np.vstack((flat_samples_corner, number_dist))"""

        for i in range(len(logD_array)):
            logK = log_K(log_mu_tilde[i], logD_array[i])
            K = 10**logK
            labels.append(fr"$\kappa_{i}$")
            truths.append(np.mean(K))
            flat_samples_corner = np.vstack((flat_samples_corner, K))

        """order_dist = scp.powerlaw.rvs(a=0.5, loc=0, scale=1 , size=num_steps*nwalkers)
        labels.append("Order Distribution")
        truths.append(np.mean(order_dist))
        flat_samples_corner = np.vstack((flat_samples_corner,order_dist))"""

        flat_samples_corner[0] = np.log10(flat_samples_corner[0])
        flat_samples_corner = flat_samples_corner.T

        fig = corner.corner(flat_samples_corner, labels=labels, truths=truths)
        fig.text(0.5, 0.95, datafile["hostname"].iloc[row_counter] + f", Order: {-order}", ha='center', va='center', fontsize=12)
        plt.savefig(datafile["hostname"].iloc[row_counter] + "_corner_plot.pdf")
        plt.show()

    a += 1
    row_counter += row_number