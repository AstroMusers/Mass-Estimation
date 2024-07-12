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
import numpy as np
import astropy.constants as ast
import corner
import pandas as pd
from util import samp_powr

filename = "only_transit_data_28062023.csv"
datafile = pd.read_csv(filename, skiprows=202)

number_of_systems = 5
a = 0
row_counter = 0
passed_systems = 0
lower_limit = 1
upper_limit = 317.8
gamma_mean = 1
gamma_std = 0.3

st_mass = datafile["st_mass"][
              row_counter] * ast.M_sun.value / ast.M_earth.value  # solar mass turned to earth mass units

while a != number_of_systems:

    #### order decision ####

    order = 2

    #order = scp.powerlaw.rvs(a=0.4, loc=0, scale=1)
    #print(order)

    ########################

    d_list = []
    row_number = 0
    for k in range(len(datafile)):
        if datafile["hostname"][k] == datafile["hostname"][row_counter] and datafile["hostname"][k] == datafile["hostname"][k+1]:
            if datafile["pl_orbper"][k + 1] < datafile["pl_orbper"][k]:
                P_i = datafile["pl_orbper"][k] / datafile["pl_orbper"][k+1]
            else:
                P_i = datafile["pl_orbper"][k + 1] / datafile["pl_orbper"][k]
            D = 2 * (((P_i ** (2 / 3)) - 1) / ((P_i ** (2 / 3)) + 1))
            d_list.append(D) #list of orbital seperations
            row_number += 1

    row_number += 1 #to look at the next system at next loop
    logD_array = np.log10(np.array(d_list))

    if len(logD_array) == 0:
        passed_systems += 1
    else:

        mass_decide_list = []
        number_values_list = []

        def log_K(log_mu_tilde, log_osep_observed):
            logK = log_osep_observed - ((1/3) * log_mu_tilde) + ((1/3) * np.log10(3.))
            return logK

        def sigmoid(x):
            function = np.exp(x)/(np.exp(x) + 1)
            return function

        def log_prior(theta):
            total_log_prior = 0

            m0 = theta[0]
            m0_dist = np.log10(order-1) - np.log10(lower_limit) - order*np.log10(m0) + order*np.log10(lower_limit)
            #m0_dist = scp.powerlaw.logpdf(a=order, loc=lower_limit, scale=interval_width, x=m0)
            total_log_prior += m0_dist

            if m0 > 317.8 or m0 < 1:
                return -np.inf

            for i in range(1, len(theta)):
                gamma = theta[i]
                gamma_dist = scp.norm.logpdf(gamma, gamma_mean, gamma_std)
                total_log_prior += gamma_dist

                if gamma <= 0 or gamma >= 1:
                    return -np.inf

            return total_log_prior

        def log_likelihood(theta, osep_observed):
            m0 = theta[0]
            gamma = theta[1:]
            total_loglikelihood = 0

            masses = np.zeros(1)
            masses = np.vstack((masses, m0))

            mass_decide = random.randint(0, 1)
            if mass_decide % 2 == 0:
                m1 = m0 / gamma[0]
            else:
                m1 = m0 * gamma[0]
            masses = np.vstack((masses, m1))
            mass_decide_list.append(mass_decide)

            number_values = []
            for i in range(len(gamma) - 1):
                number = random.randint(0, 1)
                number_values.append(number)
                if number % 2 == 0:
                    m = masses[-1] / gamma[i + 1]
                else:
                    m = masses[-1] * gamma[i + 1]
                masses = np.vstack((masses, m))
            number_values_list.append(number_values)
            masses = masses[1:]

            mu_tilde_distribution = []
            for i in range(len(masses)):
                try:
                    mu_tilde = (masses[i] + masses[i + 1]) / st_mass
                    mu_tilde_distribution.append(mu_tilde)
                except IndexError:
                    pass
            mu_tilde_distribution = np.array(mu_tilde_distribution)
            log_mu_tilde = np.log10(mu_tilde_distribution)

            for i in range(len(logD_array)):
                logK = log_K(log_mu_tilde[i], logD_array[i])
                K = 10 ** logK
                if K > 40:
                    return -np.inf
                else:
                    K0 = np.sqrt(12)
                    likelihood = sigmoid(K - K0) #1 - sigmoid yapınca küçük değerler artmaya başlıyor, neden
                    loglikelihood = np.log10(likelihood)
                    total_loglikelihood += loglikelihood

            return total_loglikelihood

        def log_posterior(theta):
            logprior = log_prior(theta)
            if not np.isfinite(logprior):
                return -np.inf
            else:
                loglikelihood = log_likelihood(theta, logD_array)
                logposterior = logprior + loglikelihood
                return logposterior

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

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
        state_1 = sampler.run_mcmc(pos, burn_steps, progress=True)
        sampler.reset()
        sampler.run_mcmc(state_1, num_steps, progress=True)
        flat_samples = sampler.get_chain(flat=True)

        flat_samples = flat_samples.T

        m0 = flat_samples[0]
        gamma = flat_samples[1:]

        masses = np.zeros(num_steps*nwalkers)
        masses = np.vstack((masses,m0))
        m1_list = []

        if len(mass_decide_list) != num_steps*nwalkers:
            for i in range(num_steps*nwalkers - len(mass_decide_list)):
                mass_decide = random.randint(0, 1)
                mass_decide_list.append(mass_decide)
                number_values = []
                for j in range(len(gamma)-1):
                    number =  random.randint(0, 1)
                    number_values.append(number)
                number_values_list.append(number_values)

        for i in range(len(mass_decide_list)):
            if mass_decide_list[i] % 2 == 0:
                m1 = m0[i] / gamma[0][i]
            else:
                m1 = m0[i] * gamma[0][i]
            m1_list.append(m1)
        m1 = np.array(m1_list)
        masses = np.vstack((masses,m1))

        for i in range(len(gamma) - 1):
            m_list = []
            for j in range(len(number_values_list)):
                if number_values_list[j][i] % 2 == 0:
                    m = masses[-1][j] / gamma[i+1][j]
                else:
                    m = masses[-1][j] * gamma[i + 1][j]
                m_list.append(m)
            m = np.array(m_list)
            masses = np.vstack((masses, m))
        masses = masses[1:]

        flat_samples_corner = flat_samples
        calculated_masses = masses[1:]
        for i in range(len(calculated_masses)):
            labels.append(r"$\log M$" + fr"$_{i+1}$" + r" [$\log M_{\oplus}$]")
            truths.append(np.mean(np.log10(calculated_masses)[i]))
        flat_samples_corner = np.vstack((flat_samples_corner, np.log10(calculated_masses)))

        mu_tilde_distribution = []
        for i in range(len(masses)):
            try:
                mu_tilde = (masses[i] + masses[i+1])/st_mass
                log_mutilde = np.log10(mu_tilde)
                mu_tilde_distribution.append(mu_tilde)
                labels.append(r"$\log\tilde{\mu}$" + fr"$_{i}$")
                truths.append(np.mean(log_mutilde))
            except IndexError:
                pass
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
            truths.append(np.mean(K[i]))
            flat_samples_corner = np.vstack((flat_samples_corner, K))

        """order_dist = scp.powerlaw.rvs(a=0.5, loc=0, scale=1 , size=num_steps*nwalkers)
        labels.append("Order Distribution")
        truths.append(np.mean(order_dist))
        flat_samples_corner = np.vstack((flat_samples_corner,order_dist))"""

        flat_samples_corner[0] = np.log10(flat_samples_corner[0])
        flat_samples_corner = flat_samples_corner.T

        fig = corner.corner(flat_samples_corner, labels=labels, truths=truths)
        fig.text(0.5, 0.95, datafile["hostname"][row_counter] + f", Order: {-order}", ha='center', va='center', fontsize=12)
        plt.savefig(datafile["hostname"][row_counter] + "_corner_plot.pdf")
        plt.show()

    a += 1
    row_counter += row_number