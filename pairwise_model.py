import numpy as np
from scipy import stats
from scipy.special import log_expit


def adjacent_orbital_separations(catalog, system_position):
    """Return the existing adjacent-pair separations and system row advance."""
    hostnames = catalog['hostname']
    periods = catalog['pl_orbper']
    hostname = hostnames.iloc[system_position]
    if hostnames.isna().iloc[system_position] or (isinstance(hostname, str) and not hostname.strip()):
        raise ValueError('Selected host must have a nonmissing, nonblank identifier')
    positions = np.flatnonzero(hostnames.eq(hostname).fillna(False).to_numpy())
    if not len(positions) or positions[0] != system_position or not np.all(np.diff(positions) == 1):
        raise ValueError('Selected host rows must be contiguous and begin at system_position')
    if len(positions) > 1:
        selected_periods = periods.iloc[positions]
        if selected_periods.isna().any() or not np.isfinite(selected_periods).all() or (selected_periods <= 0).any():
            raise ValueError('Paired orbital periods must be finite and positive')
    separations = []
    for position in positions[:-1]:
        first_period = periods.iloc[position]
        second_period = periods.iloc[position + 1]
        if second_period < first_period:
            larger_period, smaller_period = first_period, second_period
        else:
            larger_period, smaller_period = second_period, first_period
        with np.errstate(over='ignore'):
            period_ratio = larger_period / smaller_period
        if np.isfinite(larger_period) and np.isfinite(smaller_period) and smaller_period > 0 and (not np.isfinite(period_ratio) or period_ratio - 1 < np.sqrt(np.finfo(float).eps)):
            if np.isfinite(period_ratio):
                log_ratio = np.log1p((larger_period - smaller_period) / smaller_period)
            else:
                log_ratio = np.log(larger_period) - np.log(smaller_period)
            separation = 2 * np.tanh(log_ratio / 3)
        else:
            separation = 2 * ((period_ratio ** (2 / 3) - 1) / (period_ratio ** (2 / 3) + 1))
        separations.append(separation)
    return np.array(separations), len(positions)


def log_K(log_mu_tilde, log_osep_observed):
    """Return the existing base-ten orbital-separation transform."""
    return log_osep_observed - (1 / 3) * log_mu_tilde + (1 / 3) * np.log10(3.0)


def reconstruct_neighbor_masses(first_mass, mass_ratios, directions):
    """Apply the existing neighboring-mass multiply/divide choices per sample."""
    if np.ma.is_masked(first_mass) or np.ma.is_masked(mass_ratios):
        raise ValueError('Mass inputs and ratios must not contain masked values')
    if np.ma.is_masked(directions):
        raise ValueError('Mass directions must not contain masked choices')
    first_mass = np.asarray(first_mass)
    mass_ratios = np.asarray(mass_ratios)
    directions = np.asarray(directions)
    if mass_ratios.shape != directions.shape or mass_ratios.ndim != first_mass.ndim + 1 or mass_ratios.shape[1:] != first_mass.shape:
        raise ValueError('Mass ratios and directions must align with each sample')
    if directions.dtype.kind not in 'biuf' or not np.isin(directions, [0, 1]).all():
        raise ValueError('Mass directions must contain only binary zero/one choices')
    for value, name in ((first_mass, 'Initial mass'), (mass_ratios, 'Mass ratios')):
        if value.dtype.kind not in 'iuf' or not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError(f'{name} must contain finite positive real numeric values')
    masses = [first_mass]
    for ratio, direction in zip(mass_ratios, directions):
        divide = direction == 0
        next_mass = np.empty(first_mass.shape, dtype=np.result_type(first_mass.dtype, mass_ratios.dtype, float))
        np.divide(masses[-1], ratio, out=next_mass, where=divide)
        np.multiply(masses[-1], ratio, out=next_mass, where=~divide)
        masses.append(next_mass)
    return np.stack(masses)


def orbital_separation(log_mu_tilde, log_osep_observed):
    """Evaluate the existing orbital separation, retaining floating-point limits."""
    with np.errstate(over='ignore', under='ignore'):
        return 10.0 ** log_K(log_mu_tilde, log_osep_observed)


def orbital_log_likelihood(log_mu_tilde, log_osep_observed):
    """Evaluate the existing orbital log-likelihood, retaining its extreme limits."""
    separation = orbital_separation(log_mu_tilde, log_osep_observed)
    return log_expit(np.sqrt(12.0) - separation)


def normal_log_density(value, mean, standard_deviation):
    """Evaluate a scalar Gaussian log density, retaining representable tails."""
    if not np.isfinite(mean) or not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError('Gaussian location must be finite and scale must be finite and positive')
    with np.errstate(over='ignore', under='ignore'):
        density = stats.norm.logpdf(value, mean, standard_deviation)
    if not np.isfinite(density):
        with np.errstate(over='ignore', under='ignore'):
            difference = value - mean
            standardized = difference / standard_deviation
            if not np.isfinite(difference) and np.isfinite(value):
                standardized = value / standard_deviation - mean / standard_deviation
            density = (-0.5 * standardized) * standardized - np.log(standard_deviation) - 0.5 * np.log(2.0 * np.pi)
    return density


def log_prior(theta):
    """Evaluate the existing pairwise prior, retaining its original density factors."""
    log_mutilde, gamma = theta
    if not np.isfinite(log_mutilde) or not np.isfinite(gamma) or not 0 < gamma < 1:
        return -np.inf
    mean_log_mutilde = -4.31921754762478
    std_log_mutilde = 0.28778378882844535
    mean_gamma = 0.5
    std_gamma = 0.3
    log_mass_gaussian = normal_log_density(log_mutilde, mean_log_mutilde, std_log_mutilde)
    if not np.isfinite(log_mass_gaussian):
        return -np.inf
    log_mutilde_dist = log_mass_gaussian - np.log(10.0) * log_mutilde
    log_gamma_dist = stats.norm.logpdf(np.log10(gamma), mean_gamma, std_gamma) - np.log(gamma)
    return log_mutilde_dist + log_gamma_dist


def combine_log_posterior(theta, prior, likelihood):
    """Combine caller-supplied terms without evaluating likelihood outside prior support."""
    logprior = prior(theta)
    if not np.isfinite(logprior):
        return -np.inf
    loglikelihood = likelihood(theta)
    with np.errstate(over='ignore'):
        return logprior + loglikelihood
