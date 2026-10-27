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
        separations.append(
            orbital_separation_from_periods(smaller_period, larger_period)
        )
    return np.array(separations), len(positions)


def orbital_separation_from_periods(first_period, second_period):
    """Return signed orbital spacing from ordered periods in the same time unit."""
    first_period = np.float64(first_period)
    second_period = np.float64(second_period)
    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        period_ratio = second_period / first_period
    if first_period > 0 and second_period > 0 and (
        not np.isfinite(period_ratio)
        or abs(period_ratio - 1) < np.sqrt(np.finfo(float).eps)
    ):
        if np.isfinite(period_ratio):
            log_ratio = np.log1p((second_period - first_period) / first_period)
        else:
            log_ratio = np.log(second_period) - np.log(first_period)
        return 2 * np.tanh(log_ratio / 3)
    with np.errstate(over='ignore', invalid='ignore'):
        ratio_power = period_ratio ** (2 / 3)
        return 2 * ((ratio_power - 1) / (ratio_power + 1))


def _adjacent_pair_positions(catalog, instrument=None):
    hostnames = catalog["hostname"]
    instruments = catalog["disc_instrument"] if instrument is not None else None
    pair_positions = []
    for position in range(len(catalog) - 1):
        if instrument is not None and instruments.iloc[position] != instrument:
            continue
        if hostnames.iloc[position] != hostnames.iloc[position + 1]:
            continue
        pair_positions.append(position)
    return np.asarray(pair_positions, dtype=int)


def adjacent_period_spacings(catalog, instrument=None):
    """Return adjacent same-host period spacings and their row positions."""
    periods = catalog["pl_orbper"]
    pair_positions = _adjacent_pair_positions(catalog, instrument)
    spacings = [
        orbital_separation_from_periods(periods.iloc[position], periods.iloc[position + 1])
        for position in pair_positions
    ]

    return pair_positions, np.asarray(spacings)


def mutual_hill_radius(planet_masses, semi_major_axes, system_mass):
    """Return the adjacent pair's mutual Hill radius [AU]."""
    pair_mass = planet_masses[0] + planet_masses[1]  # [M_earth]
    mean_axis = (semi_major_axes[0] + semi_major_axes[1]) / 2  # [AU]
    return (pair_mass / (3 * system_mass)) ** (1 / 3) * mean_axis


def adjacent_mutual_hill_radii(catalog):
    """Return row positions and mutual Hill radii for adjacent same-host pairs."""
    hostnames = catalog["hostname"]
    planet_masses = catalog["pl_bmasse"]
    semi_major_axes = catalog["pl_orbsmax"]
    stellar_masses = catalog["st_mass"]
    solar_mass_in_earth_masses = 0.33261191609863575142462501642665 * 10**6  # [M_earth / M_sun]
    hill_radii = []  # [AU]

    pair_positions = _adjacent_pair_positions(catalog)
    for position in pair_positions:
        hostname = hostnames.iloc[position]
        total_planet_mass = sum(
            planet_masses.iloc[index]
            for index, candidate in enumerate(hostnames)
            if candidate == hostname
        )
        system_mass = (
            total_planet_mass
            + stellar_masses.iloc[position] * solar_mass_in_earth_masses
        )  # [M_earth]
        hill_radii.append(
            mutual_hill_radius(
                planet_masses.iloc[position : position + 2].to_numpy(),
                semi_major_axes.iloc[position : position + 2].to_numpy(),
                system_mass,
            )
        )
    return pair_positions, np.asarray(hill_radii)


def initial_pairwise_positions(
    number_walkers,
    log_mass_location,
    log_mass_scale,
    gamma_location=1.0,
    gamma_scale=0.3,
):
    """Draw walker coordinates as (log10(mu_tilde), gamma), with 0 < gamma < 1."""
    log_mass = stats.norm.rvs(
        size=number_walkers, loc=log_mass_location, scale=log_mass_scale
    )
    gamma = []
    while len(gamma) != number_walkers:
        candidate = stats.norm.rvs(gamma_location, gamma_scale)
        if 0 < candidate < 1:
            gamma.append(candidate)
    return np.transpose(np.vstack((log_mass, gamma)))


def pair_mass_fractions(gamma):
    """Return the lower and upper masses as fractions of pair mass ``mu_tilde``."""
    gamma = np.asarray(gamma)
    inverse_total = (1 + gamma) ** (-1)
    return gamma * inverse_total, inverse_total


def adjacent_pair_mass_ratios(masses, stellar_mass):
    """Return each adjacent planet-mass sum divided by stellar mass."""
    masses = np.asarray(masses)
    return (masses[:-1] + masses[1:]) / stellar_mass


def log_K(log_mu_tilde, log_osep_observed):
    """Return the existing base-ten orbital-separation transform."""
    return log_osep_observed - (1 / 3) * log_mu_tilde + (1 / 3) * np.log10(3.0)


def log_mu_tilde_from_log_K(log_osep_observed, log_K_observed):
    """Invert ``log_K`` to recover the pair's base-ten mass ratio."""
    return 3 * (log_osep_observed - log_K_observed) + np.log10(3.0)


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
    from tdpy.numerics import logpdf_gaus

    return logpdf_gaus(value, mean, standard_deviation)


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
    log_gamma_dist = normal_log_density(
        np.log10(gamma), mean_gamma, std_gamma
    ) - np.log(gamma)
    return log_mutilde_dist + log_gamma_dist


def combine_log_posterior(theta, prior, likelihood):
    """Combine caller-supplied terms without evaluating likelihood outside prior support."""
    logprior = prior(theta)
    if not np.isfinite(logprior):
        return -np.inf
    loglikelihood = likelihood(theta)
    with np.errstate(over='ignore'):
        return logprior + loglikelihood
