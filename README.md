# Mass-Estimation

How massive can the planets in a multiplanetary system be before their orbits become unstable? We estimate maximum planet masses under different stability conditions and simulation methods.

## Local paths

From the checkout root, source `scripts/set_repository_path.sh` to set `MASS_ESTIMATION_PATH` in the current shell. Existing catalog files remain at the repository root; the ignored `data/` and `visuals/` directories are available for local inputs and generated figures.

## Pairwise models

The pairwise scripts sample `(log10(mu_tilde), gamma)`, where `mu_tilde` is the
sum of the two planet masses divided by the stellar mass and `gamma` is the
smaller planet mass divided by the larger. Their prior uses Gaussian terms in
these coordinates together with the density factors `1/mu_tilde` and `1/gamma`.
The single-pair script evaluates one observed separation. The all-pairs script
sums log-likelihood contributions from independent observed separations.

The multi-system model uses a different parameter order, `(gamma,
log10(mu_tilde))`, a distinct prior, and a base-10 likelihood convention. Its
parameters and likelihood should not be interchanged with those of the pairwise
scripts.

## Catalog input

The multi-system scripts use adjacent catalog rows to form planet pairs. Keep
each host's rows together and begin a selected block at its first row. The
scripts preserve that row order rather than sorting or regrouping planets. Host
identifiers must be present, and paired periods must be finite and positive.
Invalid or incomplete selections raise an error. A host's stellar mass is read
for that system.

## Planet-mass sampling

`MCMC_planet_masses.py` samples an initial planet mass and ratios between
neighboring masses. The prior bounds the first mass between 1 and 317.8 Earth
masses and each ratio strictly between zero and one. Reconstruction stores a
binary direction with each sampled state, so rejected proposals remain aligned
with their repeated coordinates. Direction zero divides by the neighboring mass
ratio. Direction one multiplies by it.

Reconstruction requires real, finite, positive masses and ratios. Direction
values must be unmasked zeros or ones. Observed log separations must be a
one-dimensional sequence with one value per neighboring pair. Invalid shapes,
masked inputs, and `NaN` values raise an error before random directions are
drawn. Infinite log separations retain the likelihood's limiting behavior.

## Interpreting results

Summary figures use the fitted samples and retain host and planet-pair labels.
The mass histogram reports sample counts, and the corner plot uses the full
sample mean for each derived `kappa` column. A short run checks that the workflow
executes. Assess chain convergence and the model assumptions before interpreting
the posterior.
