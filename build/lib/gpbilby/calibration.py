import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
import lal

from scipy.interpolate import interp1d

from scipy.stats import skewnorm, norm
from scipy.optimize import minimize
from scipy.signal import windows


def fit_skewnorm(q16, q50, q84):
    def loss(params):
        alpha, loc, scale = params
        q16_est = skewnorm.ppf(0.16, alpha, loc=loc, scale=scale)
        q50_est = skewnorm.ppf(0.50, alpha, loc=loc, scale=scale)
        q84_est = skewnorm.ppf(0.84, alpha, loc=loc, scale=scale)
        return (q16_est - q16) ** 2 + (q50_est - q50) ** 2 + (q84_est - q84) ** 2

    initial_guess = [0.0, q50, (q84 - q16) / 2]
    bounds = [(-10, 10), (None, None), (1e-6, None)]
    result = minimize(loss, initial_guess, bounds=bounds)
    return result.x


def draw_realistations(
    sample_frequencies, envelope_frequencies, q16, q50, q84, nsim=100
):
    # Fit the skew-normal parameters for the given quantiles
    params_list = np.array(
        [fit_skewnorm(q16, q50, q84) for q16, q50, q84 in zip(q16, q50, q84)]
    )

    # Interpolate the fitted parameters to match the frequencies of the FFT
    interp_func = interp1d(
        envelope_frequencies,
        params_list,
        axis=0,
        bounds_error=False,
        fill_value="extrapolate",
    )
    params_list = interp_func(sample_frequencies)

    samples = np.zeros((nsim, len(sample_frequencies)))
    for ii in range(nsim):
        samples[ii] = np.array(
            [
                skewnorm.rvs(alpha, loc=loc, scale=scale, size=1)
                for alpha, loc, scale in params_list
            ]
        ).flatten()

    return samples


def apply_amplitude_and_phase_perturbations(h_f, mag_samples, phase_samples):

    # Apply amplitude perturbations
    perturbed_mags = np.abs(h_f) * mag_samples

    # Apply phase perturbations
    perturbed_phases = np.angle(h_f) + phase_samples

    # Combine magnitude and phase to get the perturbed frequency domain strain
    return perturbed_mags * np.exp(1j * perturbed_phases)


def measure_yerr_from_calibration_envelope(
    frequencies,
    h_f,
    psd,
    calibration_envelope,
    nsim=10,
    min_frequency=20,
    max_frequency=1024,
):

    # Extract values
    sampling_frequency = 2 * frequencies[-1]
    duration = 1 / (frequencies[1] - frequencies[0])

    # Whiten the data
    h_t_w = np.sqrt(2 * sampling_frequency / duration) * np.fft.irfft(h_f / np.sqrt(psd))

    # Read in the calibration envelope data
    envelope_frequencies, q50_mag, q50_phase, q16_mag, q16_phase, q84_mag, q84_phase = np.loadtxt(
        calibration_envelope, unpack=True
    )

    # Interpolate, fit skew-normal parameters, and draw samples
    mag_samples = draw_realistations(frequencies, envelope_frequencies, q16_mag, q50_mag, q84_mag, nsim=nsim)
    phase_samples = draw_realistations(
        frequencies, envelope_frequencies, q16_phase, q50_phase, q84_phase, nsim=nsim
    )

    # Apply amplitude and phase perturbations to the original frequency domain strain
    h_f_perturbed = apply_amplitude_and_phase_perturbations(
        h_f, mag_samples, phase_samples
    )

    # Apply the frequency mask to the perturbed strain
    freq_mask = (frequencies >= min_frequency) & (frequencies <= max_frequency)
    h_f_perturbed[:, ~freq_mask] = 0

    # Compute the whitened time domain strain from the perturbed frequency domain strain
    h_t_w_perturbed = np.sqrt(2 * sampling_frequency / duration) * np.fft.irfft(
        h_f_perturbed / np.sqrt(psd), axis=1
    )

    residuals = h_t_w - h_t_w_perturbed
    yerr = np.std(residuals, axis=0, ddof=1, out=None)

    return yerr
