import argparse
import glob
import sys
import logging

import bilby
import bilby_pipe
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

from .inputs import create_parser, GPBilbyInputs
from .data import get_data_dictionary
from .prior import get_priors
from .likelihood import get_likelihood

plt.rcParams.update(
    {
        # "text.usetex": True,
        # "font.serif": "Computer Modern Roman",
        # "font.family": "serif",
        "font.size": 8,
        "axes.grid": True,
        "grid.color": "gray",
        "grid.linewidth": 0.1,
        "axes.axisbelow": True,
        "axes.edgecolor": "black",
    }
)


def plot_corner(result, keys, label_append, inputs):

    priors = result.priors
    variable_keys = [
        key for key in priors.keys() if key in keys and key in priors.non_fixed_keys
    ]

    if inputs.injection_dict:
        parameters = {}
        for key in variable_keys:
            if key in inputs.injection_dict:
                parameters[key] = inputs.injection_dict[key]
            else:
                parameters[key] = None
    else:
        parameters = variable_keys

    filename = f"{inputs.result_directory}/{result.label}_corner-{label_append}"
    if len(parameters) > 0:
        result.plot_corner(parameters=parameters, filename=filename)
        plt.clf()


def plot_data(data_dictionary, inputs, likelihood_dictionary=dict(), **kwargs):
    for det in data_dictionary:
        if det in likelihood_dictionary:
            likelihood = likelihood_dictionary[det]
        else:
            likelihood = None
        plot_per_ifo_data(
            data_dictionary[det], inputs=inputs, likelihood=likelihood, **kwargs
        )


def plot_per_ifo_data(
    strain,
    inputs,
    label_append=None,
    posterior=None,
    likelihood=None,
    color="C4",
    plot_median=True,
    intervals=[0.95],
    full_span=False,
    add_gp_prediction=False,
    add_signal_prediction=False,
    outdir=None,
    simulation_parameters=None,
):
    label_list = [strain.detector, inputs.label, "whitened_TD"]
    if label_append is not None:
        label_list.append(label_append)
    plot_label = "_".join(label_list)

    fig, ax = plt.subplots(figsize=(3.375, 2))
    ax.plot(
        strain.x_shifted,
        strain.y,
        color=strain.color,
        label=f"{strain.detector}",
        zorder=0,
        lw=0.5,
    )
    if "whiten" in inputs.cleaning:
        ax.set_ylabel(r"Whitened Strain")
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    else:
        ax.set_ylabel(r"$h(t)$")

    ax.set_xlabel(f"Time - {strain.trigger_time} [s]")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(10))

    if add_gp_prediction:
        maxL_sample = dict(posterior.iloc[np.argmax(posterior.log_likelihood)])
        maxL_sample = {
            key: val for key, val in maxL_sample.items() if key in likelihood.parameters
        }
        # idxs = (strain.x_shifted> inputs.plot_xmin) & (strain.x_shifted < inputs.plot_xmax)
        mean, var = likelihood.predict(strain.x, maxL_sample)
        x_shifted = strain.x_shifted
        std = np.sqrt(var)
        ax.fill_between(
            x_shifted,
            mean - 5 * std,
            mean + 5 * std,
            lw=0,
            color="C1",
            alpha=0.4,
            zorder=100,
        )

        # ypred, ypred_var = likelihood.predict(strain.x, maxL_sample)
        # res = ypred - strain.y[idxs]
        # ypred_std = np.sqrt(ypred_var)
        # normal_test_result = normaltest(res)

        # fig1, ax1 = plt.subplots()
        # ax1.plot(x_shifted, res, 'k')
        ##ax1.fill_between(x_shifted, res - ypred_std, res + ypred_std, color='k', alpha=0.4)
        # ax1.set_xlabel(f"Time - {strain.trigger_time} [s]")
        # ax1.set_ylabel("Residual")
        # ax1.xaxis.set_major_locator(ticker.MaxNLocator(5))
        # ax1.xaxis.set_minor_locator(ticker.MaxNLocator(10))
        # ax1.set_title(f"Residual normal test $p$={normal_test_result.pvalue:1.2e}")
        # fname = f"{inputs.data_directory}/{plot_label}_residual"
        # fig1.tight_layout()
        # fig1.savefig(fname, dpi=700)

    if simulation_parameters:
        p_simulation = likelihood.predict_signal(strain.x, simulation_parameters)
        ax.plot(
            strain.x_shifted,
            p_simulation,
            color="k",
            label="Simulation",
            alpha=0.8,
            lw=0.5,
            ls="-",
            zorder=1000,
        )

    if add_signal_prediction and posterior is not None and likelihood is not None:
        N = 100
        signal_predictions = []
        for ii in range(N):
            sample = dict(posterior.sample())
            p_signal = likelihood.predict_signal(strain.x, sample)
            signal_predictions.append(p_signal * strain.scale)

        if plot_median:
            ax.plot(
                strain.x_shifted,
                np.quantile(signal_predictions, 0.5, axis=0),
                color=color,
                label="signal",
                alpha=0.8,
                lw=0.75,
            )

        for p in intervals:
            ax.fill_between(
                strain.x_shifted,
                np.quantile(signal_predictions, 0.5 - p / 2, axis=0),
                np.quantile(signal_predictions, 0.5 + p / 2, axis=0),
                color=color,
                alpha=0.5,
                edgecolor=None,
                zorder=10000,
            )

    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))

    if full_span is False:
        if inputs.plot_xmin is not None:
            xmin = inputs.plot_xmin
        else:
            xmin = strain.x_shifted[0]
        if inputs.plot_xmax is not None:
            xmax = inputs.plot_xmax
        else:
            xmax = strain.x_shifted[-1]
        ax.set_xlim(xmin, xmax)
    else:
        plot_label += "_full"

    if inputs.plot_ymin and inputs.plot_ymax:
        ax.set_ylim(inputs.plot_ymin, inputs.plot_ymax)

    if outdir is None:
        outdir = inputs.data_directory
    fname = f"{outdir}/{plot_label}"
    print(f"Saving file to {fname}.png")
    fig.tight_layout()
    fig.savefig(fname + ".png", format="png", dpi=700)
    return fig, ax


def cl_plot():
    # Switch off logging
    for module in ["bilby", "bilby_pipe"]:
        logger = logging.getLogger(module)
        logger.setLevel("WARNING")

    # Set up an argument parser for this command line tool
    parser = argparse.ArgumentParser()
    parser.add_argument("result_file", help="The final result file to analyse")
    parser.add_argument("-o", "--outdir", default=None, type=str, help="")
    parser.add_argument("-g", "--add-gp-prediction", action="store_true")
    parser.add_argument("-i", "--add-injection", action="store_true")
    parser.add_argument("-s", "--add-signal-prediction", action="store_true")
    parser.add_argument("--plot-xmin", type=float, default=None)
    parser.add_argument("--plot-xmax", type=float, default=None)
    parser.add_argument("--plot-ymin", type=float, default=None)
    parser.add_argument("--plot-ymax", type=float, default=None)

    args = parser.parse_args()

    # Read in the result file and extract the data
    result = bilby.core.result.read_in_result(args.result_file)
    outdir = result.meta_data["command_line_args"]["outdir"]
    data_dump_file = result.meta_data["data_dump"]

    # Wrapper to recrate the inputs for a standard GPBilby analysis
    parser = create_parser()
    sys.argv[1] = glob.glob(f"{outdir}/*ini")[0]
    fargs, funknown_args = parser.parse_known_args()
    fargs.data_dump_file = data_dump_file
    inputs = GPBilbyInputs(fargs, funknown_args)

    inputs.plot_xmin = args.plot_xmin
    inputs.plot_xmax = args.plot_xmax
    inputs.plot_ymin = args.plot_ymin
    inputs.plot_ymax = args.plot_ymax

    # Extract the data, priors, and likelihood
    data_dictionary = get_data_dictionary(inputs)
    signal_and_noise_priors = get_priors(inputs, signal=True)
    signal_and_noise_likelihood = get_likelihood(
        data_dictionary,
        inputs,
        signal_and_noise_priors,
        signal=True,
        log_noise_evidence=np.nan,
    )

    label_append = "data"
    # Convert the injection dict if required
    if fargs.injection_dict is not None and args.add_injection:
        injection_dict = bilby_pipe.utils.convert_string_to_dict(fargs.injection_dict)
        label_append += "_simulation"
    else:
        injection_dict = None

    if args.add_signal_prediction:
        label_append += "_signal-prediction"

    if args.add_gp_prediction:
        label_append += "_gp-prediction"

    plot_data(
        data_dictionary,
        inputs,
        posterior=result.posterior,
        likelihood_dictionary=signal_and_noise_likelihood.likelihood_dictionary,
        label_append=label_append,
        add_gp_prediction=args.add_gp_prediction,
        add_signal_prediction=args.add_signal_prediction,
        outdir=f"{outdir}/data",
        simulation_parameters=injection_dict,
    )
