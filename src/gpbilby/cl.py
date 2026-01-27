import signal

import bilby
from bilby_pipe.utils import logger, CHECKPOINT_EXIT_CODE
from bilby_pipe.data_analysis import sighandler
import numpy as np

from .inputs import create_parser, GPBilbyInputs
from .data import get_data_dictionary
from .prior import get_priors
from .likelihood import get_likelihood


def log_version_information():
    import bilby
    import bilby_pipe
    import gpbilby

    logger.info(f"Running bilby_pipe version: {bilby_pipe.__version__}")
    logger.info(f"Running bilby version: {bilby.__version__}")
    logger.info(f"Running gpbilby version: {gpbilby.__version__}")


def main():
    log_version_information()

    logger.name = "GPBilby"
    parser = create_parser()
    args, unknown_args = parser.parse_known_args()
    inputs = GPBilbyInputs(args, unknown_args)

    data_dictionary = get_data_dictionary(inputs)

    skwargs = dict(
        sampler=inputs.sampler,
        outdir=inputs.result_directory,
        meta_data=inputs.meta_data,
        save=inputs.result_format,
        injection_parameters=inputs.meta_data["injection_parameters"],
        exit_code=CHECKPOINT_EXIT_CODE
    )
    skwargs.update(inputs.sampler_kwargs)

    # Start rthe alarm to checkpoint on the expected exit code
    signal.signal(signal.SIGALRM, handler=sighandler)
    signal.alarm(inputs.periodic_restart_time)

    if inputs.noise_only_run:
        noise_priors = get_priors(inputs, signal=False)
        noise_likelihood = get_likelihood(
            data_dictionary, inputs, noise_priors, signal=False
        )
        noise_result = bilby.run_sampler(
            noise_likelihood,
            noise_priors,
            label=inputs.label + "_noise",
            **skwargs
        )
        noise_result.plot_corner()

    if inputs.noise_only_run:
        log_noise_evidence = noise_result.log_evidence
    else:
        log_noise_evidence = np.nan

    signal_and_noise_priors = get_priors(inputs, signal=True)
    signal_and_noise_likelihood = get_likelihood(
        data_dictionary,
        inputs,
        signal_and_noise_priors,
        signal=True,
        log_noise_evidence=log_noise_evidence,
    )

    if inputs.sampler == "bilby_mcmc":
        small_weight = 1
        tiny_weight = 0.1
        signal_only_priors = signal_and_noise_priors.__class__(
            {k: v for k, v in signal_and_noise_priors.items() if "kernel" not in k}
        )
        noise_only_priors = signal_and_noise_priors.__class__(
            {k: v for k, v in signal_and_noise_priors.items() if "kernel" in k}
        )

        proposal_list = bilby.bilby_mcmc.proposals.get_proposal_cycle(
            inputs.sampler_kwargs.get("proposal_cycle", "default"), signal_only_priors
        ).proposal_list

        kernel_subset = list(noise_only_priors.keys())
        for detector in ["H1", "L1"]:
            for index in range(len(inputs.kernel_list)):
                if len(inputs.kernel_list) == 1:
                    match = f"{detector}-kernel"
                else:
                    match = f"{detector}-kernel:terms[{index}]"
                subset = [key for key in kernel_subset if match in key]
                if len(subset) > 0:
                    proposal_list.append(
                        bilby.bilby_mcmc.proposals.AdaptiveGaussianProposal(
                            signal_and_noise_priors,
                            weight=small_weight,
                            subset=kernel_subset,
                        )
                    )
                    proposal_list.append(
                        bilby.bilby_mcmc.proposals.DifferentialEvolutionProposal(
                            signal_and_noise_priors,
                            weight=small_weight,
                            subset=kernel_subset,
                        )
                    )
                    proposal_list.append(
                        bilby.bilby_mcmc.proposals.UniformProposal(
                            signal_and_noise_priors,
                            weight=tiny_weight,
                            subset=kernel_subset,
                        )
                    )

        proposals = bilby.bilby_mcmc.proposals.ProposalCycle(proposal_list)
        skwargs["proposal_cycle"] = proposals

    logger.info("Running the sampler")
    signal_and_noise_result = bilby.run_sampler(
        signal_and_noise_likelihood,
        signal_and_noise_priors,
        label=inputs.label,
        **skwargs,
    )

    # Generate all params and resave
    posterior = signal_and_noise_result.posterior
    logger.debug("Generating full posterior")
    posterior = bilby.gw.conversion.generate_all_bbh_parameters(posterior)

    if "zenith" in posterior and "azimuth" in posterior:
        logger.debug("Generating ra/dec posterior")
        reference_frame = inputs.whittle_likelihood.reference_frame
        ra_list = []
        dec_list = []
        for _, sample in posterior.iterrows():
            ra, dec = bilby.gw.utils.zenith_azimuth_to_ra_dec(
                sample["zenith"],
                sample["azimuth"],
                sample["geocent_time"],
                reference_frame,
            )
            ra_list.append(ra)
            dec_list.append(dec)

        posterior["ra"] = ra_list
        posterior["dec"] = dec_list

    signal_and_noise_result.posterior = posterior
    signal_and_noise_result.save_to_file(extension=inputs.result_format, overwrite=True)
