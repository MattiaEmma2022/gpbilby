import copy

from bilby.core.prior import PriorDict, ConditionalBeta, Prior, Beta
from bilby_pipe.utils import get_function_from_string_path
import numpy as np


class MinimumPrior(ConditionalBeta):
    def __init__(
        self,
        order,
        minimum=0,
        maximum=1,
        name=None,
        latex_label=None,
        unit=None,
        boundary=None,
    ):
        super(MinimumPrior, self).__init__(
            alpha=1,
            beta=order,
            minimum=minimum,
            maximum=maximum,
            name=name,
            latex_label=latex_label,
            unit=unit,
            boundary=boundary,
            condition_func=self.minimum_condition,
        )
        self.order = order
        split = self.name.split(":")
        knum = split[1]
        knumm1 = int(knum.replace("terms[", "").replace("]", "")) - 1
        split[1] = f"terms[{knumm1}]"
        self.reference_name = ":".join(split)
        self._required_variables = [self.reference_name]

    def minimum_condition(self, reference_params, **kwargs):
        return dict(minimum=kwargs[self.reference_name] + 1e-2)

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        return Prior.get_instantiation_dict(self)


def get_priors(inputs, signal=True):
    if signal:
        priors = inputs.priors

        if inputs.default_prior in inputs.combined_default_prior_dicts.keys():
            prior_class = inputs.combined_default_prior_dicts[inputs.default_prior]
        elif "." in inputs.default_prior:
            prior_class = get_function_from_string_path(inputs.default_prior)
        else:
            raise ValueError("Unable to set prior: default_prior unavailable")
    else:
        priors = {}
        prior_class = PriorDict

    for detector in inputs.detectors:
        if len(inputs.kernel_list) == 1:
            singular = True
        else:
            singular = False

        for idx, kernel_name in enumerate(inputs.kernel_list):
            if singular:
                idx = "singular"

            if kernel_name == "matern32term":
                priors = add_priors_by_names(
                    ["log_sigma", "log_rho"], priors, inputs, detector, idx
                )
            elif kernel_name == "realterm":
                priors = add_priors_by_names(
                    ["log_a", "log_c"], priors, inputs, detector, idx
                )
            elif kernel_name == "jitterterm":
                priors = add_priors_by_names(
                    ["log_sigma"], priors, inputs, detector, idx
                )
            elif kernel_name == "fshoterm":
                priors = add_priors_by_names(
                    ["log_Q", "frequency", "log_S0"], priors, inputs, detector, idx
                )
            else:
                raise ValueError(f"Kernel {kernel_name} not implemented")

    priors = prior_class(priors)
    return priors


latex_lookup = dict(
    log_sigma="$\\log(\\sigma_{{{detector}}}^{{{idx}}})$",
    log_rho="$\\log(\\rho_{{{detector}}}^{{{idx}}})$",
    log_a="$\\log(a_{{{detector}}}^{{{idx}}})$",
    log_c="$\\log(c_{{{detector}}}^{{{idx}}})$",
    log_Q="$\\log(Q_{{{detector}}}^{{{idx}}})$",
    log_omega0="$\\log(\\omega_{{{detector}}}^{{{idx}}})$",
    frequency="$f_{{{detector}}}^{{{idx}}}$",
    log_S0="$\\log(S_{{{detector}}}^{{{idx}}})$",
)


def get_generic_prior(name, inputs, detector, idx, i_shot_dict=None):
    if idx == "singular":
        key = f"{detector}-kernel:{name}"
    else:
        key = f"{detector}-kernel:terms[{idx}]:{name}"

    if i_shot_dict is None:
        inputs.i_shot_dict = {}

    if name == "frequency":
        minimum = inputs.kernel_prior_dict[name].minimum
        maximum = inputs.kernel_prior_dict[name].maximum

        n_shots = np.sum(np.array(inputs.kernel_list) == "fshoterm")
        i_shot = inputs.i_shot_dict.get(detector, -1) + 1
        latex_label = latex_lookup[name].format(detector=detector, idx=idx)
        if i_shot == 0:
            prior = Beta(
                minimum=minimum,
                maximum=maximum,
                alpha=1,
                beta=n_shots,
                name=key,
                latex_label=latex_label,
            )
        else:
            prior = MinimumPrior(
                order=n_shots - i_shot,
                minimum=minimum,
                maximum=maximum,
                name=key,
                latex_label=latex_label,
            )
            # Hack to fix the class naming
            prior.__class__.__name__ = "MinimumPrior"
        inputs.i_shot_dict[detector] = i_shot
    else:
        prior = copy.copy(inputs.kernel_prior_dict[name])
        prior.latex_label = latex_lookup[name].format(detector=detector, idx=idx)
        prior.name = key
    return key, prior, i_shot_dict


def add_priors_by_names(names, priors, inputs, detector, idx):
    i_shot_dict = None
    for name in names:
        key, prior, i_shot_dict = get_generic_prior(
            name, inputs, detector, idx, i_shot_dict
        )
        priors[key] = prior
    return priors
