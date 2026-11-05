from scipy.stats import dirichlet, geom, truncnorm, laplace
from scipy import special
import numpy as np


def dirichlet_mixture_prior(alpha1, alpha2, composition, theta, w1=0.5, w2=0.5):
    """
    Computes prior probability density for
    a knapsack composition using a mixture
    of Dirichlet distributions.
    """
    k = len(composition)
    alpha1 = k * [alpha1]
    alpha2 = k * [alpha2]
    density = (
        w1 * dirichlet.pdf(composition, alpha1)
        + w2 * dirichlet.pdf(composition, alpha2)
    ) * geom.pmf(k, theta)
    return density


def composition_complexity_v1(
    x, w, z, alpha1=0.5, alpha2=2, theta=0.8, weighted=None, max_capacity=100
):
    """
    w ... worker capacities
    z ... task loads
    x ... binary decision matrix
    """
    if weighted is None:
        weights = np.ones(len(w))
    elif weighted == "absolute":
        weights = w / 100
    elif weighted == "relative":
        weights = w / np.sum(w)
    else:
        raise Exception("weighted argument can only be None, 'relative' or 'absolute'.")

    compositions = np.where(x == 0, 0, z[:, np.newaxis])
    w_rest = w - compositions.sum(axis=0)
    compositions = np.vstack([compositions, w_rest])
    compositions = compositions / compositions.sum(axis=0)
    cc = 0
    for i, comp in enumerate(compositions.T):
        comp = comp[comp != 0]
        cc += weights[i] * -np.log(dirichlet_mixture_prior(alpha1, alpha2, comp, theta))
    return cc


def cont_bern_log_norm(lam, l_lim=0.49, u_lim=0.51):
    # Compute the log normalizing constant of a continuous Bernoulli distribution in a numerically stable way.
    # Returns the log normalizing constant for lam in (0, l_lim) U (u_lim, 1) and a Taylor approximation in [l_lim, u_lim].

    # Check conditions to avoid evaluating log_norm near 0.5
    cut_lam = np.where(
        np.logical_or(lam < l_lim, lam > u_lim), lam, l_lim * np.ones_like(lam)
    )

    # Log normalizing constant computation for lam outside the interval [l_lim, u_lim]
    log_norm = np.log(np.abs(2.0 * np.arctanh(1 - 2.0 * cut_lam))) - np.log(
        np.abs(1 - 2.0 * cut_lam)
    )

    # Taylor series approximation near 0.5
    taylor = (
        np.log(2.0)
        + 4.0 / 3.0 * np.power(lam - 0.5, 2)
        + 104.0 / 45.0 * np.power(lam - 0.5, 4)
    )

    # Return the appropriate value based on the condition
    return np.where(np.logical_or(lam < l_lim, lam > u_lim), log_norm, taylor)


def cont_bern_log_pdf(x, lam):
    """
    Compute the log-PDF of the continuous Bernoulli distribution.

    Parameters:
    x : array-like
        The data points where the log-PDF is evaluated.
    lam : array-like
        The continuous Bernoulli distribution parameter.

    Returns:
    log_pdf : array-like
        The log-PDF of the continuous Bernoulli distribution at each data point.
    """
    # Ensure x and lam are numpy arrays for element-wise operations
    x = np.asarray(x)
    lam = np.asarray(lam)

    # Compute the log normalizing constant
    log_norm_const = cont_bern_log_norm(lam)

    # Compute the log-PDF using the formula: log_pdf = log C(lam) + x * log(lam) + (1 - x) * log(1 - lam)
    log_pdf = log_norm_const + special.xlogy(x, lam) + special.xlog1py(1 - x, -lam)

    return log_pdf


def truncated_laplace_logpdf(x, loc=0, scale=1, a=0, b=1):
    """
    Compute the log-PDF of a truncated Laplace distribution.

    Parameters:
    x (float): The value for which to compute the log-PDF.
    loc (float): The location parameter of the Laplace distribution.
    scale (float): The scale parameter of the Laplace distribution.
    a (float): The lower truncation bound.
    b (float): The upper truncation bound.

    Returns:
    float: The log-PDF value of the truncated Laplace distribution at x.
    """
    # Calculate the normalization constant
    cdf_a = laplace.cdf(a, loc=loc, scale=scale)
    cdf_b = laplace.cdf(b, loc=loc, scale=scale)
    normalization_constant = cdf_b - cdf_a

    # Compute the log-PDF, considering truncation
    if a <= x <= b:
        log_pdf = laplace.logpdf(x, loc=loc, scale=scale) - np.log(
            normalization_constant
        )
    else:
        log_pdf = (
            -np.inf
        )  # Log probability is negative infinity outside the truncation bounds

    return log_pdf


def logp_complexity_prior(
    C,
    E,
    geom_p=0.5,
    empty_param=0.05,
    comp_alpha=2.0,
    empty_distribution="normal",
    dirichlet_correction=False,
):
    """
    Compute the log-probability of the complexity prior, incorporating both the
    composition vector and the empty proportion value.

    Parameters:
    ----------
    C : array
        The composition vector. Can be empty. If not empty, it must sum to 1.0.
    E : float
        The empty proportion value. This must be within the range [0.0, 1.0].
    geom_p : float, optional
        The parameter for the geometric distribution, with a default of 0.5.
    empty_param : float, optional
        The scale parameter for the distributions used in the mixture. The default is 0.05.
    comp_alpha : float, optional
        The concentration parameter for the Dirichlet distribution. The default is 2.0.
    empty_distribution : str, optional
        The distribution type for the mixture of the empty proportion. Can be "normal",
        "laplace", or "cont_bern". The default is "normal".
    dirichlet_correction : bool, optional
        If True, applies a correction to the Dirichlet log probability by subtracting the
        log probability of a uniform Dirichlet distribution. The default is False.

    Returns:
    -------
    float
        The total log probability of the complexity prior.
    tuple of floats
        A tuple containing the individual log probabilities for the size of the
        composition vector (log_p_N), the composition vector given its size (log_p_C_given_N),
        and the empty proportion given the size of the composition vector (log_p_E_given_N).

    Raises:
    ------
    ValueError
        If `empty_distribution` is not one of "normal", "laplace", or "cont_bern".
    AssertionError
        If `C` is not empty and does not sum to 1.0, or if `E` is not within the range [0.0, 1.0],
        or if `C` is empty and `E` is not equal to 1.0.
    """
    if len(C) > 0:
        assert np.isclose(
            sum(C), 1.0
        ), "The composition vector C must sum to 1.0 if it is not empty."
    else:
        assert np.isclose(
            E, 1.0
        ), "If the composition vector C is empty, the empty proportion E must be 1.0."

    N = len(C)
    log_p_N = geom.logpmf(N, geom_p, loc=-1)
    if N == 0:
        log_p_C_given_N = 0.0
    else:
        log_p_C_given_N = dirichlet.logpdf(C, [comp_alpha] * N)
        if dirichlet_correction:
            log_p_C_given_N -= dirichlet.logpdf([1 / N] * N, [comp_alpha] * N)

    mix_w = 0.5
    if empty_distribution == "normal":
        a_trunc, b_trunc = 0.0, 1.0
        loc_lo, loc_hi = 0.0, 1.0
        a_lo, b_lo = (a_trunc - loc_lo) / empty_param, (b_trunc - loc_lo) / empty_param
        a_hi, b_hi = (a_trunc - loc_hi) / empty_param, (b_trunc - loc_hi) / empty_param
        log_p_E_given_N = np.logaddexp(
            np.log(mix_w)
            + truncnorm.logpdf(E, a_lo, b_lo, loc=loc_lo, scale=empty_param),
            np.log(1 - mix_w)
            + truncnorm.logpdf(E, a_hi, b_hi, loc=loc_hi, scale=empty_param),
        )
    elif empty_distribution == "laplace":
        a_trunc, b_trunc = 0.0, 1.0
        loc_lo, loc_hi = 0.0, 1.0
        log_p_E_given_N = np.logaddexp(
            np.log(mix_w)
            + truncated_laplace_logpdf(
                E, loc=loc_lo, scale=empty_param, a=a_trunc, b=b_trunc
            ),
            np.log(1 - mix_w)
            + truncated_laplace_logpdf(
                E, loc=loc_hi, scale=empty_param, a=a_trunc, b=b_trunc
            ),
        )
    elif empty_distribution == "cont_bern":
        lam_lo, lam_hi = empty_param, 1 - empty_param
        log_p_E_given_N = np.logaddexp(
            np.log(mix_w) + cont_bern_log_pdf(E, lam_lo),
            np.log(1 - mix_w) + cont_bern_log_pdf(E, lam_hi),
        )
    else:
        raise ValueError(
            "empty_distribution must be either 'normal', 'laplace', or 'cont_bern'"
        )

    # Total log-probability
    return log_p_N + log_p_C_given_N + log_p_E_given_N, (
        log_p_N,
        log_p_C_given_N,
        log_p_E_given_N,
    )


def get_C_and_E(x, w, z, i_w):
    """
    Compute relative empty space and composition values between 0 and 1.

    Parameters:
    ----------
    x : array
        The binary decision matrix.
    w : array
        The worker capacities.
    z : array
        The task loads.
    i_w : int
        The worker index.

    Returns:
    -------
    tuple
        The composition array and the proportion of empty space.
    """
    C_raw = z * x[:, i_w]
    C_raw = C_raw[C_raw != 0]  # remove empty ones
    if len(C_raw) > 0:
        C = C_raw / C_raw.sum()  # normalize
    else:
        C = C_raw
    E = (w[i_w] - np.sum(C_raw)) / w[i_w]
    return C, E


def composition_complexity_v2(
    x, w, z, geom_p, empty_param, comp_alpha, empty_distribution, dirichlet_correction
):
    """
    Compute complexity of a solution based on a complex distribution as the average of all workers'
    negative log probability (i.e., worker size is not taken into account).

    Parameters:
    ----------
    x : array
        The binary decision matrix.
    w : array
        The worker capacities.
    z : array
        The task loads.
    geom_p : float, optional
        The parameter for the geometric distribution, with a default of 0.5.
    empty_param : float, optional
        The scale parameter for the distributions used in the mixture. The default is 0.05.
    comp_alpha : float, optional
        The concentration parameter for the Dirichlet distribution. The default is 2.0.
    empty_distribution : str, optional
        The distribution type for the mixture of the empty proportion. Can be "normal",
        "laplace", or "cont_bern". The default is "normal".
    dirichlet_correction : bool, optional
        If True, applies a correction to the Dirichlet log probability by subtracting the
        log probability of a uniform Dirichlet distribution. The default is False.

    Returns:
    -------
    float
        The average negative log probability of all workers.
    """

    # iterate over all workers of a solution
    complexity = 0
    n_workers = len(w)
    for i_worker in range(n_workers):
        C, E = get_C_and_E(x, w, z, i_worker)
        neg_logp = -logp_complexity_prior(
            C=C,
            E=E,
            geom_p=geom_p,
            empty_param=empty_param,
            comp_alpha=comp_alpha,
            empty_distribution=empty_distribution,
            dirichlet_correction=dirichlet_correction,
        )[0]
        weight = 1
        complexity += neg_logp * weight
    # average
    complexity = complexity / n_workers

    return complexity
