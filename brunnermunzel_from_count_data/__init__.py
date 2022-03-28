__version__ = "1.0.0"


from typing import Dict, Mapping, Tuple, Union

import numpy as np
from scipy.stats import distributions
from scipy.stats.stats import BrunnerMunzelResult, _contains_nan
from typing_extensions import Literal


value_t = Union[int, float]
count_t = int
count_data_t = Mapping[value_t, count_t]
rank_t = Union[int, float]


def join_count_data(*args: count_data_t) -> count_data_t:
    """
    Join count datas into single count data.

    Parameters
    ----------
    x1, x2, ... : Mapping[Union[int, float], int]
        Mappings from values to its occurence counts.

    Returns
    -------
    ret : Mapping[Union[int, float], int]
        The joined count data.

    Examples
    --------
    >>> from brunnermunzel_from_count_data import join_count_data
    >>> x1 = {1: 11, 2: 2, 4: 1}
    >>> x2 = {1: 3, 2: 1, 3: 4, 4: 2, 5: 1}
    >>> join_count_data(x1, x2)
    {1: 14, 2: 3, 4: 3, 3: 4, 5: 1}

    """
    ret: Dict[value_t, count_t] = {}

    for x in args:
        for v, c in x.items():
            ret[v] = ret.get(v, 0) + c

    return ret


def rank_count_data(
    x: count_data_t, method: Literal["average"] = "average"
) -> Mapping[value_t, rank_t]:
    """
    Assign ranks to values in count data, dealing with ties appropriately.

    Ranks begin at 1.  The `method` argument controls how ranks are assigned
    to equal values.

    This function is based on scipy.stats.rankdata.

    Parameters
    ----------
    x : Mapping[Union[int, float], int]
        Mapping from values to its occurence counts to be ranked.
    method : {'average'}, optional
        The method used to assign ranks to tied elements.
        The following methods are available (default is 'average'):

          * 'average': The average of the ranks that would have been assigned to
            all the tied values is assigned to each value.

    Returns
    -------
    ranks : Mapping[Union[int, float], Union[int, float]]
         Mapping from values to its rank.

    Examples
    --------
    >>> from brunnermunzel_from_count_data import rank_count_data
    >>> rank_count_data({0:1, 2: 2, 3: 1})
    {0: 1.0, 2: 2.5, 3: 4.0}

    """
    if method not in ("average"):
        raise ValueError('unknown method "{0}"'.format(method))

    r = 1
    minmaxranks: Dict[value_t, Tuple[int, int]] = {}
    for v in sorted(list(x.keys())):
        minmaxranks[v] = (r, r + x[v] - 1)
        r += x[v]

    return {v: (minrank + maxrank) / 2 for v, (minrank, maxrank) in minmaxranks.items()}


def brunnermunzel_from_count_data(
    x: count_data_t,
    y: count_data_t,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    distribution: Literal["t", "normal"] = "t",
    nan_policy: Literal["propagate", "raise", "omit"] = "propagate",
) -> BrunnerMunzelResult:
    """
    Compute the Brunner-Munzel test on count data x and y.

    When the count datas are expanded into arrays, this function is expected to
    return the same result as scipy.stats.brunnermunzel.

    Parameters
    ----------
    x, y : Mapping[Union[int, float], int]
        Mapping from values to its occurence counts.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

          * 'two-sided'
          * 'less': one-sided
          * 'greater': one-sided
    distribution : {'t', 'normal'}, optional
        Defines how to get the p-value.
        The following options are available (default is 't'):

          * 't': get the p-value by t-distribution
          * 'normal': get the p-value by standard normal distribution.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values

    Returns
    -------
    statistic : float
        The Brunner-Munzer W statistic.
    pvalue : float
        p-value assuming an t distribution. One-sided or
        two-sided, depending on the choice of `alternative` and `distribution`.

    Examples
    --------
    >>> from brunnermunzel_from_count_data import brunnermunzel_from_count_data
    >>> x1 = {1: 11, 2: 2, 4: 1}
    >>> x2 = {1: 3, 2: 1, 3: 4, 4: 2, 5: 1}
    >>> w, p_value = brunnermunzel_from_count_data(x1, x2)
    >>> w
    3.1374674823029505
    >>> p_value
    0.005786208666151538

    """
    # check both x and y
    if [c for c in x.values() if c < 0] or [c for c in y.values() if c < 0]:
        raise ValueError("The input contains negative counts")
    cnx, npx = _contains_nan((np.array(list(x.keys()))), nan_policy)
    cny, npy = _contains_nan(np.array(list(y.keys())), nan_policy)
    contains_nan = cnx or cny
    if npx == "omit" or npy == "omit":
        nan_policy = "omit"

    if contains_nan and nan_policy == "propagate":
        return BrunnerMunzelResult(np.nan, np.nan)
    elif contains_nan and nan_policy == "omit":
        x = {v: c for v, c in x.items() if not np.isnan(v)}
        y = {v: c for v, c in y.items() if not np.isnan(v)}

    nx = sum(list(x.values()))
    ny = sum(list(y.values()))
    if nx == 0 or ny == 0:
        return BrunnerMunzelResult(np.nan, np.nan)
    rankc = rank_count_data(join_count_data(x, y))
    rankcx_mean = sum([rankc[v] * c for v, c in x.items()]) / nx
    rankcy_mean = sum([rankc[v] * c for v, c in y.items()]) / ny
    rankx = rank_count_data(x)
    ranky = rank_count_data(y)
    rankx_mean = sum([rankx[v] * c for v, c in x.items()]) / nx
    ranky_mean = sum([ranky[v] * c for v, c in y.items()]) / ny

    Sx = sum(
        [
            pow(rankc[v] - rankx[v] - rankcx_mean + rankx_mean, 2) * c
            for v, c in x.items()
        ]
    )
    Sx /= nx - 1
    Sy = sum(
        [
            pow(rankc[v] - ranky[v] - rankcy_mean + ranky_mean, 2) * c
            for v, c in y.items()
        ]
    )
    Sy /= ny - 1

    wbfn = nx * ny * (rankcy_mean - rankcx_mean)
    wbfn /= (nx + ny) * np.sqrt(nx * Sx + ny * Sy)

    if distribution == "t":
        df_numer = np.power(nx * Sx + ny * Sy, 2.0)
        df_denom = np.power(nx * Sx, 2.0) / (nx - 1)
        df_denom += np.power(ny * Sy, 2.0) / (ny - 1)
        df = df_numer / df_denom
        p = distributions.t.cdf(wbfn, df)
    elif distribution == "normal":
        p = distributions.norm.cdf(wbfn)
    else:
        raise ValueError("distribution should be 't' or 'normal'")

    if alternative == "greater":
        pass
    elif alternative == "less":
        p = 1 - p
    elif alternative == "two-sided":
        p = 2 * np.min([p, 1 - p])
    else:
        raise ValueError("alternative should be 'less', 'greater' or 'two-sided'")

    return BrunnerMunzelResult(wbfn, p)
