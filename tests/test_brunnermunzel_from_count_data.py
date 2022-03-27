import math

import numpy as np
import pytest
from scipy.stats import brunnermunzel

from brunnermunzel_from_count_data import __version__, brunnermunzel_from_count_data


def test_version():
    assert __version__ == "1.0.0"


count_datas_match = [
    {0: 1, 1: 2, 2: 3},  # ints
    {0.5: 1, 1.5: 2, 2.5: 3},  # float
    {0: 1, 1: 2, 2.5: 3},  # mixed
    {np.nan: 1, 1: 2, 2.5: 3},  # contains nan value
    {0: 0, 1: 2, 2.5: 3},  # contains zero count
    {},  # empty
    {0: 0, 1: 0, 2.5: 0},  # empty
]


@pytest.mark.parametrize("x", count_datas_match)
@pytest.mark.parametrize("y", count_datas_match)
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
@pytest.mark.parametrize("distribution", ["t", "normal"])
@pytest.mark.parametrize("nan_policy", ["propagate", "raise", "omit"])
def test_matches_scipy_brunnermunzel(x, y, alternative, distribution, nan_policy):
    def count_data_to_array(c):
        return np.array([k for k, v in c.items() for _ in range(v)])

    def isclose(a: float, b: float) -> bool:
        return math.isclose(a, b) or (np.isnan(a) and np.isnan(b))

    try:
        expected = brunnermunzel(
            x=count_data_to_array(x),
            y=count_data_to_array(y),
            alternative=alternative,
            distribution=distribution,
            nan_policy=nan_policy,
        )
    except BaseException as e_expected:
        with pytest.raises(type(e_expected)):
            brunnermunzel_from_count_data(
                x=x,
                y=y,
                alternative=alternative,
                distribution=distribution,
                nan_policy=nan_policy,
            )
    else:
        actual = brunnermunzel_from_count_data(
            x=x,
            y=y,
            alternative=alternative,
            distribution=distribution,
            nan_policy=nan_policy,
        )

        assert isclose(actual.statistic, expected.statistic)
        assert isclose(actual.pvalue, expected.pvalue)


@pytest.mark.parametrize(
    "x, y",
    [
        ({0: -1, 1: 2}, {0: 1, 1: 2}),  # x
        ({0: 1, 1: 2}, {0: -1, 1: 2}),  # y
        ({0: -1, 1: 2}, {0: -1, 1: 2}),  # both
    ],
)
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
@pytest.mark.parametrize("distribution", ["t", "normal"])
@pytest.mark.parametrize("nan_policy", ["propagate", "raise", "omit"])
def test_raises_for_negative_counts(x, y, alternative, distribution, nan_policy):
    with pytest.raises(ValueError):
        brunnermunzel_from_count_data(
            x=x,
            y=y,
            alternative=alternative,
            distribution=distribution,
            nan_policy=nan_policy,
        )
