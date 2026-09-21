"""The parser of ``options["search_acq_fcn"]`` when it holds a string."""

import re

import pytest

import pyvbmc.acquisition_functions
from pyvbmc.acquisition_functions import (
    AbstractAcqFcn,
    AcqFcnIMIQR,
    AcqFcnLog,
    AcqFcnVIQR,
)
from pyvbmc.acquisition_functions.utilities import string_to_acq


@pytest.mark.parametrize(
    "string, expected_type, expected_info",
    [
        ("AcqFcnLog", AcqFcnLog, {}),
        ("AcqFcnLog()", AcqFcnLog, {}),
        ("  AcqFcnLog()  ", AcqFcnLog, {}),
        ("AcqFcnVIQR", AcqFcnVIQR, {"quantile": 0.75, "loss": "iqr"}),
        ("AcqFcnVIQR()", AcqFcnVIQR, {"quantile": 0.75, "loss": "iqr"}),
        ("AcqFcnVIQR(0.9)", AcqFcnVIQR, {"quantile": 0.9, "loss": "iqr"}),
        (
            "AcqFcnVIQR(quantile=0.9)",
            AcqFcnVIQR,
            {"quantile": 0.9, "loss": "iqr"},
        ),
        (
            "AcqFcnVIQR(0.9, 'iqr_reduction')",
            AcqFcnVIQR,
            {"quantile": 0.9, "loss": "iqr_reduction"},
        ),
        (
            "AcqFcnVIQR(0.9, loss='iqr_reduction')",
            AcqFcnVIQR,
            {"quantile": 0.9, "loss": "iqr_reduction"},
        ),
        (
            "AcqFcnVIQR(quantile=0.9,loss='iqr_reduction')",
            AcqFcnVIQR,
            {"quantile": 0.9, "loss": "iqr_reduction"},
        ),
        (
            "AcqFcnVIQR(quantile=0.9, loss='iqr_reduction')",
            AcqFcnVIQR,
            {"quantile": 0.9, "loss": "iqr_reduction"},
        ),
        (
            "AcqFcnVIQR( quantile=0.9)",
            AcqFcnVIQR,
            {"quantile": 0.9, "loss": "iqr"},
        ),
        (
            "AcqFcnVIQR(quantile = 0.9)",
            AcqFcnVIQR,
            {"quantile": 0.9, "loss": "iqr"},
        ),
        (
            'AcqFcnVIQR(loss="iqr_reduction")',
            AcqFcnVIQR,
            {"quantile": 0.75, "loss": "iqr_reduction"},
        ),
        ("AcqFcnIMIQR(0.9)", AcqFcnIMIQR, {"quantile": 0.9}),
        ("AcqFcnIMIQR(quantile = 0.9)", AcqFcnIMIQR, {"quantile": 0.9}),
    ],
)
def test_a_call_is_read_as_python_reads_it(
    string, expected_type, expected_info
):
    """A class name, on its own or with literal positional and keyword
    arguments, wherever the spaces fall."""
    acq_fcn = string_to_acq(string)
    assert type(acq_fcn) is expected_type
    for key, value in expected_info.items():
        assert acq_fcn.acq_info[key] == value


class _RecordingAcq(AbstractAcqFcn):
    """An acquisition that keeps the arguments it was built with."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def _compute_acquisition_function(
        self,
        Xs,
        vp,
        gp,
        function_logger,
        optim_state,
        f_mu,
        f_s2,
        f_bar,
        var_tot,
    ):
        pass


def test_a_literal_holding_a_comma_or_an_equals_is_read_whole(monkeypatch):
    """The arguments are the ones Python would pass, so a string value is
    read whole however it is punctuated."""
    monkeypatch.setattr(
        pyvbmc.acquisition_functions,
        "_RecordingAcq",
        _RecordingAcq,
        raising=False,
    )
    acq_fcn = string_to_acq("_RecordingAcq('a,b', label='c=d')")
    assert acq_fcn.args == ("a,b",)
    assert acq_fcn.kwargs == {"label": "c=d"}


@pytest.mark.parametrize(
    "string",
    [
        "",
        "NoSuchAcquisition",
        "NoSuchAcquisition()",
        "string_to_acq()",
        "AcqFcnVIQR(",
        "AcqFcnVIQR(0.9",
        "AcqFcnVIQR(quantile=)",
        "AcqFcnVIQR(quantile)",
        "AcqFcnVIQR(loss=iqr_reduction)",
        "AcqFcnVIQR(**{'quantile': 0.9})",
        "AcqFcnVIQR(0.9) + 1",
        "AcqFcnVIQR(quantile=len('ab'))",
    ],
)
def test_what_cannot_be_read_raises_and_quotes_the_string(string):
    """Nothing is dropped in silence: a string that is not the name of an
    acquisition function class, with literal arguments, is refused."""
    with pytest.raises(ValueError, match=re.escape(repr(string))):
        string_to_acq(string)
