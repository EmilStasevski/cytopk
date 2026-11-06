"""cytopk: IV PK/PD simulators and an FCS reader/writer (research prototype, not for clinical use)."""

from ._cytopk import (
    FcsFile,
    FcsReadError,
    models,
    read_fcs,
    read_fcs_all,
    read_fcs_bytes,
    read_fcs_many,
    simulate,
    simulate_ddm,
    write_fcs,
)

__all__ = [
    "FcsFile",
    "FcsReadError",
    "models",
    "read_fcs",
    "read_fcs_all",
    "read_fcs_bytes",
    "read_fcs_many",
    "simulate",
    "simulate_ddm",
    "simulate_twocmt_iv",
    "write_fcs",
]


def simulate_twocmt_iv(theta, times, dose_time, dose_amount, dose_duration, **kwargs):
    """Shorthand for ``simulate("twocmt_iv", ...)``. Kept for the go/no-go benchmark script."""
    return simulate("twocmt_iv", theta, times, dose_time, dose_amount, dose_duration, **kwargs)
