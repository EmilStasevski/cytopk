"""cytopk: fast IV PK/PD simulators (research prototype, not for clinical use)."""

from ._cytopk import models, simulate, simulate_ddm

__all__ = ["models", "simulate", "simulate_ddm", "simulate_twocmt_iv"]


def simulate_twocmt_iv(theta, times, dose_time, dose_amount, dose_duration, **kwargs):
    """Shorthand for ``simulate("twocmt_iv", ...)``. Kept for the go/no-go benchmark script."""
    return simulate("twocmt_iv", theta, times, dose_time, dose_amount, dose_duration, **kwargs)
