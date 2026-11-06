"""Uniform adapters: path -> (channel names, events x parameters NumPy array). Each raises on failure.

Options are each library's defaults (fcsparser: channel names from $PnN, as every other reader here), except where a library has an explicit "be lenient" switch. Those switches
get a second, `_lenient` variant, because a user facing a broken file would try them.
flowCore runs in R through flowcore_reader.R and is driven from bench_*.py, not from here.
"""

from __future__ import annotations

import warnings

import numpy as np


def cytopk_lenient(p):
    import cytopk
    f = cytopk.read_fcs(p)
    return f.names, f.data


def cytopk_strict(p):
    import cytopk
    f = cytopk.read_fcs(p, strict=True)
    return f.names, f.data


def _flowio(p, **kw):
    import flowio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = flowio.FlowData(str(p), **kw)
        names = list(f.pnn_labels)
        return names, np.asarray(f.as_array(preprocess=False))


def flowio_default(p):
    return _flowio(p)


def flowio_lenient(p):
    return _flowio(p, ignore_offset_error=True, ignore_offset_discrepancy=True)


def _flowkit(p, **kw):
    import flowkit as fk
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = fk.Sample(str(p), **kw)
        return list(s.pnn_labels), s.get_events(source="raw")


def flowkit_default(p):
    return _flowkit(p)


def flowkit_lenient(p):
    return _flowkit(p, ignore_offset_error=True, ignore_offset_discrepancy=True)


def fcsparser_default(p):
    import fcsparser
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        meta, df = fcsparser.parse(str(p), reformat_meta=True, channel_naming="$PnN")
        return list(meta["_channel_names_"]), df.to_numpy()


READERS = {
    "cytopk": cytopk_lenient,
    "cytopk_strict": cytopk_strict,
    "flowio": flowio_default,
    "flowio_lenient": flowio_lenient,
    "flowkit": flowkit_default,
    "flowkit_lenient": flowkit_lenient,
    "fcsparser": fcsparser_default,
    "fcsparser_numpy1": fcsparser_default,  # same reader, run in a NumPy 1.26 environment (see bench_robustness)
}
IMPORT = {"cytopk": "cytopk", "cytopk_strict": "cytopk", "flowio": "flowio", "flowio_lenient": "flowio",
          "flowkit": "flowkit", "flowkit_lenient": "flowkit", "fcsparser": "fcsparser",
          "fcsparser_numpy1": "fcsparser"}
