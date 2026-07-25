"""
Parity tests for the flattened Arkouda attribution pipeline.

The Arkouda branch of ``AttributionEngine.compute`` was rewritten to be
loop-free (constant server command count in ranks×depths) so it scales across
locales instead of anti-scaling on communication. Because that branch is written
purely in terms of the backend-neutral ``ak`` proxy, it can be exercised on the
pandas backend by forcing ``get_backend() == 'arkouda'``. The pandas fast path
(``get_backend() == 'pandas'``) is the correctness oracle: the two must produce
identical ``Value`` columns for every mode.

These run anywhere pandas/numpy are installed — no Arkouda server needed.
"""
import os
import sys
import types

import numpy as np
import pytest

# The package __init__ imports .visualizer, which pulls in plotly/seaborn/mpl.
# Stub them so the attribution core is importable in a headless test env.
for _name in ('plotly', 'plotly.graph_objects', 'plotly.express',
              'plotly.subplots', 'seaborn', 'matplotlib', 'matplotlib.pyplot'):
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        _mod.__getattr__ = lambda n: (lambda *a, **k: None)  # type: ignore[attr-defined]
        sys.modules[_name] = _mod

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import re  # noqa: E402
import ampere as A  # noqa: E402
from ampere import (  # noqa: E402
    AttributionEngine, Ensemble, MetricConfig, MetricType, set_backend,
)

set_backend('pandas')
from ampere._backend import ak  # noqa: E402

TRACE = os.path.join(os.path.dirname(__file__), '..', 'examples', 'hpl-trace')
CONFIGS = {
    re.compile(r".*energy.*"): MetricConfig(MetricType.CUMULATIVE, scale_factor=1e-6),
    re.compile(r".*power.*"):  MetricConfig(MetricType.INSTANTANEOUS, scale_factor=1e-6),
}
METRIC = 'A2rocm_smi:::energy_count:device=0'


def _load_ranks():
    """Load the two ranks that share device 0 from the example HPL trace."""
    node_ranks = {'Node0': [f'MPI Rank {i}' for i in range(8)]}
    ens = Ensemble.from_trace_paths([TRACE], node_ranks, CONFIGS)
    node = ens.runs[0].nodes[0]
    ranks = [r for r in node.ranks if r.name in ('MPI Rank 0', 'MPI Rank 1')]
    return node.metrics.get(METRIC), ranks


def _compute(metric, ranks, *, force_arkouda, **kwargs):
    orig = A.get_backend
    if force_arkouda:
        A.get_backend = lambda: 'arkouda'
    try:
        return AttributionEngine.compute(metric, ranks, **kwargs)
    finally:
        A.get_backend = orig


def _values(res):
    return {k: np.asarray(v['Value'], dtype=np.float64) for k, v in res.items()
            if v.size > 0}


@pytest.fixture(scope='module')
def loaded():
    metric, ranks = _load_ranks()
    assert metric is not None, f"metric {METRIC!r} not found in trace"
    assert len(ranks) == 2
    return metric, ranks


@pytest.mark.parametrize('concurrency_mode', ['shared', 'independent'])
@pytest.mark.parametrize('strategy', ['inclusive', 'exclusive'])
@pytest.mark.parametrize('output_mode', ['quantity', 'rate', 'min', 'max'])
def test_arkouda_branch_matches_pandas_oracle(loaded, concurrency_mode, strategy, output_mode):
    metric, ranks = loaded
    kw = dict(concurrency_mode=concurrency_mode, strategy=strategy, output_mode=output_mode)
    oracle = _values(_compute(metric, ranks, force_arkouda=False, **kw))
    flat   = _values(_compute(metric, ranks, force_arkouda=True,  **kw))
    assert set(oracle) == set(flat)
    for name in oracle:
        np.testing.assert_allclose(
            flat[name], oracle[name], rtol=1e-9, atol=1e-9,
            err_msg=f"rank {name} mismatch ({concurrency_mode}/{strategy}/{output_mode})",
        )


@pytest.mark.parametrize('strategy', ['inclusive', 'exclusive'])
def test_time_profile_matches_pandas_oracle(loaded, strategy):
    """metric=None path (time profiling)."""
    _, ranks = loaded
    kw = dict(concurrency_mode='independent', strategy=strategy)
    oracle = _values(_compute(None, ranks, force_arkouda=False, **kw))
    flat   = _values(_compute(None, ranks, force_arkouda=True,  **kw))
    for name in oracle:
        np.testing.assert_allclose(flat[name], oracle[name], rtol=1e-9, atol=1e-9)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
