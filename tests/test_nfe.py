import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

# pipeline.py imports the harness at module load; stub it so the pure helpers are testable without a GPU stack.
_stub = types.ModuleType("pipelines.sd35.pipeline_stable_diffusion_3_methods")
_stub.StableDiffusion3MethodsPipeline = object
for _name in ("pipelines", "pipelines.sd35"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules.setdefault(_stub.__name__, _stub)

_spec = importlib.util.spec_from_file_location(
    "cfgmp_pipeline", Path(__file__).resolve().parents[1] / "cfgmp_eval" / "pipeline.py"
)
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)


def test_demo_defaults_cost_62_nfe():
    # 10 steps, 7 of them above t=0.6, 3 iterations x 2 forwards each, plus 2 forwards per sampling step
    assert pipeline.planned_nfe(10, 3, 0.6) == 62


def test_sigma_grid_matches_authors_scheduler():
    sigmas = pipeline.fine_grained_sigmas(10, 3.0)
    assert len(sigmas) == 21 and sigmas[0] == 1.0 and sigmas[-1] == 0.0
    assert np.all(np.diff(sigmas) < 0)


def test_anderson_reaches_fixed_point_of_linear_map():
    import torch

    torch.manual_seed(0)
    target = torch.randn(2, 4, 3, 3)
    g = lambda z: 0.5 * z + 0.5 * target  # contraction with fixed point `target`
    z, zs, fs = torch.zeros_like(target), [], []
    for _ in range(3):
        zs.append(z)
        fs.append(g(z) - z)
        zs, fs = zs[-2:], fs[-2:]
        z = pipeline.solve_anderson_mixing(zs, fs, beta=1.0, ridge=0.0)
    assert torch.allclose(z, target, atol=1e-4)
