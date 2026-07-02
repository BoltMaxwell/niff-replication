"""NIFF replication — a self-contained reimplementation of the numerical
methods from Hao & Bilionis, *Neural Information Field Filter* (MSSP 2025).

Modules
-------
``niff.nsvi``       Nested stochastic variational inference (optimizer).
``niff.npsgld``     Nested (preconditioned) SGLD sampler; ``preconditioner`` ∈
                    {identity (=NSGLD), rmsprop (=NPSGLD), diag_fisher, dense_fisher}.
``niff.duffing_s51``  Paper §5.1 Duffing oscillator problem (data, fields, energies).
``niff.fields`` / ``niff.utils``  Fourier basis and collocation helpers.
"""

from niff.nsvi import NSVIConfig, NSVIResult, draw_nsvi_samples, run_nsvi
from niff.npsgld import NPSGLDConfig, NPSGLDResult, run_npsgld

__all__ = [
    "NSVIConfig",
    "NSVIResult",
    "run_nsvi",
    "draw_nsvi_samples",
    "NPSGLDConfig",
    "NPSGLDResult",
    "run_npsgld",
]
