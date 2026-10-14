r"""Preconditioners for spherical-coordinate time propagation."""

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from ._validation import _orbitals, _positive_real
from .atomic_core import SphericalCoreHamiltonian


class ShiftedSphericalCorePreconditioner:
    r"""Apply angular-channel solves for a shifted central core operator.

    This approximates the inverse Crank--Nicolson matrix by

    ``[I + i*step_size*H_core/2]**-1``.

    ``step_size`` is the duration of the Crank--Nicolson substep. For a
    `Strang2` outer step of length ``dt``, it is ``dt/2``. Each radial block
    is LU-factorized once and reused for every orbital and Krylov iteration.
    """

    def __init__(self, core, step_size):
        if not isinstance(core, SphericalCoreHamiltonian):
            raise TypeError("core must be a SphericalCoreHamiltonian.")
        self.core = core
        self.step_size = _positive_real(step_size, "step_size")
        identity = np.eye(len(core.r), dtype=np.complex128)
        self._factors = {
            l: lu_factor(identity + 0.5j * self.step_size * core.radial_matrix(l))
            for l in range(core.l_max + 1)
        }

    def __call__(self, orbitals):
        """Apply the preconditioner and preserve a single or batched shape."""
        values, single = _orbitals(
            orbitals, self.core.orbital_shape, allow_single=True
        )
        result = np.empty_like(values)
        for channel, (l, _) in enumerate(self.core.orbital_lm):
            result[:, :, channel] = lu_solve(
                self._factors[l], values[:, :, channel].T
            ).T
        return result[0] if single else result


class ShiftedAtomicCorePreconditioner(ShiftedSphericalCorePreconditioner):
    """Backward-compatible name for the shifted spherical-core preconditioner."""
