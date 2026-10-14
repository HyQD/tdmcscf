r"""Matrix-free Crank--Nicolson propagation for linear equations."""

from collections.abc import Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres

from ._validation import finite_real, matching_array, positive_real, state_array


class LinearSolveError(RuntimeError):
    """Raised when an iterative Crank--Nicolson solve does not converge."""


class CrankNicolson:
    r"""Implicit midpoint propagation of a linear differential equation.

    ``linear_rhs(state, time)`` represents ``L(t) state`` in

    ``dstate/dt = L(t) state``.

    One step of length ``dt`` solves

    ``(I - dt*L(t + dt/2)/2) y_new = (I + dt*L(t + dt/2)/2) y``.

    Thus, for ``i dy/dt = H(t)y``, supply ``linear_rhs = -1j * H``. The
    callback is required to be linear in its state argument. Arrays may have
    any nonempty shape and are flattened only inside the Krylov solver.

    Parameters
    ----------
    solver : {"gmres", "bicgstab"}, optional
        SciPy Krylov solver. Defaults to GMRES.
    rtol, atol : real scalar, optional
        Relative and absolute residual tolerances passed to SciPy.
    maxiter : positive int, optional
        Maximum Krylov iterations. ``None`` uses SciPy's default.
    restart : positive int, optional
        GMRES restart length. Ignored by BiCGSTAB.
    preconditioner : callable or scipy.sparse.linalg.LinearOperator, optional
        Approximate inverse of the left-hand-side matrix. A callable receives
        an array with the original state shape and must return that shape.
    """

    _SOLVERS = {"gmres": gmres, "bicgstab": bicgstab}

    def __init__(
        self,
        *,
        solver="gmres",
        rtol=1e-10,
        atol=0.0,
        maxiter=None,
        restart=None,
        preconditioner=None,
    ):
        if solver not in self._SOLVERS:
            choices = ", ".join(sorted(self._SOLVERS))
            raise ValueError(f"solver must be one of: {choices}.")
        self.solver = solver
        self.rtol = positive_real(rtol, "rtol", allow_zero=True)
        self.atol = positive_real(atol, "atol", allow_zero=True)
        if self.rtol == self.atol == 0:
            raise ValueError("At least one of rtol and atol must be positive.")
        if maxiter is not None:
            if isinstance(maxiter, (bool, np.bool_)) or not isinstance(
                maxiter, (int, np.integer)
            ):
                raise TypeError("maxiter must be a positive integer or None.")
            if maxiter <= 0:
                raise ValueError("maxiter must be positive.")
            maxiter = int(maxiter)
        if restart is not None:
            if isinstance(restart, (bool, np.bool_)) or not isinstance(
                restart, (int, np.integer)
            ):
                raise TypeError("restart must be a positive integer or None.")
            if restart <= 0:
                raise ValueError("restart must be positive.")
            restart = int(restart)
        if preconditioner is not None and not (
            callable(preconditioner) or isinstance(preconditioner, LinearOperator)
        ):
            raise TypeError("preconditioner must be callable or a LinearOperator.")
        self.maxiter = maxiter
        self.restart = restart
        self.preconditioner = preconditioner
        self.last_iterations = None
        self.last_residual_norm = None

    def _preconditioner(self, shape):
        if self.preconditioner is None:
            return None
        size = int(np.prod(shape))
        if isinstance(self.preconditioner, LinearOperator):
            if self.preconditioner.shape != (size, size):
                raise ValueError(
                    "preconditioner LinearOperator shape does not match the state."
                )
            return self.preconditioner

        def matvec(vector):
            result = self.preconditioner(vector.reshape(shape))
            return matching_array(result, shape, "preconditioner").ravel()

        return LinearOperator((size, size), matvec=matvec, dtype=np.complex128)

    def __call__(self, state, time, step_size, linear_rhs: Callable):
        """Advance one implicit midpoint step without modifying *state*."""
        if not callable(linear_rhs):
            raise TypeError("linear_rhs must be callable.")
        values = state_array(state)
        shape = values.shape
        size = values.size
        time = finite_real(time, "time")
        step_size = finite_real(step_size, "step_size")
        midpoint = time + 0.5 * step_size

        def evaluate(argument):
            result = linear_rhs(argument, midpoint)
            return matching_array(result, shape, "linear_rhs")

        rhs = values + 0.5 * step_size * evaluate(values)

        def matvec(vector):
            argument = vector.reshape(shape)
            return (argument - 0.5 * step_size * evaluate(argument)).ravel()

        matrix = LinearOperator(
            (size, size), matvec=matvec, dtype=np.complex128
        )
        iteration_count = 0

        def callback(_):
            nonlocal iteration_count
            iteration_count += 1

        kwargs = {
            "x0": values.ravel(),
            "M": self._preconditioner(shape),
            "rtol": self.rtol,
            "atol": self.atol,
            "maxiter": self.maxiter,
            "callback": callback,
        }
        if self.solver == "gmres":
            kwargs["restart"] = self.restart
            kwargs["callback_type"] = "pr_norm"

        solution, info = self._SOLVERS[self.solver](matrix, rhs.ravel(), **kwargs)
        self.last_iterations = iteration_count
        residual = matrix @ solution - rhs.ravel()
        self.last_residual_norm = float(np.linalg.norm(residual))
        if info != 0:
            if info > 0:
                message = f"{self.solver} did not converge (info={info})."
            else:
                message = f"{self.solver} failed with info={info}."
            raise LinearSolveError(message)
        return solution.reshape(shape)
