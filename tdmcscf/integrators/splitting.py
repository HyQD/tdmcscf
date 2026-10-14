"""Composition methods for split equations of motion."""

from collections.abc import Callable

from ._validation import finite_real


class Strang2:
    r"""Second-order symmetric ``A/2 - B - A/2`` composition.

    Parameters
    ----------
    linear_integrator, nonlinear_integrator : callable
        Step callables with signature ``step(state, time, step_size, rhs)``.
        ``state`` is deliberately opaque to this class, allowing a method-
        specific stepper to propagate arrays or coupled orbital/coefficient
        states.

    Notes
    -----
    For an outer interval ``[t, t + dt]``, the first A step covers
    ``[t, t + dt/2]``, the B step covers ``[t, t + dt]``, and the second A
    step covers ``[t + dt/2, t + dt]``. A midpoint rule used for the A steps
    consequently evaluates a time-dependent A operator at ``t + dt/4`` and
    ``t + 3*dt/4``.
    """

    def __init__(self, linear_integrator, nonlinear_integrator):
        if not callable(linear_integrator):
            raise TypeError("linear_integrator must be callable.")
        if not callable(nonlinear_integrator):
            raise TypeError("nonlinear_integrator must be callable.")
        self.linear_integrator = linear_integrator
        self.nonlinear_integrator = nonlinear_integrator

    def __call__(
        self,
        state,
        time,
        step_size,
        linear_rhs: Callable,
        nonlinear_rhs: Callable,
    ):
        """Advance one symmetric split step."""
        if not callable(linear_rhs):
            raise TypeError("linear_rhs must be callable.")
        if not callable(nonlinear_rhs):
            raise TypeError("nonlinear_rhs must be callable.")
        time = finite_real(time, "time")
        step_size = finite_real(step_size, "step_size")
        half_step = 0.5 * step_size
        state = self.linear_integrator(state, time, half_step, linear_rhs)
        state = self.nonlinear_integrator(
            state, time, step_size, nonlinear_rhs
        )
        return self.linear_integrator(
            state, time + half_step, half_step, linear_rhs
        )
