"""Explicit Runge--Kutta time integrators."""

from collections.abc import Callable

from ._validation import finite_real, matching_array, state_array


class RK4:
    r"""Classical four-stage, fourth-order Runge--Kutta integrator.

    The right-hand side has signature ``rhs(state, time)`` and returns the
    time derivative of an array with the same shape as ``state``. For an
    equation written as ``i dstate/dt = action(state, time)``, pass a wrapper
    returning ``-1j * action(state, time)``.
    """

    def __call__(self, state, time, step_size, rhs: Callable):
        """Advance one step without modifying *state*."""
        if not callable(rhs):
            raise TypeError("rhs must be callable.")
        values = state_array(state)
        shape = values.shape
        time = finite_real(time, "time")
        step_size = finite_real(step_size, "step_size")

        def evaluate(argument, evaluation_time):
            return matching_array(rhs(argument, evaluation_time), shape, "rhs")

        k1 = evaluate(values, time)
        k2 = evaluate(values + 0.5 * step_size * k1, time + 0.5 * step_size)
        k3 = evaluate(values + 0.5 * step_size * k2, time + 0.5 * step_size)
        k4 = evaluate(values + step_size * k3, time + step_size)
        return values + (step_size / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
