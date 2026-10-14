# Time integrators

`tdmcscf.integrators` contains coordinate-independent time-stepping methods.
The integrators receive operator or right-hand-side callables; they do not
import spherical-grid or GTO implementations.

## Second-order Strang splitting

`Strang2` composes two supplied step methods as

$$
\mathcal U_A(\Delta t/2)\,
\mathcal U_B(\Delta t)\,
\mathcal U_A(\Delta t/2).
$$

This composition supports the variational splitting method used by Sato *et
al.* [Phys. Rev. A **94**, 023405 (2016)](https://doi.org/10.1103/PhysRevA.94.023405)
for TD-MCSCF propagation. In that application, $A$ is the stiff linear
one-body equation and $B$ contains the coupled nonlinear orbital and
CI-coefficient equations.

The split composition treats its state as an opaque object. Method-specific
sub-integrators may therefore propagate a NumPy orbital array or a compound
state containing orbitals and coefficient variables.

For explicitly time-dependent $A(t)$, the first and second half-steps cover
$[t,t+\Delta t/2]$ and $[t+\Delta t/2,t+\Delta t]$. A midpoint method
therefore evaluates $A$ at $t+\Delta t/4$ and $t+3\Delta t/4$.

```python
from tdmcscf.integrators import CrankNicolson, RK4, Strang2

integrator = Strang2(
    linear_integrator=CrankNicolson(
        solver="gmres",
        rtol=1e-11,
        preconditioner=preconditioner,
    ),
    nonlinear_integrator=RK4(),
)

new_state = integrator(
    state,
    time,
    time_step,
    linear_rhs,
    nonlinear_rhs,
)
```

Every right-hand side returns a time derivative. Thus, when an operator
action is written as

$$
i\dot y=H(t)y,
$$

the corresponding callable returns `-1j * H.apply(y, t)`.

## Crank--Nicolson

For a linear equation

$$
\dot y=L(t)y,
$$

`CrankNicolson` solves the implicit-midpoint equation

$$
\left[I-\frac{\Delta t}{2}L(t+\Delta t/2)\right]y_{n+1}
=
\left[I+\frac{\Delta t}{2}L(t+\Delta t/2)\right]y_n.
$$

The left-hand side is exposed to SciPy as a matrix-free `LinearOperator`.
GMRES is the default; BiCGSTAB is also available. A preconditioner can be a
SciPy `LinearOperator` or a callable accepting and returning the original
state shape. The latter form lets a coordinate representation apply its own
block solves without exposing those details to the integrator.

For spherical atomic calculations, an appropriate preconditioner
approximately applies

$$
\left[I+\frac{i\Delta t}{2}H_{\mathrm{core}}\right]^{-1}
$$

independently in each angular channel. Its implementation belongs to the
spherical propagation layer. The Krylov tolerance should make the linear
solve error smaller than the local splitting error.

After a step, `last_iterations` and `last_residual_norm` report diagnostics
from the Krylov solve.

## RK4

`RK4` implements the classical four-stage explicit Runge--Kutta method for
an array-valued right-hand side `rhs(state, time)`. It preserves the input
shape and does not modify the input array.
