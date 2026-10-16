# Time integrators

TDMC-SCF uses the coordinate-independent time-stepping methods provided by
`HyQD-integrator-lib`. The integrators receive operator or right-hand-side
callables; they do not import spherical-grid or GTO implementations.

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
from integrator_lib import CrankNicolson, RK4, Strang2

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

## Constant-mean-field implicit midpoint

`IMCMF1` and `IMCMF2` use a frozen-operator callback supplied by the physical
model. Calling `freeze(reference_state)` constructs the state-dependent data
once and returns a right-hand side that is linear in its trial state while
retaining explicit time dependence.

`IMCMF1` freezes at the initial state and advances one implicit-midpoint step.
`IMCMF2` takes one predictor half-step, freezes at the predicted midpoint
state, and restarts from the initial state. Its corrector uses two half-steps
by default, or one whole step when `corrector_steps=1`.

TDMC-SCF supplies this protocol through
`SphericalRestrictedTDHF.freeze_rhs`. Construction and application of the
spherical Coulomb fields remain in the coordinate-specific package.

## Constant-density implicit midpoint

The IM-CDM methods use the same coordinate-independent frozen-operator
drivers as IM-CMF but give the callback a different physical meaning.
`SphericalRestrictedTDHF.freeze_density_rhs` fixes the TDHF density entering
the direct and exchange operators. The resulting Fock action remains linear
in every Krylov trial state.

`im-cdm1` freezes the initial density and takes one full implicit-midpoint
step. `im-cdm2` first takes one predictor half-step with the initial density,
then freezes the predicted midpoint density and restarts from the initial
state. Its corrector can use one whole step or two half-steps. This is the
CDM2 sequence described in Appendix B of Aurbakken *et al.*, J. Chem. Phys.
**165**, 084307 (2026).

Both orbital gauges are available. In the orthogonal gauge, the
occupied-space Fock matrix is evaluated from the same fixed reference
orbitals, while explicit time dependence in the one-body field is retained.

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

For spherical central-potential calculations, an appropriate preconditioner
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
