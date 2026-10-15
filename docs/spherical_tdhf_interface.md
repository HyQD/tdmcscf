# High-level spherical TDHF calculations

`SphericalTDHF` configures the radial grid, angular spaces, initial state,
electric-dipole fields, propagation method, sampling, masks, and checkpoints
for a closed-shell `SphericalElectronicSystem`.

## Basic calculation

The temporal field is supplied by `electromagnetic_fields`. Its gauge must
match `field_gauge`; this ensures that a length-gauge calculation receives
the electric field and a velocity-gauge calculation receives the consistent
vector potential.

```python
import numpy as np
from electromagnetic_fields.dipole_fields import SineSquareLaser

from tdmcscf.grid.spherical import SphericalTDHF
from tdmcscf.systems import Atomic

omega = 0.057
pulse = SineSquareLaser(
    E0=1e-4,
    omega=omega,
    ncycles=1,
    phase=0.0,
    gauge="velocity",
)

calculation = SphericalTDHF(
    Atomic("Ne"),
    nodes=[0.0, 4.0, 8.0, 12.0, 16.0],
    n_points=[21, 21, 21, 21],
    l_max=4,
    m_max=4,
    L_max=8,
    M_max=8,
    poisson_method="solve",
    initial_state="groundstate",
    ground_state_options={
        "conv_tol_grad": 1e-10,
    },
    fields=[(pulse, (0.0, 0.0, 1.0))],
    field_gauge="velocity",
    orbital_gauge="orthogonal",
    integrator="strang2",
    time_step=0.01,
    final_time=2 * np.pi / omega,
)
result = calculation.run(progress=True)
```

The step size is adjusted uniformly to `duration/ceil(duration/time_step)`,
so the time grid ends exactly at `final_time`. The actual step is available
as `calculation.time_step` and `result.time_step`.

## Grid and angular spaces

The FEM-DVR grid uses the same `nodes`, `n_points`, `r_max`, and `n_elements`
controls as `SphericalHartreeFock`. Orbital harmonics are truncated by
`(l_max, m_max)`, and Coulomb multipoles are truncated independently by
`(L_max, M_max)`.

The defaults are `m_max=l_max`, `L_max=2*l_max`, and
`M_max=min(L_max, 2*m_max)`. Thus, a full orbital space defaults to
`M_max=L_max`.

For field-free and purely z-polarized calculations, the interface detects
that magnetic quantum numbers are conserved. If every initial orbital has a
definite `m`, the polar-only mean-field path is selected automatically. Any
x or y component selects the general angular path.

## Initial states

`initial_state="groundstate"` runs `SphericalHartreeFock` on the propagation
grid. SCF controls belong in `ground_state_options`. The ground-state
orbitals are returned directly in the requested propagation angular space,
so unoccupied higher-l channels start at zero.

An existing `HartreeFockResult` is also accepted and is embedded into a
larger compatible angular space. Alternatively, pass an array with shape

```text
(number of occupied spatial orbitals, number of interior radial nodes,
 number of retained angular channels)
```

Custom initial orbitals must be orthonormal in the radial quadrature metric.

## Fields

`fields` accepts any mixture of:

- A temporal callable, interpreted as z-polarized
- A `DipoleFieldTerm`
- A `(temporal_callable, polarization)` pair

Every polarization vector is normalized. Field strength therefore belongs
to the temporal profile; for example, `(2, 0, 1)` specifies only a direction.
Multiple terms are summed to form the total Cartesian field. All temporal
objects that expose a `gauge` attribute must use the selected `field_gauge`.

The particle charge defaults to `-1`, and the mass is taken from the system.

## Propagation methods

`integrator="strang2"` uses the second-order variational splitting employed
by Sato *et al.* The linear one-body half-steps use preconditioned
Crank--Nicolson, and RK4 advances the nonlinear part. Options for the SciPy
linear solver can be supplied through `integrator_options`, for example:

```python
integrator_options={"solver": "gmres", "rtol": 1e-11, "atol": 0.0}
```

`integrator="cn-cmf1"` freezes the mean-field operator at the beginning of
each outer interval and performs one preconditioned Crank--Nicolson step.
This is first order with respect to the state-dependent mean field.

`integrator="cn-cmf2"` predicts the midpoint with the initial mean field,
freezes the operator at that predicted state, and restarts from the initial
orbitals for two Crank--Nicolson corrector half-steps. The predictor uses two
quarter-steps by default. One predictor half-step can be selected with:

```python
integrator_options={
    "solver": "gmres",
    "rtol": 1e-11,
    "atol": 0.0,
    "predictor_steps": 1,
}
```

The CN-CMF methods support both orbital gauges. The orthogonal gauge remains
the default and is the established choice when mean fields are held constant
over an integration interval. Explicit time dependence from the applied field
is still evaluated at every Crank--Nicolson substep midpoint.

`integrator="rk4"` applies ordinary RK4 to the complete orbital equation and
rebuilds the mean fields at every stage. Constructing an RK4 calculation
issues a warning containing a conservative kinetic-spectrum estimate. The
estimate is based on the imaginary-axis stability interval,

```math
\Delta t_{\mathrm{estimate}}
= s\frac{2\sqrt{2}}{\max |\lambda(T)|},
```

with safety factor `s=0.5`. Fields and mean fields can require a smaller
step. The estimate is also available through `estimate_rk4_time_step`.

## Radial masks

`RadialCosineMask(r0, n=4)` supplies the mask already used by `grid-lib`:

```math
M(r)=\cos^{1/n}\left[\frac{\pi(r-r_0)}{2(r_{\max}-r_0)}\right]
```

between `r0` and the outer boundary. It equals one below `r0` and zero at the
boundary. The mask is applied after every completed propagation step.
Orbitals are deliberately not renormalized or orthogonalized, so their norms
record loss through the boundary.

A callable evaluated on the interior radial nodes or an explicit array of
values in `[0, 1]` can be supplied instead.

## Sampling

By default, every step records:

- `orbital_norms`, with shape `(n_samples, n_orbitals)`
- `position_expectations`, with shape `(n_samples, n_orbitals, 3)`

The closed-shell electronic dipole is available as `result.total_dipole`.
Set `observables=()` to disable the built-ins or select either name
explicitly. `sample_interval` controls the number of steps between samples;
the initial and final states are always sampled.

Additional quantities can be registered with a mapping. Each callable
receives `(calculation, orbitals, time)` and must return a fixed-shape finite
numeric value:

```python
samplers={
    "maximum_amplitude": lambda calculation, orbitals, time: np.max(
        np.abs(orbitals)
    )
}
```

All sampled arrays are available in `result.samples`.

## Checkpoints

Set `checkpoint_path` and `checkpoint_interval` to write compressed NumPy
checkpoints during propagation. A final checkpoint is always written when a
path is configured. Each checkpoint contains:

- Current time, step, and orbitals
- FEM-DVR nodes, points per element, radial nodes, and weights
- Orbital and Coulomb angular cutoffs
- System, field, gauge, integrator, and time-grid metadata
- All samples accumulated so far

Load or resume a compatible calculation with

```python
from tdmcscf.grid.spherical import TDHFCheckpoint

checkpoint = TDHFCheckpoint.load("checkpoint.npz")
result = calculation.run(checkpoint=checkpoint)
```

Temporal callables and custom potentials are not serialized. Recreate the
same `SphericalTDHF` calculation before resuming; the stored grid, angular
spaces, time step, integrator, and sampler names are checked.
