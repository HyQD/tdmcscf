# Closed-shell spherical systems

The spherical restricted Hartree--Fock and TDHF operators support closed-shell
electrons in a real, time-independent central potential. A system contains the
potential, particle mass, and complete-shell occupation independently of its
FEM-DVR representation.

The Hamiltonian is

$$H=\sum_i\left[-\frac{1}{2\mu}\nabla_i^2+V(r_i)\right]+\sum_{i<j}\frac{1}{r_{ij}}.$$

Every occupied spatial orbital is doubly occupied. The available system types
are `Atomic`, `HarmonicOscillator`, `GaussianWell`, and
`CustomCentralPotential`.

## Closed-shell occupations

`ClosedShellOccupation` records the number of occupied radial orbitals in each
angular-momentum sector:

```python
from tdmcscf.systems import ClosedShellOccupation

occupation = ClosedShellOccupation({0: 2, 1: 1})
assert occupation.n_doubly_occupied == 5
assert occupation.electron_count == 10
```

The mapping `{0: 2, 1: 1}` contains two radial `s` orbitals and one radial `p`
orbital. Every radial orbital occurs for all `m=-l,...,l`, so the electron
count is $N=2\sum_l(2l+1)n_l$.

Generic systems label radial states by the zero-based radial index `n_r` and
return `(n_r, l, m)` orbital labels. Catalog atoms retain their spectroscopic
`(n, l, m)` labels.

## Built-in systems

Atoms and ions can be selected by catalog label, nuclear charge, or an
explicit atomic configuration:

```python
from tdmcscf.systems import Atomic

neon = Atomic("Ne")
same_neon = Atomic(10)
chloride = Atomic(17, charge=-1)
```

A numeric nuclear charge selects the neutral catalog entry unless `charge` is
given. A closed-shell atom outside the catalog requires an explicit
occupation:

```python
beryllium_ion = Atomic(4, charge=2, occupation={0: 1})
```

The harmonic-oscillator potential uses the same mass as the kinetic energy,

$$V(r)=\frac{1}{2}\mu\omega^2r^2,$$

and the Gaussian well is

$$V(r)=-V_0\exp(-ar^2),\qquad V_0>0,\quad a>0.$$

```python
from tdmcscf.systems import GaussianWell, HarmonicOscillator

harmonic = HarmonicOscillator(omega=0.5, mass=1.0, occupation={0: 1})
well = GaussianWell(V0=4.0, a=0.5, mass=1.0, occupation={0: 1})
```

A custom central potential is any callable accepting the positive interior
radial nodes and returning a finite real scalar or an array of the same shape:

```python
import numpy as np
from tdmcscf.systems import CustomCentralPotential

screened = CustomCentralPotential(
    lambda r: -2.0 * np.exp(-0.4 * r**2),
    occupation={0: 1},
    mass=1.0,
    label="screened well",
)
```

## Ground states

`SphericalHartreeFock` handles every spherical system with the same grid,
Coulomb-truncation, and SCF controls as the atomic solver:

```python
from tdmcscf.grid.spherical import SphericalHartreeFock

solver = SphericalHartreeFock(
    harmonic,
    nodes=[0.0, 2.0, 5.0, 10.0, 20.0],
    n_points=15,
    coulomb_l_max=0,
    poisson_method="solve",
)
result = solver.solve()
```

A complete runnable version is available in
[`examples/rhf_harmonic_oscillator.py`](../examples/rhf_harmonic_oscillator.py).

`SphericalAtomicHartreeFock` remains available with its existing interface and
returns an `AtomicHartreeFockResult` containing the original
`AtomicConfiguration`. It delegates the shared SCF work to
`SphericalHartreeFock`.
