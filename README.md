# tdmcscf

Python package for time-dependent multi-configurational self-consistent field tools.

Spherical-coordinate Coulomb mean-field construction and orbital action are
available in `tdmcscf.grid.spherical`. See the
[representation conventions and examples](docs/spherical_mean_fields.md).

The independent [spherical kinetic-energy operator](docs/spherical_kinetic.md)
supports a configurable particle or reduced mass and the same nodal layout.
The [central nuclear potential and atomic core Hamiltonian](docs/spherical_atomic_core.md)
provide nuclear attraction, orbital action, and radial matrices for eigenvalue problems.

The [closed-shell spherical systems](docs/spherical_systems.md) support atomic,
harmonic-oscillator, Gaussian-well, and custom central potentials. The
[atomic Hartree--Fock interface](docs/spherical_hartree_fock.md) is retained as
a convenience wrapper around the general symmetry-adapted radial solver.

The [spin-unpolarized spherical Kohn--Sham solver](docs/spherical_kohn_sham.md)
supports configurable local functionals, including Dirac exchange, Slater
X-alpha exchange, VWN5 correlation, and the combined LDA-VWN5 approximation.

The [electric-dipole field operators](docs/spherical_dipole_fields.md) apply
length- and velocity-gauge interactions with arbitrary Cartesian polarization
and sums of temporal profiles from `electromagnetic_fields`.

The [high-level spherical TDHF interface](docs/spherical_tdhf_interface.md)
configures ground-state initialization, fields, Strang2, RK4, IM-CMF, or
IM-CDM propagation, sampling, radial masks, and restartable checkpoints.

## Requirements

Runtime requirements match `HyQD-grid-lib`: Python >=3.12,<3.15,
NumPy >=2.5.2,<2.6, SciPy >=1.18.0,<1.19, and Numba >=0.67.0,<0.68,
along with opt-einsum, SymPy, packaging, tqdm, and Matplotlib.
`HyQD-grid-lib>=1.1.0` and `HyQD-integrator-lib>=0.2.0` are runtime
dependencies. Temporal field profiles are provided by
`HyQD-electromagnetic-fields`.

## Installation

Install in editable mode:

```bash
python -m pip install -e .
```

The existing `scipy-1.18.0` conda environment on this machine satisfies the
runtime requirements. From this repository, use its installed build tools:

```bash
conda activate scipy-1.18.0
python -m pip install --no-build-isolation -e .
```

Install with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Development

Run tests:

```bash
python -m pytest
```
