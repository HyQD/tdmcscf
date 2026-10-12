# tdmcscf

Python package for time-dependent multi-configurational self-consistent field tools.

Spherical-coordinate Coulomb mean-field construction and orbital action are
available in `tdmcscf.grid.spherical`. See the
[representation conventions and examples](docs/spherical_mean_fields.md).

The independent [spherical kinetic-energy operator](docs/spherical_kinetic.md)
supports a configurable particle or reduced mass and the same nodal layout.
The [central nuclear potential and atomic core Hamiltonian](docs/spherical_atomic_core.md)
provide nuclear attraction, orbital action, and radial matrices for eigenvalue problems.

The [closed-shell atomic Hartree--Fock solver](docs/spherical_hartree_fock.md)
uses symmetry-adapted radial Fock blocks, configurable FEM-DVR grids and
Coulomb multipole truncation, damped SCF iterations, and optional DIIS.

## Requirements

Runtime requirements match `HyQD-grid-lib`: Python >=3.12,<3.15,
NumPy >=2.5.2,<2.6, SciPy >=1.18.0,<1.19, and Numba >=0.67.0,<0.68,
along with opt-einsum, SymPy, packaging, tqdm, and Matplotlib.
`HyQD-grid-lib>=1.1.0` is also a runtime dependency.

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
