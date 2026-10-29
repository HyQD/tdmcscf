# Axial diatomic one-body core

`AxialDiatomic` and `AxialCoreHamiltonian` provide the clamped-nuclei
one-body part needed for a midpoint-centred spherical treatment of a linear
diatomic system. They do not implement molecular Hartree--Fock occupations or
an SCF solver.

## Geometry and Hamiltonian

The nuclei are placed at

```math
\mathbf R_1=(0,0,-a),\qquad \mathbf R_2=(0,0,a),
```

so the internuclear distance is `2*a`. The electronic core Hamiltonian is

```math
h=-\frac{1}{2\mu}\nabla^2
-\frac{Z_1}{|\mathbf r-\mathbf R_1|}
-\frac{Z_2}{|\mathbf r-\mathbf R_2|}.
```

The nuclear repulsion

```math
E_{\mathrm{nuc}}=\frac{Z_1Z_2}{2a}
```

is exposed separately as `core.nuclear_repulsion_energy` and is not included
in the electronic operator action.

Nuclei can be specified by chemical symbols through krypton or by positive
real charges. `charge` is the total system charge and is used to derive the
integer electron count. The object does not attach an orbital occupation to
that count.

## FEM-DVR and angular representation

The reduced radial convention is

```math
\phi(\mathbf r)=\frac{1}{r}\sum_{lm}u_{lm}(r)Y_{lm}(\Omega).
```

Arrays contain nodal values; no quadrature weights are absorbed. During this
initial implementation, `a` must coincide with an internal finite-element
boundary. This places the radial derivative discontinuity of every nuclear
multipole at an element boundary and also permits the Poisson construction to
represent its delta source on a grid node.

The axial nuclear potential contains only `M=0` multipoles, so it conserves
signed `m` but couples different `l` channels. The default nuclear cutoff is
`nuclear_L_max=2*l_max`, which is sufficient for its exact projection into an
orbital space truncated at `l_max`.

For a fixed signed magnetic number, arrays have shape

```text
(n_r, l_max - abs(m) + 1)
```

with radial-first flattening. The strong matrix is Hermitian in the FEM-DVR
metric

```math
W=\operatorname{diag}(w_a)\otimes I_l,
```

and therefore satisfies

```math
H^\dagger W=WH.
```

Use `fixed_m_matrix(m, representation="symmetric")` for a conventional
Hermitian eigensolver. This returns

```math
H_{\mathrm{sym}}=W^{1/2}HW^{-1/2}.
```

The corresponding eigenvectors can be mapped back to nodal values with
`from_symmetric_fixed_m`.

## Construction

```python
import numpy as np

from grid_lib.pseudospectral_grids import (
    FEMDVR,
    GaussLegendreLobatto,
    Linear_map,
)
from tdmcscf.grid.spherical import AxialCoreHamiltonian
from tdmcscf.systems import AxialDiatomic

system = AxialDiatomic(
    "H",
    "H",
    half_bond_length=1.0,
    charge=1,
    label="H2+",
)
grid = FEMDVR(
    np.asarray([0.0, 1.0, 3.0, 6.0]),
    np.asarray([9, 9, 9]),
    Linear_map,
    GaussLegendreLobatto,
    symmetrize=False,
)
core = AxialCoreHamiltonian(
    grid,
    l_max=6,
    m_max=0,
    system=system,
)
matrix = core.fixed_m_matrix(0, representation="symmetric")
```

Set `nuclear_method="poisson"` to use the discrete radial Coulomb response
instead of the default analytic point-charge coefficients. The analytic path
is preferred for production calculations; the Poisson path provides an
independent discrete check. `poisson_method` selects `"inverse"` or `"solve"`
when the Poisson response is constructed.

Both the existing central cores and `AxialCoreHamiltonian` implement the
structural `SphericalCoreOperator` protocol. Consequently, the low-level
`SphericalRestrictedTDHF` and `SphericalRestrictedTDKS` orbital equations can
accept either operator without a dependency on `grid-tdse`.
