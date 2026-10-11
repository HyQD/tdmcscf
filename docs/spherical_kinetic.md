# Spherical kinetic energy

`SphericalKineticEnergy` applies $\hat T=-\nabla^2/(2\mu)$ to orbitals stored
in the same nodal representation as the spherical mean fields:

$$
\phi_p(r,\Omega)=\frac1r\sum_{lm}u^p_{lm}(r)Y_{lm}(\Omega).
$$

Its output represents the reduced radial coefficients of $\hat T\phi_p$:

$$
(Tu)^p_{a,lm}=-\frac{1}{2\mu}\sum_bD^{(2)}_{ab}u^p_{b,lm}
+\frac{l(l+1)}{2\mu r_a^2}u^p_{a,lm}.
$$

Both the radial derivative and centrifugal term belong to kinetic energy
and scale with the same inverse mass. The mass is a positive finite real
scalar, in electron-mass units, and defaults to one. It is fixed when the
operator is constructed; use separate instances for different species.

## Usage

```python
import numpy as np
from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto, Linear_map,
)
from tdmcscf.grid.spherical import SphericalKineticEnergy

radial = FEMDVR(
    np.array([0., 1., 3., 6.]), np.array([9, 10, 11]),
    Linear_map, GaussLegendreLobatto, symmetrize=False,
)
kinetic = SphericalKineticEnergy(radial, l_max=2, mass=2.5)
orbitals = np.zeros((1,) + kinetic.orbital_shape, dtype=complex)
i00 = kinetic.orbital_lm.index((0, 0))
orbitals[0, :, i00] = kinetic.r * (6 - kinetic.r)
Tu = kinetic.apply(orbitals)
np.testing.assert_allclose(Tu[0, :, i00], 1/2.5, atol=2e-11)
```

`apply` accepts `(n_r, n_lm)` for one orbital or
`(n_orbitals, n_r, n_lm)` for a batch and returns the same shape, with complex
output. It preserves its inputs. `orbital_lm` has the same M-major ordering
as grid-lib's `LM_to_I` and `SphericalMeanFieldBuilder`.

The operator is diagonal in angular channels, so general and fixed-m orbitals
use exactly the same application. No m metadata or angular grid is needed.
No Coulomb matrices or angular-transform tables are constructed.

## Radial grid and metric

Use a fixed real FEM-DVR grid starting at zero, constructed with
`symmetrize=False`. The first and last radial nodes are excluded from orbital
arrays, imposing $u(0)=u(r_{\max})=0$. The centrifugal factor is evaluated
only at positive interior radii. Supply the full grid to the constructor;
it selects `D2[1:-1, 1:-1]` itself.

Arrays contain nodal values, not normalized DVR coefficients. Grid-lib's
nodal `D2` already contains the inverse quadrature overlap matrix. Thus no
additional radial weights or overlap inverse enter `apply`. Weights enter
inner products and expectation values:

$$
\langle\phi_p|T|\phi_q\rangle\approx
\sum_{a,lm}w_a[u^p_{a,lm}]^*(Tu)^q_{a,lm}.
$$

The discrete Hermiticity condition is $T_l^\dagger S=ST_l$, with
$S=\operatorname{diag}(w_a)$. The nodal matrix need not be symmetric in the
ordinary Euclidean metric. `D2_symmetric` acts in a different coefficient
representation and is not used here.

This class applies kinetic energy only. Multiplicative one-body potentials,
orbital-space Q projection, and the factor $-i$ for real-time propagation
are separate operations. Position-dependent masses, complex radial contours,
and alternative endpoint conditions are outside this initial implementation.
