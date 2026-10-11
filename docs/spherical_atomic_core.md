# Central Coulomb potential and atomic core Hamiltonian

`SphericalCoulombPotential` represents the attraction to a point nucleus at the
origin. `AtomicCoreHamiltonian` combines it with `SphericalKineticEnergy`:

$$
V(r)=-\frac{Z}{r}, \qquad
H_{\mathrm{core}}=-\frac{1}{2\mu}\nabla^2-\frac{Z}{r}.
$$

All quantities are in atomic units. The required charge `Z` is positive, finite,
and real; noninteger effective charges are supported. The particle or reduced
mass `mass` defaults to 1 and scales only the kinetic energy.

## Nodal representation and boundaries

The [mean-field representation](spherical_mean_fields.md) is retained:

$$
\phi_p(r,\Omega)=\frac{1}{r}\sum_{lm}u^p_{lm}(r)Y_{lm}(\Omega).
$$

Arrays store the nodal values of the reduced radial functions, with shape
`(n_r, n_lm)` or `(n_orbitals, n_r, n_lm)`. Both operators preserve this shape,
return complex128 arrays, and leave inputs unchanged. Channels follow grid-lib's
M-major ordering, exposed by `orbital_lm`. Each channel acts independently, so
both general expansions and fixed-m orbitals are supported. `m_max` restricts
the retained channels to all `abs(m) <= m_max`; a fixed-m orbital is represented
by setting the other channels to zero.

Use a fixed real grid starting at zero, with nodal `D2` as produced by FEMDVR
with `symmetrize=False`. Both endpoints are excluded and the reduced radial
functions vanish there. The potential is evaluated only at interior radii;
there is no division by zero or softened nuclear potential. The standalone
potential needs only the grid's `r` and `weights` arrays.

For interior nodes, the potential action and radial core block are

$$
(Vu)_{a,lm}=-\frac{Z}{r_a}u_{a,lm},
$$

$$
(H_l)_{ab}=-\frac{(D_2)_{ab}}{2\mu}
+\delta_{ab}\left[\frac{l(l+1)}{2\mu r_a^2}-\frac{Z}{r_a}\right].
$$

Here `D2` is restricted to interior rows and columns. These are nodal action
matrices: the inverse quadrature overlap is already accounted for. Do not
multiply `apply()` or `radial_matrix(l)` by another inverse overlap or quadrature
weight. Quadrature weights remain necessary for orbital inner products.

## Construction and orbital action

```python
import numpy as np
from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from tdmcscf.grid.spherical import AtomicCoreHamiltonian, SphericalCoulombPotential

grid = FEMDVR(
    np.array([0., 1., 3., 7., 15., 30., 60., 100.]),
    np.full(7, 17),
    Linear_map,
    GaussLegendreLobatto,
    symmetrize=False,
)
core = AtomicCoreHamiltonian(grid, l_max=2, Z=1.0)  # mass defaults to 1
potential = SphericalCoulombPotential(grid, l_max=2, Z=1.0)

u = np.zeros(core.orbital_shape, dtype=complex)
s_channel = core.orbital_lm.index((0, 0))
u[:, s_channel] = core.r * np.exp(-core.r) * (1 - core.r / grid.r[-1])
Vu = potential.apply(u)
Hu = core.apply(u)
np.testing.assert_allclose(Hu, core.kinetic.apply(u) + core.potential.apply(u))
```

`core.kinetic` and `core.potential` are independently usable. The potential's
`values` array contains `-Z/r` at the interior nodes. Both the kinetic operator
and the core expose `radial_matrix(l)` for integers `0 <= l <= l_max`. Each call
returns an independent matrix, with no dependence on `m`; modifying this return
value does not modify the operator. Treat operator attributes as construction
data; construct a new operator to change its grid, charge, or mass.

## Atomic eigenvalue problems

The nodal matrix is Hermitian with respect to the quadrature metric,

$$
S_{ab}=w_a\delta_{ab}, \qquad H_l^\dagger S=S H_l.
$$

It need not be symmetric in the ordinary Euclidean metric. Transform it before
using `eigh`, then convert the eigenvectors back to nodal values:

```python
H_l = core.radial_matrix(l=0)
root_w = np.sqrt(core.radial_weights)
H_symmetric = root_w[:, None] * H_l / root_w[None, :]
energies, vectors = np.linalg.eigh(H_symmetric)
u_radial = vectors / root_w[:, None]

# The lowest s state as an orbital suitable for apply() and mean-field routines.
u_ground = np.zeros(core.orbital_shape, dtype=complex)
u_ground[:, s_channel] = u_radial[:, 0]
np.testing.assert_allclose(core.apply(u_ground), energies[0] * u_ground, atol=1e-10)
np.testing.assert_allclose(
    np.sum(core.radial_weights * abs(u_radial[:, 0])**2), 1.0
)
```

For an adequately resolved grid and a sufficiently distant outer boundary, the
bound-state energies approach

$$
E_n=-\frac{\mu Z^2}{2n^2}, \qquad n=l+1,l+2,\ldots.
$$

Finite-box and discretization errors remain; weakly bound states require larger
radial domains. This core contains kinetic energy and central nuclear attraction.
Mean fields and the orbital projector must be added by the orbital-equation
implementation. Nonspherical one-body potentials are outside this API's scope.
