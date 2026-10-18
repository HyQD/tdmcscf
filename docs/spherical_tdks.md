# Spherical time-dependent Kohn--Sham propagation

`SphericalTDKS` configures and propagates spin-restricted time-dependent
Kohn--Sham orbitals in spherical coordinates. It uses the same FEM-DVR grid,
electric-dipole fields, orbital gauges, integrators, masks, samplers, and
checkpoint workflow as `SphericalTDHF`.

The implemented adiabatic orbital equation is

$$i\frac{\partial}{\partial t}\phi_p(\mathbf r,t)=\left[h(t)+v_{\mathrm H}[n](\mathbf r,t)+v_{\mathrm{xc}}[n](\mathbf r,t)\right]\phi_p(\mathbf r,t).$$

The total spin-unpolarized density is reconstructed from the occupied spatial
orbitals at every varying-mean-field evaluation:

$$n(\mathbf r,t)=2\sum_{p=1}^{N/2}|\phi_p(\mathbf r,t)|^2.$$

The same ground-state local functional is evaluated at this instantaneous
density. The implementation is therefore adiabatic and contains no
exchange-correlation memory.

## Initial state

With `initial_state="groundstate"`, `SphericalTDKS` invokes
`SphericalKohnSham` on the propagation grid using the selected functional.
Ground-state SCF options are passed separately:

```python
from tdmcscf.dft import LDAVWN
from tdmcscf.grid.spherical import SphericalTDKS
from tdmcscf.systems import Atomic

calculation = SphericalTDKS(
    Atomic("Ne"),
    functional=LDAVWN(),
    nodes=[0.0, 4.0, 8.0, 12.0, 16.0],
    n_points=21,
    l_max=4,
    m_max=4,
    L_max=8,
    M_max=8,
    initial_state="groundstate",
    ground_state_options={
        "max_iterations": 100,
        "conv_tol_grad": 1e-10,
    },
    time_step=0.05,
    final_time=10.0,
)
```

A compatible `KohnShamResult` may be supplied directly. Its occupied
components are embedded in a larger propagation angular basis when needed.
A numeric array of orthonormal occupied orbitals is also accepted.

## Hartree and local XC action

The Hartree potential is constructed from diagonal occupied-orbital pair
densities using the existing spherical Coulomb multipoles. For each spatial
orbital, the factor of two accounts for closed-shell occupation.

The local XC evaluator synthesizes the reduced orbital expansions on a
Gauss--Legendre--Fourier angular grid, forms the physical density including
the radial `1/r**2` factor, evaluates the local functional pointwise,
multiplies the orbitals by the resulting potential, and projects back to the
retained spherical-harmonic basis.

Local nonlinear functionals are not finitely band limited. The
`xc_angular_bandlimit` parameter therefore controls angular oversampling, not
a physical truncation of the XC potential. It defaults to `4*l_max` and must
be at least `2*l_max`. Production calculations should verify convergence with
respect to this value.

## Orbital gauges and integrators

Both occupied-orbital gauges are available:

- `orbital_gauge="canonical"` propagates the complete local Kohn--Sham
  Hamiltonian.
- `orbital_gauge="orthogonal"` applies its projection outside the occupied
  orbital space.

The supported integrators match `SphericalTDHF`: Strang2, RK4, IM-CMF1,
IM-CMF2, IM-CDM1, and IM-CDM2. RK4 rebuilds the instantaneous density,
Hartree potential, and XC potential at each stage. The implicit constant-field
schemes freeze the density-dependent Hartree and XC potentials over their
linear solves.

For a local Kohn--Sham Hamiltonian, freezing the density uniquely fixes both
potentials. Consequently, IM-CMF and IM-CDM use the same frozen TDKS action;
both names remain available for interface consistency with TDHF.

Strang2 treats the central and electric-dipole one-body operators as its
linear part and the Hartree-XC contribution as its nonlinear part.

## Fields, masks, and observables

Length- and velocity-gauge electric-dipole fields use the existing
`SphericalDipoleField` implementation. Arbitrary Cartesian polarizations and
sums of field profiles are supported. Purely `z`-polarized fields preserve
the definite magnetic quantum number of each initial orbital and enable the
fixed-`m` Hartree path.

The default sampled quantities are individual orbital norms and Cartesian
position expectations. The closed-shell total dipole is obtained by summing
the orbital position contributions with occupation two and the configured
particle charge. Custom samplers use the same callable interface as TDHF.

`RadialCosineMask` may be applied after each completed step. Orbitals are not
renormalized after masking. TDKS checkpoints record the functional metadata
and XC angular bandlimit and reject incompatible restarts.
