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

The same ground-state local or semilocal functional is evaluated at this
instantaneous density. The implementation is therefore adiabatic and contains no
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
    xc_quadrature_order=None,
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

For a semilocal functional, the propagation and internally generated ground
state use the same radial XC quadrature. By default each finite element uses
twice its number of FEM-DVR points. Set `xc_quadrature_order` to one integer
or one integer per element to control this overintegration explicitly.

## Hartree and XC action

The Hartree potential is constructed from diagonal occupied-orbital pair
densities using the existing spherical Coulomb multipoles. For each spatial
orbital, the factor of two accounts for closed-shell occupation.

The XC evaluator synthesizes the reduced orbital expansions on a
Gauss--Legendre--Fourier angular grid and forms the physical density including
the radial `1/r**2` factor. Local functionals are evaluated pointwise,
multiplied into the orbitals, and projected back to the retained
spherical-harmonic basis.

For PBE, the density and its gradient are instead evaluated on the angular
grid and a separate Gauss--Legendre radial quadrature. Angular orbital
derivatives are evaluated analytically from the spherical harmonics, which
preserves the spherical symmetry of complete subshells. A frozen density
defines the Hermitian weak operator

$$\langle\chi|v_{\mathrm{xc}}|\psi\rangle=\int f_n\chi^*\psi\,d^3r+\int 2f_\sigma\nabla n\mathbin{\cdot}\nabla(\chi^*\psi)\,d^3r.$$

The operator is applied matrix-free and then converted back to nodal reduced
radial values. This is the same variational discretization used by the PBE
ground-state solver, so a converged field-free ground state is stationary in
the orthogonal orbital gauge. The sampled strong-form Euler potential is
retained only as a diagnostic. `PBE()`, `PBEExchange()`, and
`PBECorrelation()` may be used wherever an LDA functional is accepted.

Nonlinear XC functionals are not finitely band limited. The
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
Hartree potential, and XC operator at each stage. The implicit constant-field
schemes freeze the density-dependent Hartree and XC operators over their
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
renormalized after masking. TDKS checkpoints record the functional metadata,
XC angular bandlimit, and resolved radial XC quadrature and reject
incompatible restarts.
