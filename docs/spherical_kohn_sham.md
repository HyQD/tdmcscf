# Spin-unpolarized spherical Kohn--Sham ground states

`SphericalKohnSham` solves the restricted Kohn--Sham equations for a
closed-shell electronic system in a real central potential.
`SphericalAtomicKohnSham` is the convenience interface for the closed-shell
atoms and ions in the built-in catalog.

The implementation uses the total spin-summed density. For complete occupied
subshells with reduced radial orbitals `u_nl(r)`,

$$n(r)=\frac{1}{2\pi r^2}\sum_{nl}(2l+1)|u_{nl}(r)|^2.$$

The corresponding radial electron density is

$$P(r)=4\pi r^2n(r)=2\sum_{nl}(2l+1)|u_{nl}(r)|^2,$$

and the FEM-DVR quadrature satisfies

$$\sum_a w_aP(r_a)=N.$$

The effective radial Hamiltonian in angular-momentum sector `l` is

$$H_l^{\mathrm{KS}}=T_l+V_{\mathrm{ext}}(r)+V_{\mathrm H}[n](r)+V_{\mathrm{xc}}[n](r).$$

Only the `L=0` radial Coulomb response is needed for the spherical Hartree
potential. The same potential is applied to every occupied `l` block.

## Local functionals

The coordinate-independent local functionals are available in `tdmcscf.dft`.
Every functional accepts the total density and returns its pointwise energy
density and multiplicative potential.

Dirac spin-unpolarized exchange is

$$E_x[n]=-\frac{3}{4}\left(\frac{3}{\pi}\right)^{1/3}\int n(\mathbf r)^{4/3}\,d^3r,$$

with

$$v_x(n)=-\left(\frac{3}{\pi}\right)^{1/3}n^{1/3}.$$

It is exposed as `DiracExchange()`. `VWN5Correlation()` provides the
spin-unpolarized VWN5 correlation energy and potential, while `LDAVWN()`
combines the two.

To avoid undefined intermediate expressions in the asymptotic zero-density
limit, VWN5 returns its limiting zero energy density and potential for
densities at or below `1e-30` by default. The numerical cutoff can be changed
with `VWN5Correlation(density_cutoff=...)`.

`XAlphaExchange(alpha)` uses the standard Slater convention

$$v_x^{X\alpha}(n)=-\frac{3\alpha}{2}\left(\frac{3}{\pi}\right)^{1/3}n^{1/3}.$$

Thus, `alpha=2/3` reproduces Dirac exchange. This value is the default, but
any finite nonnegative value can be selected. `XAlphaVWN(alpha)` combines the
chosen X-alpha exchange with VWN5 correlation.

```python
from tdmcscf.dft import LDAVWN, XAlphaExchange, XAlphaVWN

lda_vwn = LDAVWN()
xalpha = XAlphaExchange(alpha=0.7)
xalpha_vwn = XAlphaVWN(alpha=0.7)
```

Custom local functionals may be supplied if they implement
`evaluate(density)` and return finite real arrays named `energy_density` and
`potential`, each with the same shape as `density`.

## Atomic calculation

```python
from tdmcscf.dft import LDAVWN
from tdmcscf.grid.spherical import SphericalAtomicKohnSham

solver = SphericalAtomicKohnSham(
    "Ne",
    functional=LDAVWN(),
    nodes=[0.0, 0.3, 1.0, 3.0, 7.0, 15.0, 30.0],
    n_points=17,
    poisson_method="solve",
    max_iterations=100,
    conv_tol_e=1e-10,
    conv_tol_grad=1e-8,
    damping=0.2,
    diis=True,
)
result = solver.solve()
```

The initial guess is obtained by diagonalizing the noninteracting central
Hamiltonian in every occupied `l` block. The solver then constructs the
density, Hartree potential, and XC potential; applies damping or commutator
DIIS; and diagonalizes the updated Kohn--Sham blocks.

As in the Hartree--Fock solver, `conv_tol_grad` is the sole convergence
criterion when supplied. Set `conv_tol_grad=None` to converge using
`conv_tol_e` instead.

The result includes the expanded orbitals and unique radial orbitals together
with

- `density` for `n(r)`,
- `radial_density` for `P(r)`,
- `hartree_potential` and `xc_potential`, and
- `kinetic_energy`, `external_energy`, `hartree_energy`, and `xc_energy`.

The reported total energy is evaluated as

$$E=T_s+\int n(\mathbf r)V_{\mathrm{ext}}(\mathbf r)\,d^3r+\frac{1}{2}\int n(\mathbf r)V_{\mathrm H}(\mathbf r)\,d^3r+E_{\mathrm{xc}}[n].$$

## General central potentials

`SphericalKohnSham` accepts the same `SphericalElectronicSystem` instances as
the general Hartree--Fock solver:

```python
from tdmcscf.dft import XAlphaVWN
from tdmcscf.grid.spherical import SphericalKohnSham
from tdmcscf.systems import HarmonicOscillator

system = HarmonicOscillator(omega=0.5, occupation={0: 1})
result = SphericalKohnSham(
    system,
    functional=XAlphaVWN(alpha=0.7),
    nodes=[0.0, 1.0, 3.0, 7.0, 14.0],
    n_points=15,
).solve()
```

The present functional layer is deliberately limited to spin-unpolarized
local approximations. Gradient-dependent functionals require a separate,
discretely consistent density-gradient and divergence implementation.
