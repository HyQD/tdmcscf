# Spherical axial diatomic Hartree--Fock

`SphericalDiatomicHartreeFock` solves the restricted closed-shell
Hartree--Fock equations for two clamped nuclei on the spherical-coordinate
`z` axis. It uses `AxialCoreHamiltonian` for the one-body operator and
`SphericalMeanFieldBuilder` for direct and exchange fields.

## Occupation

`AxialClosedShellOccupation` specifies the number of occupied spatial
orbitals in each absolute magnetic sector. For example,

```python
occupation = AxialClosedShellOccupation({0: 2, 1: 1})
```

contains two doubly occupied sigma orbitals and one doubly occupied orbital
in each of the signed `m=-1` and `m=+1` sectors. The electron count is

```math
N_e=2\left[n_0+2\sum_{m>0}n_m\right].
```

The solver validates this count against the charge and nuclear charges stored
by `AxialDiatomic`. The requested number of orbitals is preserved separately
in every signed-`m` sector throughout the SCF calculation; orbitals are not
refilled globally according to instantaneous eigenvalues.

## Fock operator and SCF equations

For doubly occupied spatial orbitals, the fixed-density Fock action is

```math
F\psi=h\psi+\sum_j\left(2J_j-K_j\right)\psi.
```

`FixedDensityAxialFock` applies this operator without assembling a dense
exchange matrix. The direct potentials are built once for a fixed density.
Each exchange application constructs only the cross potentials between the
fixed occupied orbitals and the trial vector.

The occupied orbitals are represented in the Euclidean-Hermitian FEM-DVR
representation during SCF. The orbital residual in a signed-`m` sector is

```math
R=FC-C\left(C^\dagger FC\right),
```

where the columns of the mathematical matrix `C` are occupied orbitals. (The
array implementation stores the orbital index first.) This is the weighted
physical orbital gradient after the FEM-DVR metric transformation.

The implementation uses:

- matrix-free `scipy.sparse.linalg.eigsh` diagonalization in every occupied
  signed-`m` block;
- damping before DIIS starts;
- residual-based DIIS as a linear combination of fixed-density Fock
  operators;
- occupied-space canonicalization after convergence;
- the gradient tolerance as the sole convergence criterion when
  `conv_tol_grad` is not `None`.

The electronic energy is

```math
E_{\mathrm{el}}=\sum_i\left(\langle\phi_i|h|\phi_i\rangle
+\langle\phi_i|F|\phi_i\rangle\right),
```

and the reported total energy includes the clamped nuclear repulsion:

```math
E_{\mathrm{RHF}}=E_{\mathrm{el}}+\frac{Z_1Z_2}{2a}.
```

## Example

```python
from tdmcscf.grid.spherical import SphericalDiatomicHartreeFock
from tdmcscf.systems import AxialDiatomic

system = AxialDiatomic(
    "H",
    "H",
    internuclear_distance=1.4,
    label="H2",
)
solver = SphericalDiatomicHartreeFock(
    system,
    {0: 1},
    nodes=[0.0, 0.7, 1.5, 3.0, 6.0, 12.0, 24.0],
    n_points=9,
    l_max=6,
    m_max=0,
    conv_tol_grad=1.0e-8,
)
result = solver.solve()
print(result.energy)
```

The common nuclear radius must currently be an internal finite-element
boundary. The nuclear multipole cutoff defaults to `2*l_max`. The
electron--electron `coulomb_L_max` and `coulomb_M_max` remain independent
controls.

## Published Hartree--Fock-limit comparison

Tasi and Császár report Hartree--Fock-limit energies of
`-1.13362958(1) Eh` for H2 at `R=1.4` bohr and `-2.933103279(2) Eh` for HHe+
at `R=1.455` bohr. The reference is
[Chemical Physics Letters 438, 139--143
(2007)](https://doi.org/10.1016/j.cplett.2007.02.056).

Using the same radial grid in each row,

```text
nodes = [0, a, 1.5, 3, 6, 12, 24]
n_points = 9 per element
nuclear_L_max = coulomb_L_max = 2*l_max
```

gives the following initial angular-convergence sequence:

| System | Bond length | `l_max` | Total energy (Eh) |
|---|---:|---:|---:|
| H2 | 1.400 | 2 | -1.116997960520 |
| H2 | 1.400 | 4 | -1.128634921271 |
| H2 | 1.400 | 6 | -1.131549753172 |
| HHe+ | 1.455 | 2 | -2.695887177055 |
| HHe+ | 1.455 | 4 | -2.849213843669 |
| HHe+ | 1.455 | 6 | -2.896422661388 |

The energies decrease systematically toward the independent limits. The slow
angular convergence is expected: a one-centre partial-wave expansion must
represent nuclear cusps displaced from the expansion origin. Radial and
angular convergence should therefore be assessed separately, and substantially
higher `l_max` is required for benchmark precision.
