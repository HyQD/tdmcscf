# Closed-shell atomic Hartree--Fock

`SphericalAtomicHartreeFock` solves the restricted Hartree--Fock equations for
closed-shell atoms and ions by diagonalizing symmetry-adapted radial Fock
matrices. Every occupied subshell contains all magnetic substates and shares
one radial function over (m=-l,\ldots,l).

This interface is retained as an atomic convenience wrapper. See
[closed-shell spherical systems](spherical_systems.md) for harmonic,
Gaussian-well, and custom central potentials handled by
`SphericalHartreeFock`.

The resulting one-particle density is rotationally invariant. The Fock
operator therefore has the form

$$
F_{lm,l'm'}(r,r')
=\delta_{ll'}\delta_{mm'}F_l(r,r').
$$

The Hartree term is a local radial potential. Exchange is a nonlocal radial
matrix, but the complete-shell sum is rotationally invariant. The solver
constructs and diagonalizes one (F_l) for each occupied (l); it does not
repeat the diagonalization for degenerate magnetic substates.

## Atomic configurations

Built-in configurations are available for H-, He, Be, F-, Ne, Mg, Cl-, Ar,
Ti4+, Ca, Zn, and Kr. ASCII labels are used in Python. Each configuration
stores the nuclear charge, ionic charge, and complete occupied subshells.
Numerical basis sizes are deliberately separate from the physical
configuration.

```python
from tdmcscf.systems import closed_shell_atom

neon = closed_shell_atom("Ne")
same_neon = closed_shell_atom(10)
assert same_neon is neon
assert neon.nuclear_charge == 10
assert neon.electron_count == 10
assert neon.n_doubly_occupied == 5
assert neon.orbital_quantum_numbers == (
    (1, 0, 0),
    (2, 0, 0),
    (2, 1, -1),
    (2, 1, 0),
    (2, 1, 1),
)
```

For an ion selected by nuclear charge, also provide its integer charge:

```python
chloride = closed_shell_atom(17, charge=-1)
```

An explicit closed-shell configuration can be supplied without modifying the
catalog:

```python
from tdmcscf.systems import AtomicConfiguration, ClosedSubshell

li_plus = AtomicConfiguration(
    symbol="Li+",
    nuclear_charge=3,
    charge=1,
    subshells=(ClosedSubshell(1, 0),),
)
```

The configuration validates

$$
N=Z-Q=\sum_{nl}2(2l+1).
$$

It describes a complete-shell determinant. Frozen-core and active-space
partitions belong to the later MCSCF method configuration rather than the
atomic catalog.

## FEM-DVR controls

Users can provide explicit element boundaries and either one point count for
all elements or one count per element:

```python
from tdmcscf.grid.spherical import SphericalAtomicHartreeFock

solver = SphericalAtomicHartreeFock(
    "Ne",
    nodes=[0.0, 0.3, 1.0, 3.0, 7.0, 15.0, 30.0],
    n_points=[17, 17, 17, 17, 17, 17],
)
result = solver.solve()
```

The first boundary must be zero. The first and last FEM-DVR nodes are removed,
so reduced radial orbitals vanish at the origin and outer boundary. Supplying
`r_max` together with explicit `nodes` checks that it equals the final
boundary.

Uniform elements can instead be generated from `r_max` and `n_elements`:

```python
solver = SphericalAtomicHartreeFock(
    "He",
    r_max=30.0,
    n_elements=10,
    n_points=15,
)
```

The grid is constructed with `symmetrize=False`. Orbitals contain nodal
reduced-radial values, and radial inner products retain the FEM-DVR weights.

## Angular and Coulomb truncation

`l_max` and `m_max` control the shape and channel ordering of the returned
orbital arrays. They default to the largest occupied angular momentum but may
be increased so the ground state can be used directly in a subsequent
time-dependent calculation:

```python
solver = SphericalAtomicHartreeFock(
    "Ne",
    r_max=30.0,
    n_elements=10,
    n_points=15,
    l_max=6,
    m_max=2,
    coulomb_l_max=2,
)
result = solver.solve()
assert result.orbitals.shape == (5, *solver.orbital_shape)
```

The occupied orbitals are nonzero only in their definite ((l,m)) channels.
Both `l_max` and `m_max` must include every occupied magnetic substate. They do
not determine which orbitals are occupied.

`coulomb_l_max` independently sets the largest multipole (L) retained in the
Coulomb expansion. Its default is twice the largest occupied (l), which
retains every direct and exchange multipole needed by the occupied atomic Fock
blocks. A smaller explicit value deliberately defines a truncated interaction.
The `poisson_method` argument selects whether grid-lib constructs the radial
Coulomb response with an explicit matrix inverse (`"inverse"`, the default)
or a direct multi-right-hand-side linear solve (`"solve"`).

## SCF iteration and result

The initial guess comes from diagonalizing the atomic core Hamiltonian in each
occupied (l) block. The solver then repeats:

1. Construct the radial Hartree and nonlocal exchange contributions.
2. Form the weighted-Hermitian radial Fock matrices.
3. Apply startup damping or commutator DIIS.
4. Diagonalize each occupied (l) block and select the configured radial
   subshells.
5. Test the orbital-gradient norm, represented by the weighted Fock-density
   commutator norm, when `conv_tol_grad` is supplied. If `conv_tol_grad=None`,
   test the total-energy change against `conv_tol_e` instead. Both diagnostics
   are recorded at every iteration.

The restricted energy is

$$
E_{\mathrm{RHF}}
=2\sum_i h_{ii}
+\sum_{ij}\left(2J_{ij}-K_{ij}\right)
=\sum_i\left(h_{ii}+F_{ii}\right),
$$

where (i,j) run over occupied spatial orbitals, including their magnetic
substates.

```python
solver = SphericalAtomicHartreeFock(
    "Ne",
    nodes=[0.0, 0.3, 1.0, 3.0, 7.0, 15.0, 30.0],
    n_points=17,
    poisson_method="solve",
    max_iterations=100,
    conv_tol_e=1e-10,
    conv_tol_grad=1e-8,
    damping=0.2,
    diis=True,
    diis_start=3,
    diis_size=8,
)
result = solver.solve()
assert result.converged
print(result.energy)
print(result.orbital_energies)
```

`result.orbitals` has shape `(n_doubly_occupied, n_r, n_lm)` and is ready for
the spherical operator and mean-field APIs. `result.radial_orbitals` stores the
unique radial functions by (l). `result.history` records the energy, absolute
energy change, and orbital-gradient norm for every iteration. A calculation
that reaches `max_iterations` returns its final state with `converged=False`.
