# Restricted TDHF in spherical coordinates

`SphericalRestrictedTDHF` supplies the closed-shell Fock action and the two
right-hand sides used by a split time integrator. For occupied spatial
orbitals, the Fock action is

$$
F\phi_p=h(t)\phi_p+\sum_j(2J_j-K_j)\phi_p.
$$

The one-body part $h(t)$ contains the atomic core and an optional electric-
dipole interaction. It is exposed through `linear_rhs` for implicit
Crank--Nicolson propagation. The Coulomb contribution is rebuilt from the
current orbitals whenever `nonlinear_rhs` is evaluated, including every RK4
stage.

## Orbital gauges

With `orbital_gauge="canonical"`, the combined right-hand side represents

$$
i\dot\Phi=F\Phi.
$$

With `orbital_gauge="orthogonal"`, it represents

$$
i\dot\Phi=QF\Phi,
\qquad Q=1-\Phi\Phi^\dagger.
$$

The linear split part is $h\Phi$. Consequently, the orthogonal-gauge
nonlinear action is

$$
G\Phi-\Phi(\Phi^\dagger F\Phi),
\qquad G\Phi=\sum_j(2J_j-K_j)\Phi.
$$

Adding it to the linear action gives exactly $QF\Phi$. The orthogonal gauge
is the default.

## Coulomb and orbital cutoffs

`SphericalMeanFieldBuilder` accepts independent orbital cutoffs `(l_max,
m_max)` and Coulomb cutoffs `(L_max, M_max)`. The defaults

$$
L_{\max}=2l_{\max},\qquad M_{\max}=2m_{\max}
$$

retain the complete pair-density product space. Smaller Coulomb values
deliberately project out higher multipoles.

## Implicit preconditioner

`ShiftedAtomicCorePreconditioner` LU-factorizes the radial blocks of

$$
I+\frac{i\tau}{2}H_{\mathrm{core}},
$$

where $\tau$ is the duration of one Crank--Nicolson substep. The factors are
reused for all occupied orbitals and Krylov iterations. For an outer
`Strang2` step of length $\Delta t$, use $\tau=\Delta t/2$.

The complete example in `examples/tdhf_atom.py` constructs a Hartree--Fock
ground state, applies a one-cycle linearly $z$-polarized sine-square pulse,
and records each spatial orbital's $z$ expectation value and norm.
