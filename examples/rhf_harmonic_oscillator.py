"""Compute the RHF ground state of electrons in a harmonic trap."""

import matplotlib.pyplot as plt
import numpy as np

from tdmcscf.grid.spherical import SphericalHartreeFock
from tdmcscf.systems import HarmonicOscillator


# Physical system. The occupation {0: 1} gives one doubly occupied s orbital,
# corresponding to two interacting electrons.
omega = 0.5
mass = 1.0
occupation = {0: 1}
system = HarmonicOscillator(
    omega=omega,
    mass=mass,
    occupation=occupation,
)

# FEM-DVR element boundaries and Lobatto points in each element.
nodes = [0.0, 2.0, 5.0, 10.0, 20.0]
n_points = 15

# Only the monopole is needed for the occupied s orbital.
coulomb_l_max = 0

solver = SphericalHartreeFock(
    system,
    nodes=nodes,
    n_points=n_points,
    coulomb_l_max=coulomb_l_max,
    max_iterations=100,
    conv_tol_e=1e-12,
    conv_tol_grad=1e-10,
    damping=0.2,
    diis=True,
    diis_start=3,
    diis_size=8,
    poisson_method="solve",
)
result = solver.solve()
if not result.converged:
    raise RuntimeError("The harmonic-oscillator Hartree--Fock state did not converge.")

last_iteration = result.history[-1]
orbital_distributions = np.sum(np.abs(result.orbitals) ** 2, axis=2)
radial_density = 2 * np.sum(orbital_distributions, axis=0)
integrated_density = np.dot(solver.radial_weights, radial_density)
if not np.isclose(integrated_density, system.electron_count, rtol=0.0, atol=1e-10):
    raise RuntimeError("The radial density does not integrate to the electron count.")

title = "RHF results for two electrons in a harmonic trap"
print(f"\n{title}")
print("=" * len(title))
print(f"{'Converged':<24}: Yes")
print(f"{'Iterations':<24}: {result.iterations}")
print(f"{'Mass':<24}: {system.mass:.6f}")
print(f"{'Frequency':<24}: {system.omega:.6f} a.u.")
print(f"{'Element boundaries':<24}: {nodes} bohr")
print(f"{'Points per element':<24}: {n_points}")
print(f"{'ERHF':<24}: {result.energy:.15f} Eh")
print(f"{'Energy change':<24}: {last_iteration.energy_change:.3e} Eh")
print(f"{'Orbital-gradient norm':<24}: {last_iteration.gradient_norm:.3e}")
print(f"{'Density integral':<24}: {integrated_density:.12f}")
print(f"{'Expected electrons':<24}: {system.electron_count}")
print("\nOrbital energies:")
print("  n_r  l   m             energy (Eh)")
for (n_r, l, m), energy in zip(
    result.orbital_quantum_numbers, result.orbital_energies, strict=True
):
    print(f"  {n_r:3d}  {l:1d}  {m:2d}  {energy:22.15f}")
print()

r = solver.r
figure, (orbital_axes, density_axes) = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=(8, 8),
)

for radial_distribution, (n_r, l, m) in zip(
    orbital_distributions,
    result.orbital_quantum_numbers,
    strict=True,
):
    orbital_axes.plot(
        r,
        radial_distribution,
        label=rf"$n_r={n_r},\,l={l},\,m={m}$",
    )

density_axes.plot(r, radial_density, color="black", linewidth=2)
density_axes.fill_between(
    r,
    radial_density,
    color="C0",
    alpha=0.3,
    label=rf"$\int P(r)\,dr = {integrated_density:.1f}$",
)

orbital_axes.set_ylabel(r"$|u_{n_rl}(r)|^2$")
orbital_axes.set_title("Radial orbital distributions")
orbital_axes.legend()
density_axes.set_xlabel(r"$r$ (bohr)")
density_axes.set_ylabel(r"$P(r)$")
density_axes.set_title("Total radial electron density")
density_axes.legend()
figure.tight_layout()
plt.show()
