"""Compute a closed-shell atomic Hartree--Fock ground state."""

import matplotlib.pyplot as plt
import numpy as np

from tdmcscf.grid.spherical import SphericalAtomicHartreeFock
from tdmcscf.systems import atomic_configurations, closed_shell_atom


available_atoms = ", ".join(atomic_configurations())
print()
print(f"Available closed-shell atoms and ions: {available_atoms}")
atom = input("Select an atom or ion: ").strip()
configuration = closed_shell_atom(atom)

# FEM-DVR element boundaries and number of Lobatto points in each element.
nodes = [0.0, 4.0, 8.0, 12.0, 16.0]
n_points = [33, 25, 25, 25]

# Include every Coulomb multipole between occupied orbitals.
coulomb_l_max = 2 * configuration.occupied_l_max


solver = SphericalAtomicHartreeFock(
    configuration,
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
    poisson_method="solve"
)
result = solver.solve()
last_iteration = result.history[-1]

orbital_distributions = np.sum(np.abs(result.orbitals) ** 2, axis=2)
radial_density = 2 * np.sum(orbital_distributions, axis=0)
integrated_density = np.dot(solver.radial_weights, radial_density)
if not np.isclose(
    integrated_density,
    result.configuration.electron_count,
    rtol=0.0,
    atol=1e-10,
):
    raise RuntimeError("The radial density does not integrate to the electron count.")

title = f"Results for {result.configuration.symbol}"
print(f"\n{title}")
print("=" * len(title))
print(f"{'Converged':<22}: {'Yes' if result.converged else 'No'}")
print(f"{'Iterations':<22}: {result.iterations}")
print(f"{'Element boundaries':<22}: {nodes} bohr")
print(f"{'Points per element':<22}: {n_points}")
print(f"{'ERHF':<22}: {result.energy:.15f} Eh")
print(f"{'Energy change':<22}: {last_iteration.energy_change:.3e} Eh")
print(f"{'Orbital-gradient norm':<22}: {last_iteration.gradient_norm:.3e}")
print(f"{'Density integral':<22}: {integrated_density:.12f}")
print(f"{'Expected electrons':<22}: {result.configuration.electron_count}")
print("\nOrbital energies:")
print("  n  l   m             energy (Eh)")
for (n, l, m), energy in zip(
    result.orbital_quantum_numbers, result.orbital_energies, strict=True
):
    print(f"  {n:1d}  {l:1d}  {m:2d}  {energy:22.15f}")
print()

r = solver.r
figure, (orbital_axes, density_axes) = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=(8, 8),
)

for radial_distribution, (n, l, m) in zip(
    orbital_distributions,
    result.orbital_quantum_numbers,
    strict=True,
):
    orbital_axes.plot(
        r,
        radial_distribution,
        label=rf"$n={n},\,l={l},\,m={m}$",
    )

density_axes.plot(
    r,
    radial_density,
    color="black",
    linewidth=2,
)
density_axes.fill_between(
    r,
    radial_density,
    color="C0",
    alpha=0.3,
    label=rf"$\int P(r)\,dr = {integrated_density:.1f}$",
)

orbital_axes.set_ylabel(r"$|u_{nl}(r)|^2$")
orbital_axes.set_title(
    f"Radial orbital distributions for {result.configuration.symbol}"
)
orbital_axes.legend()
density_axes.set_xlabel(r"$r$ (bohr)")
density_axes.set_ylabel(r"$P(r)$")
density_axes.set_title("Total radial electron density")
density_axes.legend()
figure.tight_layout()
plt.show()
