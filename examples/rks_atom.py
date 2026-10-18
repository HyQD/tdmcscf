"""Compute a spin-unpolarized atomic Kohn--Sham ground state."""

import numpy as np

from tdmcscf.dft import LDAVWN
from tdmcscf.grid.spherical import SphericalAtomicKohnSham
from tdmcscf.systems import atomic_configurations, closed_shell_atom


available_atoms = ", ".join(atomic_configurations())
print()
print(f"Available closed-shell atoms and ions: {available_atoms}")
atom = input("Select an atom or ion: ").strip()
configuration = closed_shell_atom(atom)

# FEM-DVR element boundaries and Lobatto points in each element.
nodes = [0.0, 4.0, 8.0, 12.0, 16.0]
n_points = [33, 25, 25, 25]

# Replace this with XAlphaExchange(alpha=...) or XAlphaVWN(alpha=...) to
# select a configurable Slater X-alpha approximation.
functional = LDAVWN()

solver = SphericalAtomicKohnSham(
    configuration,
    functional=functional,
    nodes=nodes,
    n_points=n_points,
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
last_iteration = result.history[-1]
electron_count = np.dot(solver.radial_weights, result.radial_density)

title = f"Kohn-Sham results for {result.configuration.symbol}"
print(f"\n{title}")
print("=" * len(title))
print(f"{'Functional':<22}: {result.functional.name}")
print(f"{'Converged':<22}: {'Yes' if result.converged else 'No'}")
print(f"{'Iterations':<22}: {result.iterations}")
print(f"{'Element boundaries':<22}: {nodes} bohr")
print(f"{'Points per element':<22}: {n_points}")
print(f"{'EKS':<22}: {result.energy:.15f} Eh")
print(f"{'Kinetic energy':<22}: {result.kinetic_energy:.15f} Eh")
print(f"{'External energy':<22}: {result.external_energy:.15f} Eh")
print(f"{'Hartree energy':<22}: {result.hartree_energy:.15f} Eh")
print(f"{'XC energy':<22}: {result.xc_energy:.15f} Eh")
print(f"{'Energy change':<22}: {last_iteration.energy_change:.3e} Eh")
print(f"{'Orbital-gradient norm':<22}: {last_iteration.gradient_norm:.3e}")
print(f"{'Density integral':<22}: {electron_count:.12f}")

print("\nOrbital energies:")
print("  n  l   m             energy (Eh)")
for (n, l, m), energy in zip(
    result.orbital_quantum_numbers, result.orbital_energies, strict=True
):
    print(f"  {n:1d}  {l:1d}  {m:2d}  {energy:22.15f}")
