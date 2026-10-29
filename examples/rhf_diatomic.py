"""Restricted Hartree--Fock ground state of H2 in spherical coordinates."""

from tdmcscf.grid.spherical import SphericalDiatomicHartreeFock
from tdmcscf.systems import AxialDiatomic


# Geometry and closed-shell occupation.
system = AxialDiatomic(
    "H",
    "H",
    internuclear_distance=1.4,
    label="H2",
)
occupation = {0: 1}

# The nuclear radius R/2 must be an internal FEM element boundary.
nodes = [0.0, 0.7, 1.5, 3.0, 6.0, 12.0, 24.0]
n_points = 9

# Increase l_max systematically toward the molecular Hartree--Fock limit.
l_max = 4
m_max = 0

solver = SphericalDiatomicHartreeFock(
    system,
    occupation,
    nodes=nodes,
    n_points=n_points,
    l_max=l_max,
    m_max=m_max,
    nuclear_L_max=2 * l_max,
    coulomb_L_max=2 * l_max,
    coulomb_M_max=0,
    max_iterations=50,
    conv_tol_e=1.0e-10,
    conv_tol_grad=1.0e-8,
    damping=0.2,
    diis=True,
    diis_start=3,
    diis_size=8,
    eigensolver_tolerance=1.0e-10,
)
result = solver.solve()

print(f"System: {result.system.label}")
print(f"Converged: {result.converged} in {result.iterations} iterations")
print(f"Electronic energy: {result.electronic_energy:.15f} Eh")
print(f"Nuclear repulsion: {result.nuclear_repulsion_energy:.15f} Eh")
print(f"Total energy: {result.energy:.15f} Eh")
print(f"Orbital-gradient norm: {result.history[-1].gradient_norm:.3e}")
print("Orbital energies:")
for label, energy in zip(result.orbital_labels, result.orbital_energies):
    index, magnetic = label
    print(f"  (index, m) = ({index}, {magnetic}): {energy:.15f} Eh")
