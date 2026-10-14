"""Propagate a closed-shell atom in a linearly z-polarized laser field."""

import matplotlib.pyplot as plt
import numpy as np
from electromagnetic_fields.dipole_fields import SineSquareLaser

from tdmcscf.grid.spherical import RadialCosineMask, SphericalTDHF
from tdmcscf.systems import Atomic, atomic_configurations


available_atoms = ", ".join(atomic_configurations())
print()
print(f"Available closed-shell atoms and ions: {available_atoms}")
atom = input("Select an atom or ion: ").strip()
system = Atomic(atom)

# FEM-DVR element boundaries and Lobatto points in each element.
nodes = [0.0, 4.0, 8.0, 12.0, 16.0]
n_points = [11, 11, 11, 11]

# Orbital and electron-electron Coulomb angular spaces.
l_max = 4
m_max = 4
L_max = 2 * l_max
M_max = L_max

# The electromagnetic and occupied-orbital gauges are independent choices.
field_gauge = "velocity"  # "velocity" or "length"
orbital_gauge = "orthogonal"  # "orthogonal" or "canonical"

# One-cycle sine-square pulse, linearly polarized along z. Polarizations are
# normalized by SphericalTDHF, so their magnitude does not rescale E0.
E0 = 1e-4
omega = 0.057*4
phase = 0.0
ncycles = 1
pulse = SineSquareLaser(
    E0=E0,
    omega=omega,
    ncycles=ncycles,
    gauge=field_gauge,
    phase=phase,
)

# Propagation controls. The actual time step is adjusted slightly so that the
# final point coincides exactly with the requested duration.
integrator = "rk4"  # "strang2" or "rk4"
integrator_options = (
    {"solver": "gmres", "rtol": 1e-11, "atol": 0.0}
    if integrator == "strang2"
    else None
)
requested_time_step = 0.001
duration = 2 * np.pi * ncycles / omega

# Optional boundary mask and checkpointing. Leave these as None to disable
# them. For example: mask = RadialCosineMask(r0=12.0).
mask = None
checkpoint_path = None
checkpoint_interval = None

calculation = SphericalTDHF(
    system,
    nodes=nodes,
    n_points=n_points,
    l_max=l_max,
    m_max=m_max,
    L_max=L_max,
    M_max=M_max,
    poisson_method="solve",
    initial_state="groundstate",
    ground_state_options={
        "max_iterations": 100,
        "conv_tol_e": 1e-12,
        "conv_tol_grad": 1e-10,
        "damping": 0.2,
        "diis": True,
        "diis_start": 3,
        "diis_size": 8,
    },
    fields=[(pulse, (0.0, 0.0, 1.0))],
    field_gauge=field_gauge,
    orbital_gauge=orbital_gauge,
    integrator=integrator,
    integrator_options=integrator_options,
    time_step=requested_time_step,
    final_time=duration,
    mask=mask,
    checkpoint_path=checkpoint_path,
    checkpoint_interval=checkpoint_interval,
)
result = calculation.run(progress=True)

orbital_norms = result.orbital_norms
orbital_z = result.position_expectations[:, :, 2]
total_dipole_z = result.total_dipole[:, 2]
quantum_numbers = calculation.ground_state.orbital_quantum_numbers

title = f"TDHF propagation of {system.label}"
print(f"\n{title}")
print("=" * len(title))
print(f"{'Field gauge':<24}: {field_gauge}")
print(f"{'Orbital gauge':<24}: {orbital_gauge}")
print(f"{'Integrator':<24}: {integrator}")
print(f"{'Element boundaries':<24}: {nodes} bohr")
print(f"{'Points per element':<24}: {n_points}")
print(f"{'(l_max, m_max)':<24}: ({l_max}, {m_max})")
print(f"{'(L_max, M_max)':<24}: ({L_max}, {M_max})")
print(f"{'Pulse duration':<24}: {duration:.12f} a.u.")
print(f"{'Time step':<24}: {result.time_step:.12f} a.u.")
print(f"{'Number of steps':<24}: {result.n_steps}")
print(f"{'Magnetic m conserved':<24}: {calculation.m_conserved}")
print(f"{'Maximum norm change':<24}: {np.max(abs(orbital_norms - 1)):.3e}")

number_of_orbitals = len(quantum_numbers)
figure, axes = plt.subplots(
    number_of_orbitals + 2,
    1,
    sharex=True,
    figsize=(8, 2.2 * (number_of_orbitals + 2)),
)
dipole_axes = axes[0]
position_axes = axes[1:-1]
norm_axes = axes[-1]
dipole_axes.plot(result.times, total_dipole_z)
for p, (numbers, orbital_axes) in enumerate(zip(quantum_numbers, position_axes)):
    n, l, m = numbers
    label = rf"$n={n},\,l={l},\,m={m}$"
    orbital_axes.plot(result.times, orbital_z[:, p], label=label)
    orbital_axes.set_ylabel(r"$\langle z\rangle_p$ (bohr)")
    orbital_axes.legend()
    norm_axes.plot(result.times, orbital_norms[:, p], label=label)

dipole_axes.set_ylabel(r"$d_z(t)$ (a.u.)")
dipole_axes.set_title(f"Dipole response of {system.label}")
norm_axes.set_xlabel("Time (a.u.)")
norm_axes.set_ylabel(r"$\langle \phi_p|\phi_p\rangle$")
norm_axes.legend()
figure.tight_layout()
plt.show()
