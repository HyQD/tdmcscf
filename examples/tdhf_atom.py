"""Propagate a closed-shell atom in a linearly z-polarized laser field."""

import matplotlib.pyplot as plt
import numpy as np
from electromagnetic_fields.dipole_fields.laser_pulses import SineSquareLaser
from tqdm.auto import tqdm

from tdmcscf.grid.spherical import (
    ShiftedAtomicCorePreconditioner,
    SphericalAtomicHartreeFock,
    SphericalDipoleField,
    SphericalMeanFieldBuilder,
    SphericalRestrictedTDHF,
)
from tdmcscf.integrators import CrankNicolson, RK4, Strang2
from tdmcscf.systems import atomic_configurations, closed_shell_atom


available_atoms = ", ".join(atomic_configurations())
print()
print(f"Available closed-shell atoms and ions: {available_atoms}")
atom = input("Select an atom or ion: ").strip()
configuration = closed_shell_atom(atom)

# FEM-DVR element boundaries and Lobatto points in each element.
nodes = [0.0, 4.0, 8.0, 12.0, 16.0]
n_points = [11, 11, 11, 11]

# Orbital angular basis. For a z-polarized field, m is conserved, so the
# occupied ground-state m range is sufficient. Replace None explicitly to
# retain additional magnetic channels.
l_max = 4
m_max = None
m_max = configuration.occupied_m_max if m_max is None else m_max

# Electron-electron Coulomb multipoles. The displayed defaults retain the
# complete product space of the chosen orbital basis.
L_max = 2 * l_max
M_max = None
M_max = min(L_max, 2 * m_max) if M_max is None else M_max

# The electromagnetic and occupied-orbital gauges are independent choices.
field_gauge = "velocity"  # "velocity" or "length"
orbital_gauge = "orthogonal"  # "orthogonal" or "canonical"

# One-cycle sine-square pulse, linearly polarized along z.
E0 = 1e-4
omega = 0.057
phase = 0.0
ncycles = 1

# Propagation and iterative linear-solver controls. The final step is adjusted
# slightly so that the propagation ends at exactly one optical cycle.
requested_time_step = 0.01
krylov_rtol = 1e-11
krylov_atol = 0.0

# Hartree--Fock ground-state controls.
ground_state_solver = SphericalAtomicHartreeFock(
    configuration,
    nodes=nodes,
    n_points=n_points,
    l_max=l_max,
    m_max=m_max,
    coulomb_l_max=2 * configuration.occupied_l_max,
    max_iterations=100,
    conv_tol_e=1e-12,
    conv_tol_grad=1e-10,
    damping=0.2,
    diis=True,
    diis_start=3,
    diis_size=8,
    poisson_method="solve",
)
ground_state = ground_state_solver.solve()
if not ground_state.converged:
    raise RuntimeError("The Hartree--Fock initial state did not converge.")

pulse = SineSquareLaser(
    E0=E0,
    omega=omega,
    ncycles=ncycles,
    gauge=field_gauge,
    phase=phase,
)
dipole_field = SphericalDipoleField.from_profile(
    ground_state.grid,
    l_max,
    pulse,
    polarization=(0.0, 0.0, 1.0),
    m_max=m_max,
    gauge=field_gauge,
)
mean_fields = SphericalMeanFieldBuilder(
    ground_state.grid,
    l_max,
    m_max,
    L_max=L_max,
    M_max=M_max,
)
orbital_m = tuple(m for _, _, m in ground_state.orbital_quantum_numbers)
tdhf = SphericalRestrictedTDHF(
    ground_state_solver.core,
    mean_fields,
    dipole_field=dipole_field,
    orbital_gauge=orbital_gauge,
    orbital_m=orbital_m,
)

duration = 2 * np.pi * ncycles / omega
n_steps = int(np.ceil(duration / requested_time_step))
time_step = duration / n_steps

# Each Crank--Nicolson call advances half of one Strang step.
preconditioner = ShiftedAtomicCorePreconditioner(
    ground_state_solver.core,
    step_size=time_step / 2,
)
crank_nicolson = CrankNicolson(
    solver="gmres",
    rtol=krylov_rtol,
    atol=krylov_atol,
    preconditioner=preconditioner,
)
integrator = Strang2(crank_nicolson, RK4())

# A unit length-gauge z interaction applies the position operator z=r cos(theta).
position_z = SphericalDipoleField.from_profile(
    ground_state.grid,
    l_max,
    lambda time: 1.0,
    polarization=(0.0, 0.0, 1.0),
    m_max=m_max,
    gauge="length",
)

times = np.linspace(0.0, duration, n_steps + 1)
orbital_norms = np.empty((n_steps + 1, configuration.n_doubly_occupied))
orbital_z = np.empty_like(orbital_norms)


def sample(index, orbitals):
    """Record individual spatial-orbital norms and z expectations."""
    orbital_norms[index] = tdhf.orbital_norms(orbitals)
    z_orbitals = position_z.apply(orbitals, times[index])
    values = np.einsum(
        "pai,a,pai->p",
        orbitals.conj(),
        tdhf.radial_weights,
        z_orbitals,
        optimize=True,
    )
    if np.max(np.abs(values.imag)) > 1e-9:
        raise RuntimeError("An orbital z expectation acquired an imaginary part.")
    orbital_z[index] = values.real


orbitals = ground_state.orbitals.copy()
sample(0, orbitals)
for step in tqdm(range(n_steps), desc="TDHF propagation"):
    orbitals = integrator(
        orbitals,
        times[step],
        time_step,
        tdhf.linear_rhs,
        tdhf.nonlinear_rhs,
    )
    sample(step + 1, orbitals)

# Every spatial orbital is doubly occupied. With the nucleus fixed at the
# origin and the electronic charge equal to -1, the total dipole is
# d_z(t) = -2 sum_p <phi_p(t)|z|phi_p(t)>.
total_dipole = -2 * orbital_z.sum(axis=1)

title = f"TDHF propagation of {configuration.symbol}"
print(f"\n{title}")
print("=" * len(title))
print(f"{'Field gauge':<24}: {field_gauge}")
print(f"{'Orbital gauge':<24}: {orbital_gauge}")
print(f"{'Element boundaries':<24}: {nodes} bohr")
print(f"{'Points per element':<24}: {n_points}")
print(f"{'(l_max, m_max)':<24}: ({l_max}, {m_max})")
print(f"{'(L_max, M_max)':<24}: ({L_max}, {M_max})")
print(f"{'Pulse duration':<24}: {duration:.12f} a.u.")
print(f"{'Time step':<24}: {time_step:.12f} a.u.")
print(f"{'Number of steps':<24}: {n_steps}")
print(f"{'Maximum norm change':<24}: {np.max(abs(orbital_norms - 1)):.3e}")

number_of_orbitals = len(ground_state.orbital_quantum_numbers)
figure, axes = plt.subplots(
    number_of_orbitals + 2,
    1,
    sharex=True,
    figsize=(8, 2.2 * (number_of_orbitals + 2)),
)
dipole_axes = axes[0]
position_axes = axes[1:-1]
norm_axes = axes[-1]
dipole_axes.plot(times, total_dipole)
for p, (quantum_numbers, orbital_axes) in enumerate(
    zip(ground_state.orbital_quantum_numbers, position_axes)
):
    n, l, m = quantum_numbers
    label = rf"$n={n},\,l={l},\,m={m}$"
    orbital_axes.plot(times, orbital_z[:, p], label=label)
    orbital_axes.set_ylabel(r"$\langle z\rangle_p$ (bohr)")
    orbital_axes.legend()
    norm_axes.plot(times, orbital_norms[:, p], label=label)

dipole_axes.set_ylabel(r"$d_z(t)$ (a.u.)")
dipole_axes.set_title(f"Dipole response of {configuration.symbol}")
norm_axes.set_xlabel("Time (a.u.)")
norm_axes.set_ylabel(r"$\langle \phi_p|\phi_p\rangle$")
norm_axes.legend()
figure.tight_layout()
plt.show()
