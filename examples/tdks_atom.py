"""Propagate a closed-shell atom with adiabatic spin-unpolarized TDKS."""

import matplotlib.pyplot as plt
import numpy as np
from electromagnetic_fields.dipole_fields import SineSquareLaser

from tdmcscf.dft import LDAVWN, PBE
from tdmcscf.grid.spherical import RadialCosineMask, SphericalTDKS
from tdmcscf.systems import Atomic, atomic_configurations


available_atoms = ", ".join(atomic_configurations())
print()
print(f"Available closed-shell atoms and ions: {available_atoms}")
atom = input("Select an atom or ion: ").strip()
system = Atomic(atom)

# FEM-DVR element boundaries and Lobatto points in each element.
nodes = [0.0, 4.0, 8.0, 12.0, 16.0]
n_points = [11, 11, 11, 11]

# Orbital, Hartree, and local-XC angular controls.
l_max = 4
m_max = 0
L_max = 2 * l_max
M_max = 2 * m_max
xc_angular_bandlimit = 4 * l_max
# None selects twice n_points in each element for semilocal functionals.
xc_quadrature_order = None

# The functional is used for both the initial Kohn-Sham ground state and the
# instantaneous adiabatic XC potential during propagation. PBE() may be used
# here for the standard generalized-gradient approximation.
functional = LDAVWN()

# The electromagnetic and occupied-orbital gauges are independent choices.
field_gauge = "velocity"  # "velocity" or "length"
orbital_gauge = "orthogonal"  # "orthogonal" or "canonical"

# One-cycle sine-square pulse, linearly polarized along z. Polarization vectors
# are normalized internally, so their magnitude does not rescale E0.
E0 = 1e-4
omega = 0.057
phase = 0.0
ncycles = 1
pulse = SineSquareLaser(
    E0=E0,
    omega=omega,
    ncycles=ncycles,
    gauge=field_gauge,
    phase=phase,
)

# Propagation controls. IM-CMF and IM-CDM freeze the same density-dependent
# Hartree-XC operator in TDKS; both names remain available so the
# interface matches SphericalTDHF.
integrator = "im-cmf2"
# Choices: "strang2", "rk4", "im-cmf1", "im-cmf2", "im-cdm1", "im-cdm2"
if integrator in {
    "strang2",
    "im-cmf1",
    "im-cmf2",
    "im-cdm1",
    "im-cdm2",
}:
    integrator_options = {"solver": "gmres", "rtol": 0.0, "atol": 1e-12}
    if integrator in {"im-cmf2", "im-cdm2"}:
        integrator_options["corrector_steps"] = 1
else:
    integrator_options = None
requested_time_step = 0.2
duration = 2 * np.pi * ncycles / omega

# Optional boundary mask and checkpointing. Leave these as None to disable
# them. For example: mask = RadialCosineMask(r0=12.0).
mask = None
checkpoint_path = None
checkpoint_interval = None

calculation = SphericalTDKS(
    system,
    functional=functional,
    nodes=nodes,
    n_points=n_points,
    l_max=l_max,
    m_max=m_max,
    L_max=L_max,
    M_max=M_max,
    xc_angular_bandlimit=xc_angular_bandlimit,
    xc_quadrature_order=xc_quadrature_order,
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

title = f"TDKS propagation of {system.label}"
print(f"\n{title}")
print("=" * len(title))
print(f"{'Functional':<26}: {functional.name}")
print(f"{'Field gauge':<26}: {field_gauge}")
print(f"{'Orbital gauge':<26}: {orbital_gauge}")
print(f"{'Integrator':<26}: {integrator}")
if calculation.corrector_steps is not None:
    print(f"{'Corrector steps':<26}: {calculation.corrector_steps}")
print(f"{'Element boundaries':<26}: {nodes} bohr")
print(f"{'Points per element':<26}: {n_points}")
print(f"{'(l_max, m_max)':<26}: ({l_max}, {m_max})")
print(f"{'(L_max, M_max)':<26}: ({L_max}, {M_max})")
print(f"{'XC angular bandlimit':<26}: {xc_angular_bandlimit}")
print(
    f"{'XC radial quadrature':<26}: "
    f"{calculation.xc_quadrature_order}"
)
print(f"{'Pulse duration':<26}: {duration:.12f} a.u.")
print(f"{'Time step':<26}: {result.time_step:.12f} a.u.")
print(f"{'Number of steps':<26}: {result.n_steps}")
print(f"{'Magnetic m conserved':<26}: {calculation.m_conserved}")
print(f"{'Maximum norm change':<26}: {np.max(abs(orbital_norms - 1)):.3e}")

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
    norm_error = np.abs(1.0 - orbital_norms[:, p])
    norm_axes.semilogy(result.times, norm_error, label=label)

dipole_axes.set_ylabel(r"$d_z(t)$ (a.u.)")
dipole_axes.set_title(f"Dipole response of {system.label}")
norm_axes.set_xlabel("Time (a.u.)")
norm_axes.set_ylabel(r"$|1-\langle \phi_p|\phi_p\rangle|$")
norm_axes.legend()
figure.tight_layout()
plt.show()
