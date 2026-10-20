"""Study polynomial-order convergence of an atomic RKS ground state."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tdmcscf.dft import LDAVWN, PBE, XAlphaExchange, XAlphaVWN
from tdmcscf.grid.spherical import SphericalAtomicKohnSham
from tdmcscf.systems import atomic_configurations, closed_shell_atom


def prompt_float(prompt, *, positive=False, nonnegative=False):
    """Prompt until the user supplies a valid floating-point value."""
    while True:
        try:
            value = float(input(prompt))
        except ValueError:
            print("Please enter a number.")
            continue
        if positive and value <= 0.0:
            print("The value must be positive.")
            continue
        if nonnegative and value < 0.0:
            print("The value must be nonnegative.")
            continue
        return value


def prompt_pmax():
    """Return an odd maximum point count with at least two sampled orders."""
    while True:
        try:
            value = int(input("Maximum number of points per element (odd, >= 7): "))
        except ValueError:
            print("Please enter an integer.")
            continue
        if value < 7 or value % 2 == 0:
            print("pmax must be an odd integer greater than or equal to 7.")
            continue
        return value


def select_functional():
    """Prompt for one of the built-in complete or exchange-only functionals."""
    choices = {
        "lda-vwn": lambda: LDAVWN(),
        "pbe": lambda: PBE(),
        "xalpha": lambda: XAlphaExchange(
            alpha=prompt_float("X-alpha parameter alpha: ", nonnegative=True)
        ),
        "xalpha-vwn": lambda: XAlphaVWN(
            alpha=prompt_float("X-alpha parameter alpha: ", nonnegative=True)
        ),
    }
    aliases = {
        "lda": "lda-vwn",
        "x-alpha": "xalpha",
        "x-alpha-vwn": "xalpha-vwn",
    }
    print("Available functionals: LDA-VWN, PBE, X-alpha, X-alpha-VWN")
    while True:
        name = input("Select a functional: ").strip().lower()
        name = aliases.get(name, name)
        if name in choices:
            return choices[name]()
        print("Choose LDA-VWN, PBE, X-alpha, or X-alpha-VWN.")


available_atoms = ", ".join(atomic_configurations())
print()
print(f"Available closed-shell atoms and ions: {available_atoms}")
atom = input("Select an atom or ion: ").strip()
configuration = closed_shell_atom(atom)
functional = select_functional()

element_size = prompt_float("Element size (bohr): ", positive=True)
r_max = prompt_float("Maximum radius r_max (bohr): ", positive=True)
p_max = prompt_pmax()

number_of_elements = r_max / element_size
rounded_number_of_elements = int(round(number_of_elements))
if not np.isclose(
    number_of_elements,
    rounded_number_of_elements,
    rtol=0.0,
    atol=100.0 * np.finfo(float).eps * max(1.0, number_of_elements),
):
    raise ValueError(
        "r_max must be an integer multiple of the element size so that all "
        "elements have equal length."
    )

nodes = np.linspace(0.0, r_max, rounded_number_of_elements + 1)
p_values = np.arange(5, p_max + 1, 2, dtype=int)
energies = []

print()
print(f"System          : {configuration.symbol}")
print(f"Functional      : {functional.name}")
print(f"Element size    : {element_size:g} bohr")
print(f"Maximum radius  : {r_max:g} bohr")
print(f"Elements        : {rounded_number_of_elements}")
print()
print("  p              EKS (Eh)  conv.  iterations       |dE|       gradient")

for p in p_values:
    solver = SphericalAtomicKohnSham(
        configuration,
        functional=functional,
        nodes=nodes,
        n_points=int(p),
        max_iterations=100,
        conv_tol_e=1e-12,
        conv_tol_grad=1e-8,
        damping=0.2,
        diis=True,
        diis_start=3,
        diis_size=8,
        initial_guess="auto",
        poisson_method="solve",
    )
    result = solver.solve()
    energies.append(result.energy)
    last = result.history[-1]
    print(
        f"{p:3d}  {result.energy:22.12f}  "
        f"{'yes' if result.converged else ' no':>5s}  "
        f"{result.iterations:10d}  {last.energy_change:9.2e}  "
        f"{last.gradient_norm:9.2e}"
    )

energies = np.asarray(energies)
reference_energy = energies[-1]
relative_differences = np.abs(energies - reference_energy) / abs(reference_energy)

figure, axes = plt.subplots(figsize=(7.0, 4.5))
# The pmax value is the reference itself and is exactly zero, so it cannot be
# represented on a logarithmic axis.
plotted = relative_differences[:-1] > 0.0
axes.semilogy(
    p_values[:-1][plotted],
    relative_differences[:-1][plotted],
    linestyle="-",
    marker="o",
)
ticks = p_values[::2]
if ticks[-1] != p_values[-1]:
    ticks = np.append(ticks, p_values[-1])
axes.set_xticks(ticks)
axes.set_xlim(p_values[0], p_values[-1])
axes.set_xlabel("Points per element, p")
axes.set_ylabel(r"$|E(p)-E(p_{\max})|/|E(p_{\max})|$")
axes.set_title(
    f"{configuration.symbol}, {functional.name}, "
    rf"$r_{{\max}}={r_max:g}$ bohr"
)
axes.grid(True, which="both", alpha=0.3)
figure.tight_layout()

functional_label = functional.name.lower().replace(" ", "-")
functional_label = functional_label.replace("/", "-")
filename = Path(
    f"rks_p_convergence_{configuration.symbol}_{functional_label}_"
    f"rmax={r_max:g}.pdf"
)
figure.savefig(filename)
print()
print(f"Reference energy E(pmax): {reference_energy:.15f} Eh")
print("The identically zero p=pmax difference is omitted from the log plot.")
print(f"Saved figure: {filename}")
plt.show()
