# Spherical electric-dipole fields

`SphericalDipoleField` applies a spatially uniform external field to spherical
orbitals in either length or velocity gauge. Temporal profiles are supplied by
the `electromagnetic_fields` package; tdmcscf provides the spatial action.

For reduced radial orbitals

$$
\phi_p(\mathbf r)
=\frac{1}{r}\sum_{lm}u^p_{lm}(r)Y_{lm}(\Omega),
$$

the length-gauge interaction for a particle with charge $q$ is

$$
V_{\mathrm L}(t)=-q\,\mathbf E(t)\cdot\mathbf r.
$$

The velocity-gauge interaction implemented here is

$$
V_{\mathrm V}(t)=-\frac{q}{\mu}\mathbf A(t)\cdot\mathbf p.
$$

The spatially uniform scalar term $q^2A(t)^2/(2\mu)$ is omitted. It only
changes the global phase of the wavefunction. The defaults `charge=-1` and
`mass=1` give the electronic atomic-unit conventions
$V_{\mathrm L}=\mathbf E\cdot\mathbf r$ and
$V_{\mathrm V}=\mathbf A\cdot\mathbf p$.

## One profile with fixed polarization

An `electromagnetic_fields` object is constructed in the same gauge as the
spatial operator:

```python
from electromagnetic_fields.dipole_fields.laser_pulses import SineSquareLaser
from tdmcscf.grid.spherical import SphericalDipoleField

pulse = SineSquareLaser(
    E0=0.05,
    omega=0.8,
    ncycles=6,
    gauge="length",
)
interaction = SphericalDipoleField.from_profile(
    radial_grid,
    l_max=4,
    field=pulse,
    polarization=(1.0, 0.0, 0.0),
    gauge="length",
)

Vu = interaction.apply(orbitals, time)
```

This represents

$$
\mathbf E(t)=f(t)\boldsymbol{\epsilon}.
$$

The polarization is not normalized automatically. Its magnitude can carry a
relative amplitude when several fields are combined.
The high-level `SphericalTDHF` interface instead normalizes each polarization
so that physical field strengths remain entirely in the temporal profiles.

## Independent components

Separate component profiles permit a direction that changes with time:

```python
interaction = SphericalDipoleField.from_components(
    radial_grid,
    l_max=4,
    gauge="velocity",
    x=vector_potential_x,
    y=vector_potential_y,
    z=None,
)
```

This represents

$$
\mathbf A(t)=\bigl(A_x(t),A_y(t),A_z(t)\bigr).
$$

Phase-shifted $x$ and $y$ profiles describe circular or elliptical
polarization.

## Sums of polarized fields

The general constructor accepts any finite collection of `DipoleFieldTerm`
objects:

```python
from tdmcscf.grid.spherical import DipoleFieldTerm, SphericalDipoleField

interaction = SphericalDipoleField(
    radial_grid,
    l_max=4,
    gauge="length",
    fields=[
        DipoleFieldTerm(pulse_1, (0.0, 0.0, 1.0)),
        DipoleFieldTerm(pulse_2, (1.0, 0.0, 0.0)),
        DipoleFieldTerm(pulse_3, (1.0, 1.0, 0.0)),
    ],
)
```

At every application, the operator evaluates

$$
\mathbf E(t)=\sum_{k=1}^{n}f_k(t)\boldsymbol{\epsilon}_k
$$

or the corresponding sum of vector potentials in velocity gauge. Each
temporal callable is evaluated once. `field_vector(t)` exposes the resulting
three Cartesian components.

## Discrete action

Let $C_\alpha$ denote the angular matrix of
$\hat r_\alpha$ and let $H_\alpha$ contain the tangential derivative and the
derivative of the explicit $1/r$ factor. The reduced radial actions are

$$
(V_{\mathrm L}u)_{aI}
=-q\sum_{\alpha J}E_\alpha(t)r_a
(C_\alpha)_{IJ}u_{aJ},
$$

$$
(V_{\mathrm V}u)_{aI}
=\frac{iq}{\mu}\sum_{\alpha J}A_\alpha(t)
\left[
(C_\alpha)_{IJ}\sum_bD^{(1)}_{ab}u_{bJ}
+\frac{(H_\alpha)_{IJ}}{r_a}u_{aJ}
\right].
$$

The implementation uses grid-lib's nodal `D1` and analytic spherical-harmonic
momentum couplings. Arrays contain nodal values and exclude both radial
endpoints, as for the kinetic-energy and mean-field operators. No quadrature
weights are multiplied into `apply`; the weights remain part of inner
products.

The result is projected into the configured angular space. A Cartesian
dipole changes $l$ by one. The $z$ component preserves $m$, while the $x$ and
$y$ components change $m$ by one. Couplings outside `l_max` or `m_max` are
discarded, so a calculation using transverse polarization must retain the
required magnetic channels.
