r"""Electric-dipole field actions on reduced radial nodal orbitals."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from numbers import Real

import numpy as np
from grid_lib.spherical_coordinates.momentum_operators import MomentumOperators

from ._validation import _integer, _orbitals, _positive_real, _radial_data


def _finite_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar.")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar.")
    return float(value)


def _polarization(value):
    array = np.asarray(value)
    if (
        array.shape != (3,)
        or not np.issubdtype(array.dtype, np.number)
        or np.iscomplexobj(array)
        or not np.all(np.isfinite(array))
    ):
        raise ValueError("polarization must contain three finite real components.")
    array = np.asarray(array, dtype=float)
    if not np.any(array):
        raise ValueError("polarization cannot be the zero vector.")
    return tuple(float(component) for component in array)


@dataclass(frozen=True, init=False)
class DipoleFieldTerm:
    r"""One scalar temporal profile and its fixed Cartesian polarization.

    ``field`` is normally a gauge-specific callable from
    :mod:`electromagnetic_fields`. It must return a finite real scalar. The
    polarization is not normalized, so it may also carry a relative amplitude.
    """

    field: Callable[[float], Real]
    polarization: tuple[float, float, float]

    def __init__(self, field, polarization=(0.0, 0.0, 1.0)):
        if not callable(field):
            raise TypeError("field must be callable.")
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "polarization", _polarization(polarization))


class SphericalDipoleField:
    r"""Length- or velocity-gauge dipole interaction in spherical coordinates.

    Parameters
    ----------
    radial_grid
        Fixed real grid with full ``r``, ``weights``, and nodal ``D1`` arrays.
        The first and last nodes are removed, imposing zero reduced radial
        orbitals at both endpoints.
    l_max : int
        Maximum orbital angular momentum.
    m_max : int, optional
        Retain ``abs(m) <= m_max``. Defaults to ``l_max``.
    gauge : {"length", "velocity"}
        Gauge of every supplied temporal profile. Length-gauge profiles return
        electric-field amplitudes; velocity-gauge profiles return vector-
        potential amplitudes.
    fields : DipoleFieldTerm or iterable of DipoleFieldTerm
        Terms in the total Cartesian field. Their vectors are summed whenever
        :meth:`apply` or :meth:`field_vector` is called.
    charge : finite real scalar, optional
        Particle charge in atomic units. Defaults to ``-1`` for an electron.
    mass : positive real scalar, optional
        Particle mass in electron-mass units. Defaults to ``1``.

    Notes
    -----
    For ``phi = sum_lm u_lm(r) Y_lm(Omega)/r``, the length-gauge
    interaction is ``-charge * E(t).r``. The velocity-gauge interaction is
    ``-(charge/mass) * A(t).p``. The spatially uniform ``A(t)^2/(2*mass)``
    term is deliberately omitted.

    Angular action is projected into the retained ``(l,m)`` space. Therefore
    couplings beyond ``l_max`` or ``m_max`` are discarded. In particular,
    x- and y-polarized fields require magnetic channels with ``Delta m = +/-1``.
    """

    _AXES = np.eye(3)

    def __init__(
        self,
        radial_grid,
        l_max,
        m_max=None,
        *,
        gauge,
        fields,
        charge=-1.0,
        mass=1.0,
    ):
        self.l_max = _integer(l_max, "l_max")
        self.m_max = self.l_max if m_max is None else _integer(m_max, "m_max")
        if self.m_max > self.l_max:
            raise ValueError("m_max cannot exceed l_max.")
        if gauge not in {"length", "velocity"}:
            raise ValueError("gauge must be 'length' or 'velocity'.")
        self.gauge = gauge
        self.charge = _finite_real(charge, "charge")
        self.mass = _positive_real(mass, "mass")
        self.fields = self._field_terms(fields)

        r, weights = _radial_data(radial_grid)
        D1 = np.asarray(radial_grid.D1)
        if (
            D1.shape != (len(r), len(r))
            or np.iscomplexobj(D1)
            or not np.all(np.isfinite(D1))
        ):
            raise ValueError(
                "D1 must be a finite real nodal matrix on the full radial grid."
            )

        self.r = r[1:-1].copy()
        self.radial_weights = weights[1:-1].copy()
        self.orbital_lm = tuple(
            (l, m)
            for m in range(-self.m_max, self.m_max + 1)
            for l in range(abs(m), self.l_max + 1)
        )
        self.orbital_shape = (len(self.r), len(self.orbital_lm))
        self._momentum = MomentumOperators(
            self.l_max,
            self.m_max,
            self.r,
            self.radial_weights,
            D1[1:-1, 1:-1].copy(),
        )
        self._direction_cosines = np.array(
            [
                -0.5 * self._momentum.px_lm_1,
                0.5j * self._momentum.py_lm_1,
                self._momentum.pz_lm_1,
            ]
        )

    @staticmethod
    def _field_terms(fields):
        if isinstance(fields, DipoleFieldTerm):
            return (fields,)
        if isinstance(fields, (str, bytes)) or not isinstance(fields, Iterable):
            raise TypeError("fields must be a DipoleFieldTerm or an iterable of them.")
        terms = tuple(fields)
        if not terms:
            raise ValueError("fields must contain at least one DipoleFieldTerm.")
        if not all(isinstance(term, DipoleFieldTerm) for term in terms):
            raise TypeError("Every field term must be a DipoleFieldTerm.")
        return terms

    @classmethod
    def from_profile(
        cls,
        radial_grid,
        l_max,
        field,
        polarization=(0.0, 0.0, 1.0),
        m_max=None,
        *,
        gauge,
        charge=-1.0,
        mass=1.0,
    ):
        """Construct one scalar profile with a fixed polarization vector."""
        return cls(
            radial_grid,
            l_max,
            m_max,
            gauge=gauge,
            fields=DipoleFieldTerm(field, polarization),
            charge=charge,
            mass=mass,
        )

    @classmethod
    def from_components(
        cls,
        radial_grid,
        l_max,
        *,
        gauge,
        x=None,
        y=None,
        z=None,
        m_max=None,
        charge=-1.0,
        mass=1.0,
    ):
        """Construct independent Cartesian temporal profiles.

        At least one of ``x``, ``y``, and ``z`` must be a callable. Use
        separate phase-shifted profiles for circular or elliptical fields.
        """
        components = (x, y, z)
        terms = [
            DipoleFieldTerm(field, cls._AXES[axis])
            for axis, field in enumerate(components)
            if field is not None
        ]
        return cls(
            radial_grid,
            l_max,
            m_max,
            gauge=gauge,
            fields=terms,
            charge=charge,
            mass=mass,
        )

    def field_vector(self, t):
        """Evaluate and sum all Cartesian field terms at time ``t``."""
        vector = np.zeros(3)
        for index, term in enumerate(self.fields):
            value = np.asarray(term.field(t))
            if (
                value.ndim != 0
                or not np.issubdtype(value.dtype, np.number)
                or np.iscomplexobj(value)
                or not np.isfinite(value)
            ):
                raise ValueError(
                    f"Field term {index} must return a finite real scalar."
                )
            vector += float(value) * np.asarray(term.polarization)
        return vector

    def apply(self, orbitals, t):
        """Return the dipole-field action at time ``t``.

        Accepts one orbital with shape ``(n_r, n_lm)`` or a batch with shape
        ``(n_orbitals, n_r, n_lm)`` and preserves that input shape.
        """
        values, single = _orbitals(orbitals, self.orbital_shape, allow_single=True)
        field = self.field_vector(t)

        if self.gauge == "length":
            angular = np.einsum(
                "x,xij,paj->pai", field, self._direction_cosines, values
            )
            result = -self.charge * self.r[None, :, None] * angular
        else:
            momentum = self._momentum.get_momentum_vector(values)
            result = -(self.charge / self.mass) * np.einsum(
                "x,xpai->pai", field, momentum
            )

        return result[0] if single else result

    def __call__(self, orbitals, t):
        """Alias for :meth:`apply`."""
        return self.apply(orbitals, t)
