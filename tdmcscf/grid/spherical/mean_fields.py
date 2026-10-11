r"""Coulomb mean fields for phi_p(r,Omega) = sum_lm u_p,lm(r) Y_lm(Omega)/r.

Orbital arrays contain reduced radial nodal values, with shape
(n_orbitals, n_r, n_lm). The radial axis contains only interior grid nodes.
Radial quadrature weights are neither absorbed into orbitals nor multiplied
into their pointwise pair products. They are included in the Coulomb response
matrices supplied by grid-lib.
"""

import numpy as np

from grid_lib.spherical_coordinates import GaussLegendreFourierGrid, radial_Coulomb

from ._angular import AngularTransform


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


def _pair(value):
    pair = tuple(value)
    if len(pair) != 2:
        raise ValueError("Each pair must contain two orbital indices (q, s).")
    return tuple(_integer(i, "orbital index") for i in pair)


def _orbitals(values, shape, allow_single=False):
    array = np.asarray(values)
    single = allow_single and array.ndim == 2
    if single:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[1:] != shape or array.shape[0] == 0:
        raise ValueError(
            f"Orbitals must have shape (n_orbitals, {shape[0]}, {shape[1]})"
            + (f" or {shape}." if allow_single else ".")
        )
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("Orbital values must be numeric.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Orbital values must be finite.")
    return np.asarray(array, dtype=np.complex128), single


def _magnetic_numbers(values, orbitals, transform):
    if values is None:
        return None
    numbers = np.asarray(values)
    if numbers.ndim == 0:
        numbers = np.repeat(numbers, len(orbitals))
    if numbers.shape != (len(orbitals),) or numbers.dtype.kind not in "iu":
        raise ValueError("orbital_m must supply one integer m per orbital.")
    numbers = numbers.astype(int)
    channel_m = np.array([m for _, m in transform.channels])
    for orbital, m in zip(orbitals, numbers):
        if m not in transform.indices:
            raise ValueError(f"m={m} is outside the retained orbital channels.")
        if np.any(orbital[:, channel_m != m] != 0):
            raise ValueError("Fixed-m orbitals must be zero outside their declared m.")
    return numbers


class SphericalMeanFieldBuilder:
    r"""Cache quadrature, angular transforms, and dense radial Coulomb responses.

    Parameters
    ----------
    radial_grid
        Fixed real radial grid starting at zero, with ``r``, ``weights``,
        and nodal ``D2`` arrays, as provided by grid-lib's FEMDVR. The first
        and last nodes are excluded from orbital arrays. The Coulomb response
        uses free-space multipole boundary conditions at the outer radius.
    l_max : int
        Highest retained orbital angular momentum. Field multipoles extend
        through ``2*l_max``.
    m_max : int, optional
        Retain orbital channels with ``abs(m) <= m_max``. Defaults to l_max.
    angular_grid : GaussLegendreFourierGrid, optional
        Default: ``from_bandlimit(2*l_max)``. A supplied grid must have at
        least ``2*l_max+1`` polar and ``4*m_max+1`` azimuthal nodes, sufficient
        for full pair-density analysis and projected mean-field action.

    Notes
    -----
    ``orbital_lm`` and ``multipole_lm`` expose the channel ordering, matching
    grid-lib's LM_to_I convention. The radial nodes ``r`` and weights are
    interior-node copies. ``radial_matrices[L]`` already includes source
    weights and 4*pi/(2*L+1). It is built once per builder, not per orbital
    update. Its storage scales as (2*l_max+1)*n_r**2.

    Orthonormality is not checked or imposed by this component. The Coulomb
    map is well defined for arbitrary orbital inputs; callers implement
    normalization and orbital projection with the radial quadrature metric.
    """

    def __init__(self, radial_grid, l_max, m_max=None, *, angular_grid=None):
        self.l_max = _integer(l_max, "l_max")
        self.m_max = self.l_max if m_max is None else _integer(m_max, "m_max")
        if self.m_max > self.l_max:
            raise ValueError("m_max cannot exceed l_max.")

        r = np.asarray(radial_grid.r)
        weights = np.asarray(radial_grid.weights)
        D2 = np.asarray(radial_grid.D2)
        if (
            r.ndim != 1
            or len(r) < 3
            or np.iscomplexobj(r)
            or not np.all(np.isfinite(r))
            or r[0] != 0
            or np.any(np.diff(r) <= 0)
        ):
            raise ValueError(
                "The radial grid must be real, increasing, and start at zero."
            )
        if (
            weights.shape != r.shape
            or np.iscomplexobj(weights)
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0)
        ):
            raise ValueError("Radial quadrature weights must be real and positive.")
        if (
            D2.shape != (len(r), len(r))
            or np.iscomplexobj(D2)
            or not np.all(np.isfinite(D2))
        ):
            raise ValueError(
                "D2 must be a finite real nodal matrix on the full radial grid."
            )
        self.r = r[1:-1].copy()
        self.radial_weights = weights[1:-1].copy()

        if angular_grid is None:
            angular_grid = GaussLegendreFourierGrid.from_bandlimit(2 * self.l_max)
        if not isinstance(angular_grid, GaussLegendreFourierGrid):
            raise TypeError("angular_grid must be a GaussLegendreFourierGrid.")
        if (
            angular_grid.n_theta < 2 * self.l_max + 1
            or angular_grid.n_phi < 4 * self.m_max + 1
        ):
            raise ValueError(
                "Angular grid is too small for alias-free mean-field construction and action."
            )
        self.angular_grid = angular_grid
        self._orbital_transform = AngularTransform(self.l_max, self.m_max, angular_grid)
        self._field_transform = AngularTransform(
            2 * self.l_max, 2 * self.m_max, angular_grid
        )
        self.orbital_lm = self._orbital_transform.channels
        self.multipole_lm = self._field_transform.channels
        self.orbital_shape = (len(self.r), len(self.orbital_lm))
        self.radial_matrices = radial_Coulomb(radial_grid, 2 * self.l_max + 1)
        self._indices_by_l = tuple(
            np.array(
                [k for k, (degree, _) in enumerate(self.multipole_lm) if degree == L]
            )
            for L in range(2 * self.l_max + 1)
        )

    def build(self, orbitals, *, pairs=None, orbital_m=None):
        r"""Construct W_s^q from phi_q.conj()*phi_s for the requested pairs.

        ``orbitals`` has shape (n_orbitals, n_r, n_lm) and contains u(r_a).
        ``pairs`` is an ordered iterable of unique (q, s) indices. By default
        it contains all q <= s, ordered first by q and then by s. Reverse
        pairs can subsequently be recovered through multipole conjugation.

        To use the polar-only path, supply ``orbital_m`` (one integer per
        orbital, or a scalar shared by all orbitals). Keep the same orbital
        array shape, with exact zeros in all channels of other m. This path
        uses the analytic azimuthal integral and no FFT or azimuthal values.

        Returns a MeanFieldPotential with complex coefficients of shape
        (n_pairs, n_LM, n_r). Pair densities are temporary and are processed
        one requested pair at a time. Full field multipoles are retained
        also for fixed-m inputs, making both paths share an output layout.
        """
        orbitals, _ = _orbitals(orbitals, self.orbital_shape)
        n_orbitals = len(orbitals)
        magnetic = _magnetic_numbers(orbital_m, orbitals, self._orbital_transform)
        if pairs is None:
            pairs = tuple(
                (q, s) for q in range(n_orbitals) for s in range(q, n_orbitals)
            )
        else:
            pairs = tuple(_pair(pair) for pair in pairs)
        if len(set(pairs)) != len(pairs):
            raise ValueError("Requested pairs must be unique.")
        if any(q >= n_orbitals or s >= n_orbitals for q, s in pairs):
            raise ValueError("Pair indices must refer to supplied orbitals.")

        coefficients = np.zeros(
            (len(pairs), len(self.multipole_lm), len(self.r)), dtype=np.complex128
        )
        pair_m = None
        if magnetic is None:
            values = self._orbital_transform.synthesize(orbitals)
            for ip, (q, s) in enumerate(pairs):
                density = self._field_transform.analyze(values[q].conj() * values[s])
                for L, ids in enumerate(self._indices_by_l):
                    coefficients[ip, ids] = (
                        self.radial_matrices[L] @ density[:, ids]
                    ).T
        else:
            values = np.array(
                [
                    self._orbital_transform.synthesize_m(u, m)
                    for u, m in zip(orbitals, magnetic)
                ]
            )
            pair_m = tuple(int(magnetic[s] - magnetic[q]) for q, s in pairs)
            for ip, ((q, s), M) in enumerate(zip(pairs, pair_m)):
                density = self._field_transform.analyze_m(
                    values[q].conj() * values[s], M
                )
                for k in self._field_transform.indices[M]:
                    L, _ = self.multipole_lm[k]
                    coefficients[ip, k] = self.radial_matrices[L] @ density[:, k]
        return MeanFieldPotential(self, pairs, coefficients, pair_m)


class MeanFieldPotential:
    r"""Mean-field multipoles returned by SphericalMeanFieldBuilder.build.

    ``coefficients[ip, k, a]`` is the ordinary nodal value W_qs^(LM)(r_a),
    where ``pairs[ip] == (q, s)`` and ``multipole_lm[k] == (L, M)``.
    These are coefficients of W(r,Omega), with no radial 1/r factor.
    """

    def __init__(self, builder, pairs, coefficients, pair_m):
        self._builder = builder
        self.pairs = pairs
        self.coefficients = coefficients
        self.multipole_lm = builder.multipole_lm
        self._pair_index = {pair: ip for ip, pair in enumerate(pairs)}
        self._pair_m = pair_m
        lookup = {lm: k for k, lm in enumerate(self.multipole_lm)}
        self._reverse_m = np.array([lookup[L, -M] for L, M in self.multipole_lm])
        self._phase = np.array([(-1.0) ** M for _, M in self.multipole_lm])

    def multipoles(self, pair):
        r"""Return (n_LM, n_r) values for a stored pair or its reverse.

        Reverse fields obey W_sq^(LM) = (-1)^M conjugate(W_qs^(L,-M)).
        A stored pair returns a view of ``coefficients``; a reconstructed
        reverse pair returns a new array. Missing pairs raise KeyError.
        """
        q, s = _pair(pair)
        if (q, s) in self._pair_index:
            return self.coefficients[self._pair_index[q, s]]
        if (s, q) in self._pair_index:
            reverse = self.coefficients[self._pair_index[s, q]]
            return self._phase[:, None] * reverse[self._reverse_m].conj()
        raise KeyError(f"Neither pair {(q, s)} nor its reverse was constructed.")

    def apply(self, orbitals, pair, *, orbital_m=None):
        r"""Project W_s^q U onto the builder's retained orbital harmonics.

        Accept nodal reduced radial values of shape (n_r, n_lm) or
        (n_orbitals, n_r, n_lm), and return the same shape. Only one selected
        pair is applied, allowing callers to batch fields and density-matrix
        contractions without materializing every pair/orbital combination.

        For a field built with fixed-m orbitals, supplying the target
        ``orbital_m`` enables polar-only action with output m = m_in + m_s-m_q.
        Components outside the builder's retained channels are projected out.
        Without this metadata the general transform path is used.

        This operation includes angular projection only. It does not apply
        the orbital-space Q projector, density matrices, or time-evolution
        factors. No extra radial quadrature weights or r factors are applied.
        """
        builder = self._builder
        pair = _pair(pair)
        field = self.multipoles(pair)
        orbitals, single = _orbitals(orbitals, builder.orbital_shape, allow_single=True)
        magnetic = _magnetic_numbers(orbital_m, orbitals, builder._orbital_transform)
        if magnetic is not None and self._pair_m is not None:
            if pair in self._pair_index:
                M = self._pair_m[self._pair_index[pair]]
            else:
                M = -self._pair_m[self._pair_index[pair[::-1]]]
            field_values = builder._field_transform.synthesize_m(field.T, M)
            result = np.zeros_like(orbitals)
            for p, (orbital, m) in enumerate(zip(orbitals, magnetic)):
                values = builder._orbital_transform.synthesize_m(orbital, m)
                result[p] = builder._orbital_transform.analyze_m(
                    field_values * values, m + M
                )
        else:
            field_values = builder._field_transform.synthesize(field.T)
            values = builder._orbital_transform.synthesize(orbitals)
            result = builder._orbital_transform.analyze(values * field_values)
        return result[0] if single else result
