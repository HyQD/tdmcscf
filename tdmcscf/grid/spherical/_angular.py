"""Separated angular transforms on a Gauss-Legendre-Fourier grid."""

import numpy as np

from grid_lib.spherical_coordinates.angular_momentum import sph_harm_y


class AngularTransform:
    """Internal transform with the M-major channel order used by grid-lib."""

    def __init__(self, l_max, m_max, grid):
        self.grid = grid
        self.channels = tuple(
            (l, m) for m in range(-m_max, m_max + 1) for l in range(abs(m), l_max + 1)
        )
        self.indices = {
            m: np.array([i for i, (_, mi) in enumerate(self.channels) if mi == m])
            for m in range(-m_max, m_max + 1)
        }
        self.polar = {
            m: np.column_stack(
                [sph_harm_y(self.channels[i][0], m, grid.theta, 0.0) for i in ids]
            )
            for m, ids in self.indices.items()
        }

    def synthesize(self, coefficients):
        """Map (..., n_lm) coefficients to (..., n_theta, n_phi) values."""
        fourier = np.zeros(
            coefficients.shape[:-1] + self.grid.shape, dtype=np.complex128
        )
        for m, ids in self.indices.items():
            fourier[..., m % self.grid.n_phi] = self.grid.n_phi * np.einsum(
                "...l,jl->...j", coefficients[..., ids], self.polar[m]
            )
        return np.fft.ifft(fourier, axis=-1)

    def analyze(self, values):
        """Project (..., n_theta, n_phi) values onto the retained channels."""
        fourier = np.fft.fft(values, axis=-1) / self.grid.n_phi
        result = np.zeros(
            values.shape[:-2] + (len(self.channels),), dtype=np.complex128
        )
        for m, ids in self.indices.items():
            result[..., ids] = self._polar_analysis(
                fourier[..., m % self.grid.n_phi], m
            )
        return result

    def synthesize_m(self, coefficients, m):
        """Evaluate the polar part of a field with azimuthal factor exp(i*m*phi)."""
        return np.einsum(
            "...l,jl->...j", coefficients[..., self.indices[m]], self.polar[m]
        )

    def analyze_m(self, values, m):
        """Project a polar field with known m; absent output channels are zero."""
        result = np.zeros(
            values.shape[:-1] + (len(self.channels),), dtype=np.complex128
        )
        if m in self.indices:
            result[..., self.indices[m]] = self._polar_analysis(values, m)
        return result

    def _polar_analysis(self, values, m):
        return (
            2
            * np.pi
            * np.einsum(
                "...j,jl,j->...l",
                values,
                self.polar[m].conj(),
                self.grid.theta_weights,
            )
        )
