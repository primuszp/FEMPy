"""Általános matematikai önellenőrzések a támogatott végeselemekhez.

Az ellenőrzések nem egy konkrét szerkezeti példát vizsgálnak, hanem az
izoparametrikus elem alapazonosságait: partícióegységet, zérus gradiensösszeget,
numerikus deriváltegyezést és az identitás-geometria pontos leképezését.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .elements import Quad4, Triangle3, Triangle6


@dataclass(frozen=True, slots=True)
class ElementCheckReport:
    """Egy elem önellenőrzésének megváltoztathatatlan hibamutatói."""

    element: str
    sample_count: int
    partition_error: float
    gradient_balance_error: float
    derivative_error: float
    mapping_error: float
    tolerance: float

    @property
    def passed(self) -> bool:
        """Igaz, ha minden ellenőrzött azonosság a tolerancián belül van."""

        return self.maximum_error <= self.tolerance

    @property
    def maximum_error(self) -> float:
        """A négy független ellenőrzés legnagyobb abszolút hibája."""

        return max(
            self.partition_error,
            self.gradient_balance_error,
            self.derivative_error,
            self.mapping_error,
        )

    def summary(self) -> str:
        """Tömör, naplózható szöveges összefoglaló."""

        status = "PASS" if self.passed else "FAIL"
        return (
            f"{self.element}: {status}, samples={self.sample_count}, "
            f"max_error={self.maximum_error:.3e}, tolerance={self.tolerance:.3e}"
        )

    def raise_for_failure(self) -> None:
        """Sikertelen ellenőrzésnél részletes ``ValueError`` kivételt dob."""

        if not self.passed:
            raise ValueError(self.summary())


def _element_definition(element):
    """Alakfüggvényt, deriváltat, referencia-csomópontokat és tartományt ad."""

    if isinstance(element, Triangle3):

        def shape(xi, eta):
            return np.array((1.0 - xi - eta, xi, eta))

        def gradient(_xi, _eta):
            return np.array(((-1.0, 1.0, 0.0), (-1.0, 0.0, 1.0)))

        nodes = np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
        return shape, gradient, nodes, "triangle"
    if isinstance(element, Triangle6):
        nodes = np.array(
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (0.5, 0.0),
                (0.5, 0.5),
                (0.0, 0.5),
            )
        )
        return element._shape, element._natural_gradient, nodes, "triangle"
    if isinstance(element, Quad4):
        nodes = np.array(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)))
        return element._shape, element._natural_gradient, nodes, "quad"
    raise TypeError(f"unsupported element type: {type(element).__name__}")


def verify_element(
    element,
    *,
    sample_count: int = 20,
    tolerance: float = 1e-8,
    seed: int = 1,
) -> ElementCheckReport:
    """Numerikusan ellenőrzi egy T3, T6 vagy Q4 elem alapazonosságait.

    Args:
        element: Tetszőleges csomópontindexű ``Triangle3``, ``Triangle6`` vagy
            ``Quad4`` objektum; kizárólag az elemtípus matematikája számít.
        sample_count: Véletlen belső mintapontok száma.
        tolerance: Minden abszolút hibamutató felső korlátja.
        seed: Determinisztikus véletlengenerátor-mag.
    """

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    shape, gradient, reference_nodes, domain = _element_definition(element)
    rng = np.random.default_rng(seed)
    if domain == "triangle":
        points = rng.random((sample_count, 2))
        outside = points.sum(axis=1) > 1.0
        points[outside] = 1.0 - points[outside]
    else:
        points = rng.uniform(-0.9, 0.9, size=(sample_count, 2))

    partition_error = 0.0
    gradient_balance_error = 0.0
    derivative_error = 0.0
    mapping_error = 0.0
    step = 1e-6
    for xi, eta in points:
        values = np.asarray(shape(float(xi), float(eta)), dtype=float)
        derivatives = np.asarray(gradient(float(xi), float(eta)), dtype=float)
        numerical = np.vstack(
            (
                (shape(xi + step, eta) - shape(xi - step, eta)) / (2.0 * step),
                (shape(xi, eta + step) - shape(xi, eta - step)) / (2.0 * step),
            )
        )
        partition_error = max(partition_error, abs(float(values.sum()) - 1.0))
        gradient_balance_error = max(
            gradient_balance_error, float(np.max(np.abs(derivatives.sum(axis=1))))
        )
        derivative_error = max(derivative_error, float(np.max(np.abs(derivatives - numerical))))
        mapping_error = max(
            mapping_error,
            float(np.max(np.abs(values @ reference_nodes - np.array((xi, eta))))),
        )
    return ElementCheckReport(
        element=type(element).__name__,
        sample_count=sample_count,
        partition_error=partition_error,
        gradient_balance_error=gradient_balance_error,
        derivative_error=derivative_error,
        mapping_error=mapping_error,
        tolerance=tolerance,
    )


def verify_supported_elements(**kwargs) -> tuple[ElementCheckReport, ...]:
    """Egyetlen hívással ellenőrzi a könyvtár mindhárom publikus elemét."""

    return (
        verify_element(Triangle3((0, 1, 2)), **kwargs),
        verify_element(Triangle6((0, 1, 2, 3, 4, 5)), **kwargs),
        verify_element(Quad4((0, 1, 2, 3)), **kwargs),
    )
