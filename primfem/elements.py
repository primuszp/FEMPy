"""Kétdimenziós kontinuum végeselemek.

A modul három oktatási szempontból fontos elemet valósít meg:

* :class:`Triangle3`: háromcsomópontos, állandó alakváltozású háromszög (CST);
* :class:`Triangle6`: hatcsomópontos, kvadratikus izoparametrikus háromszög (T6);
* :class:`Quad4`: négycsomópontos bilineáris négyszög 2×2 Gauss-integrálással.

Mindhárom elem ugyanazt a kis interfészt követi. Az elem csak lokális műveleteket
végez: ``B`` mátrix, elemi merevség, tömegmátrix és eredménykinyerés. A globális
összeállítás a :mod:`primfem.model` feladata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class Element2D(Protocol):
    """A minden támogatott kétdimenziós elem által teljesített interfész.

    A ``Protocol`` nem ősosztály: típusellenőrzési szerződés. Új elem úgy
    illeszthető a solverhez, hogy ugyanezeket a metódusokat valósítja meg.
    """

    node_ids: tuple[int, ...]
    vtk_cell_type: int

    def area(self, coordinates: FloatArray) -> float:
        """Az elem fizikai területe."""
        ...

    def stiffness(
        self, coordinates: FloatArray, constitutive: FloatArray, thickness: float
    ) -> FloatArray:
        """Az elem ``Ke = integral(B.T @ D @ B)`` merevségi mátrixa."""
        ...

    def strain_at_center(self, coordinates: FloatArray, displacement: FloatArray) -> FloatArray:
        """Alakváltozás az elem geometriai/naturális középpontjában."""
        ...

    def integration_data(
        self, coordinates: FloatArray
    ) -> tuple[tuple[tuple[float, float], FloatArray, float], ...]:
        """Integrációs pontok: naturális hely, ``B`` mátrix és súly."""
        ...

    def extrapolate_to_nodes(self, integration_values: FloatArray) -> FloatArray:
        """Gauss-ponti értékek extrapolálása az elem csomópontjaira."""
        ...

    def mass_matrix(self, coordinates: FloatArray, density: float, thickness: float) -> FloatArray:
        """Konzisztens elemi transzlációs tömegmátrix."""
        ...


def _strain_matrix(gradient: FloatArray) -> FloatArray:
    """Az alakfüggvény-gradiensekből felépíti a ``B`` mátrixot.

    A szabadságfokok sorrendje ``[u1, v1, u2, v2, ...]``. A visszakapott
    vektor ``[epsilon_x, epsilon_y, gamma_xy]``; a harmadik komponens mérnöki
    nyírási alakváltozás.
    """

    count = gradient.shape[1]
    matrix = np.zeros((3, 2 * count), dtype=float)
    matrix[0, 0::2] = gradient[0]
    matrix[1, 1::2] = gradient[1]
    matrix[2, 0::2] = gradient[1]
    matrix[2, 1::2] = gradient[0]
    return matrix


@dataclass(frozen=True, slots=True)
class Triangle3:
    """Háromcsomópontos, állandó alakváltozású háromszög (CST).

    A csomópontokat az óramutató járásával ellentétes sorrendben kell megadni.
    A lineáris alakfüggvények gradiense és így a ``B`` mátrix az egész elemen
    állandó, ezért egyetlen súlyponti integráció pontos.
    """

    node_ids: tuple[int, int, int]
    vtk_cell_type: int = 5

    def __post_init__(self) -> None:
        if len(set(self.node_ids)) != 3:
            raise ValueError("Triangle3 needs three different nodes")

    @staticmethod
    def _gradient(coordinates: FloatArray) -> tuple[FloatArray, float]:
        """Alakfüggvény-gradienst és területet számít a csomópontokból."""

        x1, y1 = coordinates[0]
        x2, y2 = coordinates[1]
        x3, y3 = coordinates[2]
        twice_area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        # A twice_area előjele egyben az elem orientációja. Nulla degenerált,
        # negatív óramutató járásával megegyező csomópontsorrendet jelent.
        if twice_area <= 0.0:
            raise ValueError("Triangle3 nodes must be counter-clockwise and non-collinear")
        gradient = (
            np.array(
                [[y2 - y3, y3 - y1, y1 - y2], [x3 - x2, x1 - x3, x2 - x1]],
                dtype=float,
            )
            / twice_area
        )
        return gradient, twice_area / 2.0

    def area(self, coordinates: FloatArray) -> float:
        """A pozitív háromszögterület; hibát jelez fordított elemnél."""
        return self._gradient(coordinates)[1]

    def stiffness(
        self, coordinates: FloatArray, constitutive: FloatArray, thickness: float
    ) -> FloatArray:
        """Kiszámítja a 6×6-os CST elemi merevségi mátrixot."""
        gradient, area = self._gradient(coordinates)
        b_matrix = _strain_matrix(gradient)
        # Mivel B és D állandó, az integrál egyszerűen terület * vastagság.
        return thickness * area * (b_matrix.T @ constitutive @ b_matrix)

    def strain_at_center(self, coordinates: FloatArray, displacement: FloatArray) -> FloatArray:
        """Visszaadja a CST egész elemre érvényes állandó alakváltozását."""
        gradient, _ = self._gradient(coordinates)
        return _strain_matrix(gradient) @ displacement

    def integration_data(
        self, coordinates: FloatArray
    ) -> tuple[tuple[tuple[float, float], FloatArray, float], ...]:
        """A súlyponti szabályt adja: hely, B mátrix és fizikai súly."""

        gradient, area = self._gradient(coordinates)
        return (((1.0 / 3.0, 1.0 / 3.0), _strain_matrix(gradient), area),)

    def extrapolate_to_nodes(self, integration_values: FloatArray) -> FloatArray:
        """A CST field is constant, so its centroid value belongs to all nodes."""

        values = np.asarray(integration_values, dtype=float)
        if values.shape[0] != 1:
            raise ValueError("Triangle3 expects one integration-point value")
        return np.repeat(values, 3, axis=0)

    def mass_matrix(self, coordinates: FloatArray, density: float, thickness: float) -> FloatArray:
        """A konzisztens transzlációs tömegmátrixot adja vissza."""

        total_mass = density * thickness * self.area(coordinates)
        scalar = total_mass / 12.0 * np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
        return np.kron(scalar, np.eye(2))


@dataclass(frozen=True, slots=True)
class Triangle6:
    """Hatcsomópontos, kvadratikus izoparametrikus háromszög (T6).

    A lokális csomópontsorrend ``(1, 2, 3, 12, 23, 31)``: előbb a három
    sarokcsomópont következik pozitív körüljárással, majd rendre az 1–2, 2–3
    és 3–1 oldal középcsomópontja. Ez megegyezik a Gmsh másodrendű
    háromszögének sorrendjével.

    A merevség és a konzisztens tömeg integrálása a hétpontos, ötödrendű
    Dunavant-szabállyal történik. A szabály a kvadratikus alakfüggvényekből
    képzett tömegintegrandushoz is elegendő pontosságú, és görbült oldalaknál
    is robusztusabb a minimális hárompontos formulánál.
    """

    node_ids: tuple[int, int, int, int, int, int]
    vtk_cell_type: int = 22  # VTK_QUADRATIC_TRIANGLE

    def __post_init__(self) -> None:
        if len(set(self.node_ids)) != 6:
            raise ValueError("Triangle6 needs six different nodes")

    @staticmethod
    def _shape(xi: float, eta: float) -> FloatArray:
        """A hat kvadratikus alakfüggvény értéke a referenciaháromszögben."""

        l1 = 1.0 - xi - eta
        l2 = xi
        l3 = eta
        return np.array(
            [
                l1 * (2.0 * l1 - 1.0),
                l2 * (2.0 * l2 - 1.0),
                l3 * (2.0 * l3 - 1.0),
                4.0 * l1 * l2,
                4.0 * l2 * l3,
                4.0 * l3 * l1,
            ],
            dtype=float,
        )

    @staticmethod
    def _natural_gradient(xi: float, eta: float) -> FloatArray:
        """Az alakfüggvények ``xi`` és ``eta`` szerinti deriváltjai."""

        l1 = 1.0 - xi - eta
        l2 = xi
        l3 = eta
        return np.array(
            [
                [-(4.0 * l1 - 1.0), 4.0 * l2 - 1.0, 0.0, 4.0 * (l1 - l2), 4.0 * l3, -4.0 * l3],
                [-(4.0 * l1 - 1.0), 0.0, 4.0 * l3 - 1.0, -4.0 * l2, 4.0 * l2, 4.0 * (l1 - l3)],
            ],
            dtype=float,
        )

    @classmethod
    def _b_matrix(cls, coordinates: FloatArray, xi: float, eta: float) -> tuple[FloatArray, float]:
        """Fizikai ``B`` mátrixot és pozitív Jacobi-determinánst számít."""

        natural_gradient = cls._natural_gradient(xi, eta)
        jacobian = natural_gradient @ coordinates
        determinant = float(np.linalg.det(jacobian))
        if determinant <= 0.0:
            raise ValueError("Triangle6 has an inverted or degenerate Jacobian")
        spatial_gradient = np.linalg.solve(jacobian, natural_gradient)
        return _strain_matrix(spatial_gradient), determinant

    @staticmethod
    def _quadrature() -> tuple[tuple[float, float, float], ...]:
        """A referenciaháromszög hétpontos Dunavant-integrálási szabálya."""

        first_a, first_b = 0.059715871789770, 0.470142064105115
        second_a, second_b = 0.797426985353087, 0.101286507323456
        return (
            (1.0 / 3.0, 1.0 / 3.0, 0.1125),
            (first_a, first_b, 0.066197076394253),
            (first_b, first_a, 0.066197076394253),
            (first_b, first_b, 0.066197076394253),
            (second_a, second_b, 0.062969590272414),
            (second_b, second_a, 0.062969590272414),
            (second_b, second_b, 0.062969590272414),
        )

    def area(self, coordinates: FloatArray) -> float:
        """A görbült T6 elem integrált területe, teljes Jacobian-ellenőrzéssel."""

        return float(sum(weight for _, _, weight in self.integration_data(coordinates)))

    def stiffness(
        self, coordinates: FloatArray, constitutive: FloatArray, thickness: float
    ) -> FloatArray:
        """Kiszámítja a 12×12-es T6 elemi merevségi mátrixot."""

        matrix = np.zeros((12, 12), dtype=float)
        for _, b_matrix, weight in self.integration_data(coordinates):
            matrix += thickness * weight * (b_matrix.T @ constitutive @ b_matrix)
        return matrix

    def strain_at_center(self, coordinates: FloatArray, displacement: FloatArray) -> FloatArray:
        """Alakváltozás a háromszög súlypontjában."""

        b_matrix, _ = self._b_matrix(coordinates, 1.0 / 3.0, 1.0 / 3.0)
        return b_matrix @ displacement

    def integration_data(
        self, coordinates: FloatArray
    ) -> tuple[tuple[tuple[float, float], FloatArray, float], ...]:
        """Hét rekordot ad: naturális hely, ``B`` mátrix és fizikai súly."""

        data = []
        for xi, eta, reference_weight in self._quadrature():
            b_matrix, determinant = self._b_matrix(coordinates, xi, eta)
            data.append(((xi, eta), b_matrix, determinant * reference_weight))
        return tuple(data)

    def extrapolate_to_nodes(self, integration_values: FloatArray) -> FloatArray:
        """A hét Gauss-pont értékeire kvadratikus mezőt illeszt a hat csomópontra."""

        values = np.asarray(integration_values, dtype=float)
        if values.shape[0] != 7:
            raise ValueError("Triangle6 expects seven integration-point values")
        interpolation = np.vstack([self._shape(xi, eta) for xi, eta, _ in self._quadrature()])
        nodal_values, *_ = np.linalg.lstsq(interpolation, values, rcond=None)
        return nodal_values

    def mass_matrix(self, coordinates: FloatArray, density: float, thickness: float) -> FloatArray:
        """Konzisztens 12×12-es tömegmátrixot integrál a görbült geometrián."""

        matrix = np.zeros((12, 12), dtype=float)
        for (xi, eta), _, weight in self.integration_data(coordinates):
            shape = self._shape(xi, eta)
            matrix += density * thickness * weight * np.kron(np.outer(shape, shape), np.eye(2))
        return matrix


@dataclass(frozen=True, slots=True)
class Quad4:
    """Négycsomópontos bilineáris izoparametrikus négyszög.

    Az alakfüggvények a ``[-1, 1] × [-1, 1]`` naturális tartományban élnek.
    A merevségi integrálhoz négy, ``±1/sqrt(3)`` koordinátájú Gauss-pontot
    használunk. Ez a bilineáris elem szokásos teljes integrálása.
    """

    node_ids: tuple[int, int, int, int]
    vtk_cell_type: int = 9

    def __post_init__(self) -> None:
        if len(set(self.node_ids)) != 4:
            raise ValueError("Quad4 needs four different nodes")

    @staticmethod
    def _natural_gradient(xi: float, eta: float) -> FloatArray:
        """A négy bilineáris alakfüggvény deriváltja xi és eta szerint."""
        return 0.25 * np.array(
            [
                [-(1.0 - eta), 1.0 - eta, 1.0 + eta, -(1.0 + eta)],
                [-(1.0 - xi), -(1.0 + xi), 1.0 + xi, 1.0 - xi],
            ],
            dtype=float,
        )

    @classmethod
    def _b_matrix(cls, coordinates: FloatArray, xi: float, eta: float) -> tuple[FloatArray, float]:
        """Fizikai ``B`` mátrixot és Jacobi-determinánst számít egy pontban."""

        natural_gradient = cls._natural_gradient(xi, eta)
        # J a naturális és fizikai koordináták közti lokális leképezés.
        jacobian = natural_gradient @ coordinates
        determinant = float(np.linalg.det(jacobian))
        if determinant <= 0.0:
            raise ValueError("Quad4 has an inverted or degenerate Jacobian")
        # solve(J, dN_natural) numerikusan stabilabb, mint inv(J) @ dN.
        spatial_gradient = np.linalg.solve(jacobian, natural_gradient)
        return _strain_matrix(spatial_gradient), determinant

    def area(self, coordinates: FloatArray) -> float:
        """Terület, minden Gauss-pontban pozitív Jacobi-determinánssal.

        Az integráció közben a fordított, összecsukódott és túlzottan konkáv
        elemek is korán kiderülnek, már a :class:`Mesh` létrehozásakor.
        """

        return float(sum(weight for _, _, weight in self.integration_data(coordinates)))

    def stiffness(
        self, coordinates: FloatArray, constitutive: FloatArray, thickness: float
    ) -> FloatArray:
        """Kiszámítja a 8×8-as Quad4 elemi merevségi mátrixot."""
        matrix = np.zeros((8, 8), dtype=float)
        for _, b_matrix, weight in self.integration_data(coordinates):
            matrix += thickness * weight * (b_matrix.T @ constitutive @ b_matrix)
        return matrix

    def strain_at_center(self, coordinates: FloatArray, displacement: FloatArray) -> FloatArray:
        """Alakváltozás a ``xi=eta=0`` naturális középpontban."""
        b_matrix, _ = self._b_matrix(coordinates, 0.0, 0.0)
        return b_matrix @ displacement

    def integration_data(
        self, coordinates: FloatArray
    ) -> tuple[tuple[tuple[float, float], FloatArray, float], ...]:
        """A négy 2×2 Gauss-pontot adja vissza körüljárási sorrendben.

        Minden rekord tartalma: ``((xi, eta), B, detJ)``. A Gauss-súly mind a
        négy pontnál 1, ezért a fizikai integrációs súly egyszerűen ``detJ``.
        """

        gauss = 1.0 / np.sqrt(3.0)
        points = ((-gauss, -gauss), (gauss, -gauss), (gauss, gauss), (-gauss, gauss))
        data = []
        for xi, eta in points:
            b_matrix, determinant = self._b_matrix(coordinates, xi, eta)
            data.append(((xi, eta), b_matrix, determinant))
        return tuple(data)

    @staticmethod
    def _shape(xi: float, eta: float) -> FloatArray:
        """A négy bilineáris alakfüggvény értéke egy naturális pontban."""
        return 0.25 * np.array(
            [
                (1.0 - xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 + eta),
                (1.0 - xi) * (1.0 + eta),
            ]
        )

    def extrapolate_to_nodes(self, integration_values: FloatArray) -> FloatArray:
        """A négy Gauss-pont értékeit a négy elemcsomópontra extrapolálja."""

        values = np.asarray(integration_values, dtype=float)
        if values.shape[0] != 4:
            raise ValueError("Quad4 expects four integration-point values")
        gauss = 1.0 / np.sqrt(3.0)
        points = ((-gauss, -gauss), (gauss, -gauss), (gauss, gauss), (-gauss, gauss))
        # interpolation @ nodal_values = gauss_values, ezért az extrapolált
        # csomóponti érték a kis 4×4 rendszer megoldása.
        interpolation = np.vstack([self._shape(xi, eta) for xi, eta in points])
        return np.linalg.solve(interpolation, values)

    def mass_matrix(self, coordinates: FloatArray, density: float, thickness: float) -> FloatArray:
        """Konzisztens tömegmátrixot számít 2×2 Gauss-integrálással."""

        matrix = np.zeros((8, 8), dtype=float)
        for (xi, eta), _, weight in self.integration_data(coordinates):
            shape = self._shape(xi, eta)
            matrix += density * thickness * weight * np.kron(np.outer(shape, shape), np.eye(2))
        return matrix
