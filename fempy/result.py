"""Megoldási eredmények és kényelmes utófeldolgozó API.

Az :class:`AnalysisResult` a számítás pillanatfelvétele. Közvetlenül tartalmazza
a csomóponti elmozdulást és reakciót, az elemhez kötött részletes eredményeket,
valamint a solver diagnosztikáját. A csomóponti feszültségmezők igény szerint,
az elemek extrapolált értékeinek átlagolásával készülnek.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .model import Model
    from .solver import SolverInfo


def principal_values(
    stress: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Kiszámítja a két síkbeli főfeszültséget és az első főirányt.

    A szög a globális x tengelytől az óramutató járásával ellentétesen, radiánban
    értendő. A bemenet egyetlen ``[sx, sy, txy]`` sor vagy több soros tömb lehet.
    """

    values = np.asarray(stress, dtype=float)
    sx = values[..., 0]
    sy = values[..., 1]
    txy = values[..., 2]
    mean = 0.5 * (sx + sy)
    radius = np.sqrt((0.5 * (sx - sy)) ** 2 + txy**2)
    principals = np.stack((mean + radius, mean - radius), axis=-1)
    angle = 0.5 * np.arctan2(2.0 * txy, sx - sy)
    return principals, angle


@dataclass(frozen=True, slots=True)
class IntegrationPointResult:
    """Egy elem egyetlen integrációs pontjának teljes eredménye.

    A naturális koordináta mellett tárolja az alakváltozást, feszültséget,
    von Mises-értéket, főfeszültségeket és az első főirány szögét.
    """

    natural_coordinates: tuple[float, float]
    strain: NDArray[np.float64]
    stress: NDArray[np.float64]
    von_mises: float
    principal_stress: NDArray[np.float64]
    principal_angle: float


@dataclass(frozen=True, slots=True)
class ElementResult:
    """Egy elem középponti, Gauss-ponti és extrapolált eredményei.

    A ``strain`` és ``stress`` középponti érték. Az ``integration_points`` a
    Quad4 esetén négy, Triangle3 esetén egy, Triangle6 esetén hét rekord. A
    ``nodal_*`` tömb még elemenkénti extrapolált érték; a hálószintű átlagolás
    később történik.
    """

    strain: NDArray[np.float64]
    stress: NDArray[np.float64]
    von_mises: float
    principal_stress: NDArray[np.float64]
    principal_angle: float
    integration_points: tuple[IntegrationPointResult, ...]
    nodal_strain: NDArray[np.float64]
    nodal_stress: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Egy megoldott modell megváltoztathatatlan, magas szintű eredmény-API-ja."""

    model: Model
    displacement: NDArray[np.float64]
    reaction: NDArray[np.float64]
    element_results: tuple[ElementResult, ...]
    solver_info: SolverInfo

    @property
    def displacement_magnitude(self) -> NDArray[np.float64]:
        """Csomópontonkénti ``sqrt(ux² + uy²)`` elmozdulásnagyság."""
        return np.linalg.norm(self.displacement, axis=1)

    @property
    def strain(self) -> NDArray[np.float64]:
        """Elemközépi ``[ex, ey, gamma_xy]`` alakváltozások."""
        return np.vstack([result.strain for result in self.element_results])

    @property
    def stress(self) -> NDArray[np.float64]:
        """Elemközépi ``[sigma_x, sigma_y, tau_xy]`` feszültségek."""
        return np.vstack([result.stress for result in self.element_results])

    @property
    def von_mises(self) -> NDArray[np.float64]:
        """Elemközépi von Mises-egyenértékfeszültségek."""
        return np.array([result.von_mises for result in self.element_results])

    @property
    def principal_stress(self) -> NDArray[np.float64]:
        """Elemközépi első és második síkbeli főfeszültség."""

        return np.vstack([result.principal_stress for result in self.element_results])

    @property
    def principal_angle(self) -> NDArray[np.float64]:
        """Elemközépi első főirány a globális x tengelytől, radiánban."""

        return np.array([result.principal_angle for result in self.element_results])

    @property
    def integration_point_strain(self) -> tuple[NDArray[np.float64], ...]:
        """Elemenkénti Gauss-ponti alakváltozástömbök.

        Tuple szükséges, mert különböző elemtípusok integrációspont-száma eltér.
        """
        return tuple(
            np.vstack([point.strain for point in result.integration_points])
            for result in self.element_results
        )

    @property
    def integration_point_stress(self) -> tuple[NDArray[np.float64], ...]:
        """Elemenkénti Gauss-ponti feszültségtömbök."""
        return tuple(
            np.vstack([point.stress for point in result.integration_points])
            for result in self.element_results
        )

    @property
    def integration_point_von_mises(self) -> tuple[NDArray[np.float64], ...]:
        """Elemenkénti Gauss-ponti von Mises-értékek."""
        return tuple(
            np.array([point.von_mises for point in result.integration_points])
            for result in self.element_results
        )

    def _average_extrapolated(self, attribute: str) -> NDArray[np.float64]:
        """A közös csomópontokra extrapolált elemi értékeket átlagolja."""

        sample = np.asarray(getattr(self.element_results[0], attribute))
        totals = np.zeros((self.model.mesh.node_count,) + sample.shape[1:], dtype=float)
        counts = np.zeros(self.model.mesh.node_count, dtype=int)
        # Minden elem a saját lokális csomópontjaira ad egy értéket. Ezeket a
        # globális csomópontnál összegezzük, a counts pedig a kapcsolódó elemeket
        # számolja. Ez felel meg a szokásos egyszerű nodal averaging eljárásnak.
        for element, element_result in zip(
            self.model.mesh.elements, self.element_results, strict=True
        ):
            node_ids = np.asarray(element.node_ids)
            totals[node_ids] += getattr(element_result, attribute)
            counts[node_ids] += 1
        if np.any(counts == 0):
            raise ValueError("cannot average results: the mesh contains an unused node")
        divisor = counts.reshape((-1,) + (1,) * (totals.ndim - 1))
        return totals / divisor

    @property
    def nodal_strain(self) -> NDArray[np.float64]:
        """Gauss-pontból extrapolált, hálócsomópontokon átlagolt alakváltozás."""

        return self._average_extrapolated("nodal_strain")

    @property
    def nodal_stress(self) -> NDArray[np.float64]:
        """Gauss-pontból extrapolált, hálócsomópontokon átlagolt feszültség."""

        return self._average_extrapolated("nodal_stress")

    @property
    def nodal_von_mises(self) -> NDArray[np.float64]:
        """Az átlagolt csomóponti feszültségkomponensekből számított von Mises."""
        return np.array(
            [
                self.model.material.von_mises(stress, self.model.condition)
                for stress in self.nodal_stress
            ]
        )

    @property
    def nodal_principal_stress(self) -> NDArray[np.float64]:
        """Az átlagolt csomóponti feszültség két főértéke."""
        return principal_values(self.nodal_stress)[0]

    @property
    def nodal_principal_angle(self) -> NDArray[np.float64]:
        """Az első csomóponti főirány szöge radiánban."""
        return principal_values(self.nodal_stress)[1]

    @property
    def nodal_principal_vectors(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Az első és második főfeszültség nagyságával skálázott irányvektorok."""

        angle = self.nodal_principal_angle
        principal = self.nodal_principal_stress
        first = np.column_stack((np.cos(angle), np.sin(angle))) * principal[:, [0]]
        second = np.column_stack((-np.sin(angle), np.cos(angle))) * principal[:, [1]]
        return first, second

    def displaced_nodes(self, scale: float = 1.0) -> NDArray[np.float64]:
        """A ``x_deformed = x + scale*u`` alakváltozott koordinátákat adja."""

        if not np.isfinite(scale):
            raise ValueError("deformation scale must be finite")
        return self.model.mesh.nodes + scale * self.displacement

    def suggested_deformation_scale(self, fraction: float = 0.08) -> float:
        """A geometriához illeszkedő, jól látható deformációskálát javasol.

        A ``fraction=0.08`` azt jelenti, hogy a legnagyobb kirajzolt elmozdulás
        a befoglaló téglalap legnagyobb méretének körülbelül nyolc százaléka.
        A fizikai számítás eredményét ez nem módosítja, csak a megjelenítést.
        """

        if not np.isfinite(fraction) or fraction <= 0.0:
            raise ValueError("fraction must be positive and finite")
        span = np.ptp(self.model.mesh.nodes, axis=0).max()
        maximum_displacement = self.displacement_magnitude.max()
        if span <= 0.0 or maximum_displacement <= np.finfo(float).eps:
            return 1.0
        return float(fraction * span / maximum_displacement)

    def summary(self) -> str:
        """Rövid, ember által olvasható egysoros eredményösszefoglaló."""

        return (
            f"Analysis '{self.model.name}': "
            f"{self.model.mesh.node_count} nodes, {self.model.mesh.element_count} elements, "
            f"max |u|={self.displacement_magnitude.max():.6g}, "
            f"max nodal von Mises={self.nodal_von_mises.max():.6g}, "
            f"solver={self.solver_info.method.value}, nnz={self.solver_info.nonzero_entries}"
        )

    def write_vtk(self, path: str | Path) -> Path:
        """A teljes hálót és eredménymezőket ParaView-kompatibilis VTK-ba írja."""

        from .io import write_vtk

        return write_vtk(self, path)

    def write(self, path: str | Path) -> Path:
        """Eredményt ír a kiterjesztés által választott meshio-formátumba.

        Például ``result.write("result.vtu")`` tömör XML-VTK fájlt készít,
        míg a meglévő :meth:`write_vtk` továbbra is függőségmentes, olvasható
        legacy VTK kimenetet ad.
        """

        from .meshio_adapter import write_result

        return write_result(self, path)

    def plot(
        self,
        *,
        scale: float = 1.0,
        field: str = "von_mises",
        cmap: str | None = None,
        show_undeformed: bool = True,
        ax=None,
        style=None,
    ):
        """Deformált hálót rajzol választott elem- vagy csomóponti színezéssel."""

        from .plotting import plot_result

        return plot_result(
            self,
            scale=scale,
            field=field,
            cmap=cmap,
            show_undeformed=show_undeformed,
            ax=ax,
            style=style,
        )

    def plot_principal_directions(
        self, *, scale: float = 1.0, stride: int = 1, ax=None, style=None
    ):
        """Színezett első főfeszültségi iránynyilakat rajzol."""

        from .plotting import plot_principal_directions

        return plot_principal_directions(self, scale=scale, stride=stride, ax=ax, style=style)
