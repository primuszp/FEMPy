"""Végeselemes háló és strukturált hálógenerátorok.

A :class:`Mesh` kizárólag geometriát és topológiát tárol. Nem tartalmaz anyagot,
terhelést vagy peremfeltételt; ezek a :class:`fempy.model.Model` részei. Ez a
szétválasztás teszi lehetővé, hogy ugyanazt a hálót több számításhoz használjuk.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .elements import Element2D, Quad4, Triangle3


@dataclass(frozen=True, slots=True)
class Mesh:
    """Csomóponti koordináták és végeselemek megváltoztathatatlan gyűjteménye.

    A Python API minden csomópontindexe nullától indul. A ``nodes`` tömb alakja
    ``(node_count, 2)``, oszlopai ``x`` és ``y``. A konstruktor ellenőrzi az
    indexeket és minden elem geometriáját, majd írásvédetté teszi a koordinátákat.

    Args:
        nodes: Kétdimenziós koordinátatömb vagy beágyazott lista.
        elements: :class:`Triangle3` és/vagy :class:`Quad4` objektumok.
    """

    nodes: NDArray[np.float64]
    elements: tuple[Element2D, ...]
    node_sets: Mapping[str, tuple[int, ...]]
    edge_sets: Mapping[str, tuple[tuple[int, int], ...]]

    def __init__(
        self,
        nodes: ArrayLike,
        elements: Iterable[Element2D],
        *,
        node_sets: Mapping[str, Iterable[int]] | None = None,
        edge_sets: Mapping[str, Iterable[tuple[int, int]]] | None = None,
    ):
        # Saját másolat kell: az írásvédelem így nem módosítja a hívó által
        # átadott eredeti NumPy-tömb ``writeable`` állapotát.
        coordinates = np.array(nodes, dtype=float, copy=True)
        element_tuple = tuple(elements)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("nodes must have shape (node_count, 2)")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("node coordinates must be finite")
        if not element_tuple:
            raise ValueError("a mesh needs at least one element")
        for element in element_tuple:
            if min(element.node_ids) < 0 or max(element.node_ids) >= len(coordinates):
                raise IndexError(f"element contains an invalid node index: {element.node_ids}")
            area = element.area(coordinates[list(element.node_ids)])
            if not np.isfinite(area) or area <= 0.0:
                raise ValueError(f"element has invalid area: {element.node_ids}")
        checked_node_sets = _validate_node_sets(node_sets or {}, len(coordinates))
        checked_edge_sets = _validate_edge_sets(edge_sets or {}, len(coordinates))
        coordinates.setflags(write=False)
        object.__setattr__(self, "nodes", coordinates)
        object.__setattr__(self, "elements", element_tuple)
        object.__setattr__(self, "node_sets", MappingProxyType(checked_node_sets))
        object.__setattr__(self, "edge_sets", MappingProxyType(checked_edge_sets))

    @property
    def node_count(self) -> int:
        """A háló csomópontjainak száma."""
        return len(self.nodes)

    @property
    def element_count(self) -> int:
        """A háló végeselemeinek száma."""
        return len(self.elements)

    def nodes_where(
        self, *, x: float | None = None, y: float | None = None, tolerance: float = 1e-9
    ) -> list[int]:
        """Koordináta alapján csomópontokat keres.

        Ez a segédmetódus teszi olvashatóvá például egy teljes perem befogását:
        ``model.fix_nodes(mesh.nodes_where(x=0.0))``.

        Args:
            x: Előírt x koordináta, vagy ``None``.
            y: Előírt y koordináta, vagy ``None``.
            tolerance: Abszolút lebegőpontos összehasonlítási tűrés.

        Returns:
            A feltételt teljesítő, nullától induló csomópontindexek listája.
        """

        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be non-negative and finite")
        mask = np.ones(self.node_count, dtype=bool)
        if x is not None:
            mask &= np.isclose(self.nodes[:, 0], x, atol=tolerance, rtol=0.0)
        if y is not None:
            mask &= np.isclose(self.nodes[:, 1], y, atol=tolerance, rtol=0.0)
        return np.flatnonzero(mask).tolist()

    @property
    def boundary_names(self) -> tuple[str, ...]:
        """A hálógenerátorból megőrzött, névvel ellátott peremek."""

        return tuple(sorted(set(self.node_sets) | set(self.edge_sets)))

    def boundary_nodes(self, name: str) -> list[int]:
        """Visszaadja egy elnevezett perem csomópontjait.

        A visszaadott lista másolható és módosítható; a hálóban tárolt halmaz
        változatlan marad. Ismeretlen névnél informatív ``KeyError`` keletkezik.
        """

        if name not in self.node_sets:
            raise KeyError(
                f"unknown boundary {name!r}; available: {', '.join(self.boundary_names)}"
            )
        return list(self.node_sets[name])

    def boundary_edges(self, name: str) -> list[tuple[int, int]]:
        """Visszaadja egy elnevezett perem kétcsomópontos éleit."""

        if name not in self.edge_sets:
            raise KeyError(
                f"unknown boundary {name!r}; available: {', '.join(self.boundary_names)}"
            )
        return list(self.edge_sets[name])

    def plot(self, *, ax=None, show_node_ids: bool = False):
        """Matplotlib-ábrán megjeleníti a hálót."""

        from .plotting import plot_mesh

        return plot_mesh(self, ax=ax, show_node_ids=show_node_ids)

    def plot_boundaries(
        self,
        *,
        names=None,
        ax=None,
        show_mesh: bool = True,
        show_labels: bool = True,
    ):
        """A névvel ellátott peremeket külön színekkel jeleníti meg."""

        from .plotting import plot_boundaries

        return plot_boundaries(
            self,
            names=names,
            ax=ax,
            show_mesh=show_mesh,
            show_labels=show_labels,
        )


def _validate_node_sets(
    node_sets: Mapping[str, Iterable[int]], node_count: int
) -> dict[str, tuple[int, ...]]:
    """Ellenőrzi és rendezett, ismétlésmentes tuple-ökké alakítja a halmazokat."""

    checked = {}
    for name, node_ids in node_sets.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("node-set names must be non-empty strings")
        values = tuple(sorted({int(node) for node in node_ids}))
        if any(node < 0 or node >= node_count for node in values):
            raise IndexError(f"node set {name!r} contains an invalid node index")
        checked[name] = values
    return checked


def _validate_edge_sets(
    edge_sets: Mapping[str, Iterable[tuple[int, int]]], node_count: int
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Ellenőrzi a pereméleket, miközben megtartja azok geometriai irányát."""

    checked = {}
    for name, edges in edge_sets.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("edge-set names must be non-empty strings")
        values = tuple(dict.fromkeys((int(edge[0]), int(edge[1])) for edge in edges))
        if any(a == b or min(a, b) < 0 or max(a, b) >= node_count for a, b in values):
            raise IndexError(f"edge set {name!r} contains an invalid edge")
        checked[name] = values
    return checked


def rectangular_quad_mesh(
    nx: int,
    ny: int,
    width: float,
    height: float,
    *,
    origin: tuple[float, float] = (0.0, 0.0),
) -> Mesh:
    """Szabályos, ``nx × ny`` darab Quad4 elemből álló téglalapot készít.

    A csomópontok sorfolytonosan, alulról felfelé jönnek létre. Az elemek
    csomópontsorrendje mindig pozitív, azaz az óramutató járásával ellentétes.
    """

    if nx < 1 or ny < 1 or width <= 0.0 or height <= 0.0:
        raise ValueError("nx, ny, width and height must be positive")
    ox, oy = origin
    nodes = np.array(
        [
            [ox + x, oy + y]
            for y in np.linspace(0.0, height, ny + 1)
            for x in np.linspace(0.0, width, nx + 1)
        ]
    )
    elements = []
    for row in range(ny):
        for column in range(nx):
            lower_left = column + row * (nx + 1)
            elements.append(
                Quad4(
                    (
                        lower_left,
                        lower_left + 1,
                        lower_left + nx + 2,
                        lower_left + nx + 1,
                    )
                )
            )
    return Mesh(nodes, elements)


def rectangular_tri_mesh(
    nx: int,
    ny: int,
    width: float,
    height: float,
    *,
    origin: tuple[float, float] = (0.0, 0.0),
) -> Mesh:
    """Strukturált háromszöghálót készít, cellánként két Triangle3 elemmel.

    Először négyszögháló készül, majd minden cellát az 1–3 átló mentén két
    pozitív orientációjú háromszögre bontunk.
    """

    quad_mesh = rectangular_quad_mesh(nx, ny, width, height, origin=origin)
    triangles = []
    for element in quad_mesh.elements:
        n1, n2, n3, n4 = element.node_ids
        triangles.extend((Triangle3((n1, n2, n3)), Triangle3((n1, n3, n4))))
    return Mesh(quad_mesh.nodes, triangles)
