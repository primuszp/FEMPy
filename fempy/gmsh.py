"""Modern Gmsh Python API-ra épülő opcionális kétdimenziós hálózó.

Nem ír köztes ``.geo`` vagy ``.inp`` fájlt. A geometria, a háló és a fizikai
peremcsoportok közvetlenül memóriában haladnak át a Gmsh és a FEMPy között.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .elements import Quad4, Triangle3, Triangle6
from .geometry import Geometry2D
from .mesh import Mesh


class GmshNotInstalledError(ImportError):
    """Akkor keletkezik, ha a felhasználó Gmsh-hálózást kér Gmsh nélkül."""


@dataclass(frozen=True, slots=True)
class GmshMesher:
    """Háromszög- vagy rekombinált négyszöghálót készít egy ``Geometry2D``-ből.

    Args:
        element_size: Globális cél-elemméret a geometria mértékegységében.
        element_shape: ``"triangle"`` vagy ``"quad"``.
        order: ``1`` lineáris Triangle3/Quad4, ``2`` kvadratikus Triangle6
            hálóhoz. Másodrendű négyszög jelenleg nem támogatott.
        algorithm: Gmsh 2D algoritmuskód; a 6-os a robusztus Frontal-Delaunay.
        optimize: Kérjen-e Gmsh hálóminőség-javítást.
        terminal_output: Jelenjenek-e meg a Gmsh üzenetei a terminálban.
    """

    element_size: float
    element_shape: Literal["triangle", "quad"] = "triangle"
    order: int = 1
    algorithm: int = 6
    optimize: bool = True
    terminal_output: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.element_size) or self.element_size <= 0.0:
            raise ValueError("element_size must be positive and finite")
        if self.element_shape not in ("triangle", "quad"):
            raise ValueError("element_shape must be 'triangle' or 'quad'")
        if self.order not in (1, 2):
            raise ValueError("Gmsh element order must be 1 or 2")
        if self.order == 2 and self.element_shape != "triangle":
            raise ValueError("second-order Gmsh meshing currently supports triangles only")

    def generate(self, geometry: Geometry2D) -> Mesh:
        """Behálózza a geometriát és megőrzi annak elnevezett peremeit."""

        geometry.validate()
        gmsh = _import_gmsh()
        initialized_here = not bool(gmsh.isInitialized())
        if initialized_here:
            gmsh.initialize()
        model_added = False
        try:
            gmsh.option.setNumber("General.Terminal", int(self.terminal_output))
            gmsh.model.add(geometry.name)
            model_added = True
            surface_tag, curve_groups = self._build_geometry(gmsh, geometry)
            self._configure(gmsh, surface_tag)
            gmsh.model.mesh.generate(2)
            if self.order == 2:
                gmsh.model.mesh.setOrder(2)
            if self.optimize:
                gmsh.model.mesh.optimize()
            return self._read_mesh(gmsh, surface_tag, curve_groups)
        finally:
            if model_added:
                gmsh.model.remove()
            if initialized_here:
                gmsh.finalize()

    def _build_geometry(self, gmsh: Any, geometry: Geometry2D):
        """A független Geometry2D objektumokat Gmsh geo-entitásokká alakítja."""

        point_tags = [
            gmsh.model.geo.addPoint(point.x, point.y, 0.0, self.element_size)
            for point in geometry.points
        ]
        curve_tags = []
        curve_groups: dict[str, list[int]] = {}
        boundary_points: dict[str, set[int]] = {}
        for curve in geometry.curves:
            start, end = point_tags[curve.start], point_tags[curve.end]
            if curve.center is None:
                tag = gmsh.model.geo.addLine(start, end)
            else:
                tag = gmsh.model.geo.addCircleArc(start, point_tags[curve.center], end)
            curve_tags.append(tag)
            curve_groups.setdefault(curve.boundary, []).append(tag)
            boundary_points.setdefault(curve.boundary, set()).update((start, end))

        loop_tags = [
            gmsh.model.geo.addCurveLoop([curve_tags[index] for index in loop.curves])
            for loop in geometry.loops
        ]
        tagged_loops = list(zip(loop_tags, geometry.loops, strict=True))
        outer = next(tag for tag, loop in tagged_loops if not loop.hole)
        holes = [tag for tag, loop in tagged_loops if loop.hole]
        surface_tag = gmsh.model.geo.addPlaneSurface([outer, *holes])
        gmsh.model.geo.synchronize()

        domain_group = gmsh.model.addPhysicalGroup(2, [surface_tag])
        gmsh.model.setPhysicalName(2, domain_group, "domain")
        for name, tags in curve_groups.items():
            physical_tag = gmsh.model.addPhysicalGroup(1, tags)
            gmsh.model.setPhysicalName(1, physical_tag, name)
        for name, size in geometry.boundary_sizes.items():
            gmsh.model.mesh.setSize([(0, tag) for tag in boundary_points.get(name, ())], size)
        return surface_tag, curve_groups

    def _configure(self, gmsh: Any, surface_tag: int) -> None:
        """A hálózási algoritmust és a háromszög/négyszög módot állítja be."""

        gmsh.option.setNumber("Mesh.Algorithm", self.algorithm)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.element_size)
        gmsh.option.setNumber("Mesh.ElementOrder", self.order)
        if self.element_shape == "quad":
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
            gmsh.model.mesh.setRecombine(2, surface_tag)

    def _read_mesh(self, gmsh: Any, surface_tag: int, curve_groups) -> Mesh:
        """A Gmsh címkéit tömör, nullától induló FEMPy indexekké alakítja."""

        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        xyz = np.asarray(coordinates, dtype=float).reshape((-1, 3))
        coordinate_by_tag = {int(tag): xyz[index, :2] for index, tag in enumerate(node_tags)}

        element_types, _, element_node_tags = gmsh.model.mesh.getElements(2, surface_tag)
        raw_elements: list[tuple[str, tuple[int, ...]]] = []
        for element_type, flat_tags in zip(element_types, element_node_tags, strict=True):
            element_type = int(element_type)
            if element_type == 2:
                shape, width = "triangle3", 3
            elif element_type == 9:
                shape, width = "triangle6", 6
            elif element_type == 3:
                shape, width = "quad4", 4
            else:
                raise ValueError(f"unsupported Gmsh 2D element type: {element_type}")
            rows = np.asarray(flat_tags, dtype=np.int64).reshape((-1, width))
            raw_elements.extend((shape, tuple(map(int, row))) for row in rows)
        if not raw_elements:
            raise ValueError("Gmsh did not generate any supported 2D elements")

        used_tags = sorted({tag for _, tags in raw_elements for tag in tags})
        index_of = {tag: index for index, tag in enumerate(used_tags)}
        nodes = np.asarray([coordinate_by_tag[tag] for tag in used_tags])
        elements = []
        for shape, tags in raw_elements:
            indices = tuple(index_of[tag] for tag in tags)
            indices = _positive_orientation(indices, nodes, shape)
            element_type = {"triangle3": Triangle3, "triangle6": Triangle6, "quad4": Quad4}[shape]
            elements.append(element_type(indices))

        node_sets: dict[str, list[int]] = {}
        edge_sets: dict[str, list[tuple[int, ...]]] = {}
        for name, tags in curve_groups.items():
            boundary_node_tags: set[int] = set()
            boundary_edges: list[tuple[int, ...]] = []
            for curve_tag in tags:
                types, _, connectivity = gmsh.model.mesh.getElements(1, curve_tag)
                for element_type, flat_tags in zip(types, connectivity, strict=True):
                    element_type = int(element_type)
                    if element_type == 1:
                        width = 2
                    elif element_type == 8:
                        width = 3
                    else:
                        raise ValueError(f"unsupported Gmsh boundary element type: {element_type}")
                    rows = np.asarray(flat_tags, dtype=np.int64).reshape((-1, width))
                    for row in rows:
                        tags_on_edge = tuple(map(int, row))
                        if width == 2:
                            ordered_tags = tags_on_edge
                        else:
                            # A Gmsh Line3 sorrendje (első, utolsó, középső).
                            ordered_tags = (tags_on_edge[0], tags_on_edge[2], tags_on_edge[1])
                        boundary_node_tags.update(ordered_tags)
                        if all(tag in index_of for tag in ordered_tags):
                            boundary_edges.append(tuple(index_of[tag] for tag in ordered_tags))
            node_sets[name] = [index_of[tag] for tag in boundary_node_tags if tag in index_of]
            edge_sets[name] = boundary_edges
        return Mesh(nodes, elements, node_sets=node_sets, edge_sets=edge_sets)


def _positive_orientation(
    indices: tuple[int, ...], nodes: np.ndarray, shape: str
) -> tuple[int, ...]:
    """Megfordítja az óramutató járása szerinti Gmsh-kapcsolatot."""

    corner_count = 4 if shape == "quad4" else 3
    polygon = nodes[list(indices[:corner_count])]
    signed_twice_area = np.sum(
        polygon[:, 0] * np.roll(polygon[:, 1], -1) - polygon[:, 1] * np.roll(polygon[:, 0], -1)
    )
    if signed_twice_area > 0.0:
        return indices
    if shape == "triangle6":
        # (1,2,3,12,23,31) -> (1,3,2,31,23,12)
        return (indices[0], indices[2], indices[1], indices[5], indices[4], indices[3])
    return (indices[0], *reversed(indices[1:]))


def _import_gmsh():
    """Késleltetve importál, hogy a FEMPy alapfunkciói Gmsh nélkül is fussanak."""

    try:
        import gmsh
    except (ImportError, OSError) as error:
        raise GmshNotInstalledError(
            "Gmsh meshing requires the optional dependency; install it with "
            "'python -m pip install -e .[gmsh]'"
        ) from error
    return gmsh
