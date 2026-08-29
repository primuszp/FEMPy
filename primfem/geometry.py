"""Egyszerű, hálózótól független kétdimenziós geometrialeírás.

A modul szándékosan kevés építőelemet használ: pontot, egyenes szakaszt,
körívet és zárt hurkot. Ez elegendő sok oktatási síkbeli feladathoz, miközben
a :mod:`primfem.gmsh` adapternek nem kell végeselemes fogalmakat ismernie.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class GeometryPoint:
    """Egy síkbeli geometriai pont."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class GeometryCurve:
    """Egyenes vagy körív két végponttal és opcionális középponttal."""

    start: int
    end: int
    boundary: str
    center: int | None = None


@dataclass(frozen=True, slots=True)
class GeometryLoop:
    """Egymást követő görbékből álló zárt kontúr."""

    curves: tuple[int, ...]
    hole: bool


@dataclass(frozen=True, slots=True)
class LineSegment2D:
    """Felhasználói egyenes szakasz egy általános zárt hurokhoz."""

    start: tuple[float, float]
    end: tuple[float, float]
    boundary: str = "outer"


@dataclass(frozen=True, slots=True)
class CircularArc2D:
    """Felhasználói körív; a start–center–end pontsorrend adja az irányát."""

    start: tuple[float, float]
    center: tuple[float, float]
    end: tuple[float, float]
    boundary: str = "outer"


BoundarySegment2D: TypeAlias = LineSegment2D | CircularArc2D


@dataclass(slots=True)
class Geometry2D:
    """Láncolható API egy külső kontúr és tetszőleges lyukak leírására.

    A peremnevek már a geometria létrehozásakor bekerülnek a modellbe. A Gmsh
    ezeket fizikai görbecsoportként őrzi meg, majd a :class:`primfem.mesh.Mesh`
    név szerinti csomópont- és élhalmazként kapja vissza.
    """

    name: str = "geometry"
    points: list[GeometryPoint] = field(default_factory=list, init=False)
    curves: list[GeometryCurve] = field(default_factory=list, init=False)
    loops: list[GeometryLoop] = field(default_factory=list, init=False)
    boundary_sizes: dict[str, float] = field(default_factory=dict, init=False)

    def add_polygon(
        self,
        points: Iterable[tuple[float, float]],
        *,
        boundary: str = "outer",
        boundary_names: Sequence[str] | None = None,
        hole: bool = False,
        mesh_size: float | None = None,
    ) -> Geometry2D:
        """Sokszög alakú külső kontúrt vagy lyukat ad a geometriához.

        ``boundary_names`` segítségével minden oldal külön nevet kaphat. Ha ez
        nincs megadva, az összes oldal a ``boundary`` csoportba kerül.
        A záró kezdőpontot nem kell megismételni.
        """

        vertices = [(float(x), float(y)) for x, y in points]
        if len(vertices) > 1 and vertices[0] == vertices[-1]:
            vertices.pop()
        if len(vertices) < 3 or len(set(vertices)) != len(vertices):
            raise ValueError("a polygon needs at least three different points")
        self._require_loop_slot(hole)
        names = list(boundary_names) if boundary_names is not None else [boundary] * len(vertices)
        if len(names) != len(vertices):
            raise ValueError("boundary_names must contain one name for each polygon side")
        self._validate_boundary_names(names)

        point_ids = [self._add_point(vertex) for vertex in vertices]
        curve_ids = []
        for index, start in enumerate(point_ids):
            curve_ids.append(
                self._add_curve(start, point_ids[(index + 1) % len(point_ids)], names[index])
            )
        self.loops.append(GeometryLoop(tuple(curve_ids), hole))
        if mesh_size is not None:
            for name in set(names):
                self.set_boundary_size(name, mesh_size)
        return self

    def add_loop(
        self,
        segments: Iterable[BoundarySegment2D],
        *,
        hole: bool = False,
        mesh_size: float | None = None,
    ) -> Geometry2D:
        """Egyenesekből és körívekből álló általános zárt hurkot ad hozzá.

        A szegmenseket folytonos geometriai sorrendben kell megadni: minden
        szegmens végpontja a következő kezdőpontja, az utolsóé pedig az elsőé.
        A 180 fokos vagy nagyobb köríveket több kisebb ívre kell bontani.
        """

        segment_list = list(segments)
        if len(segment_list) < 2:
            raise ValueError("a boundary loop needs at least two segments")
        self._require_loop_slot(hole)
        names = [segment.boundary for segment in segment_list]
        self._validate_boundary_names(names)
        for current, following in zip(
            segment_list, segment_list[1:] + segment_list[:1], strict=True
        ):
            if tuple(current.end) != tuple(following.start):
                raise ValueError("boundary-loop segments must form a continuous closed chain")

        curve_ids = []
        for segment in segment_list:
            start = self._find_or_add_point(segment.start)
            end = self._find_or_add_point(segment.end)
            center = None
            if isinstance(segment, CircularArc2D):
                center = self._find_or_add_point(segment.center)
                start_radius = _distance(segment.start, segment.center)
                end_radius = _distance(segment.end, segment.center)
                tolerance = max(start_radius, end_radius, 1.0) * 1e-10
                if start_radius <= tolerance or abs(start_radius - end_radius) > tolerance:
                    raise ValueError(
                        "a circular arc needs distinct endpoints at equal center distance"
                    )
            curve_ids.append(self._add_curve(start, end, segment.boundary, center))
        self.loops.append(GeometryLoop(tuple(curve_ids), hole))
        if mesh_size is not None:
            for name in set(names):
                self.set_boundary_size(name, mesh_size)
        return self

    def add_rectangle(
        self,
        width: float,
        height: float,
        *,
        origin: tuple[float, float] = (0.0, 0.0),
        boundary_names: Sequence[str] = ("bottom", "right", "top", "left"),
        mesh_size: float | None = None,
    ) -> Geometry2D:
        """Tengelyekkel párhuzamos külső téglalapot hoz létre."""

        if width <= 0.0 or height <= 0.0:
            raise ValueError("rectangle width and height must be positive")
        ox, oy = origin
        return self.add_polygon(
            [(ox, oy), (ox + width, oy), (ox + width, oy + height), (ox, oy + height)],
            boundary_names=boundary_names,
            mesh_size=mesh_size,
        )

    def add_circle(
        self,
        center: tuple[float, float],
        radius: float,
        *,
        boundary: str = "hole",
        hole: bool = True,
        mesh_size: float | None = None,
    ) -> Geometry2D:
        """Négy 90 fokos körívből álló kört ad hozzá.

        A négy ív megbízhatóbban működik különböző Gmsh-verziókkal, mint egy
        teljes, egyetlen görbéből álló kör. ``hole=False`` esetén a kör lehet
        a modell külső kontúrja is.
        """

        if not isfinite(radius) or radius <= 0.0:
            raise ValueError("circle radius must be positive and finite")
        self._validate_boundary_names([boundary])
        self._require_loop_slot(hole)
        cx, cy = map(float, center)
        center_id = self._add_point((cx, cy))
        rim = [
            self._add_point((cx + radius, cy)),
            self._add_point((cx, cy + radius)),
            self._add_point((cx - radius, cy)),
            self._add_point((cx, cy - radius)),
        ]
        curve_ids = [
            self._add_curve(rim[index], rim[(index + 1) % 4], boundary, center_id)
            for index in range(4)
        ]
        self.loops.append(GeometryLoop(tuple(curve_ids), hole))
        if mesh_size is not None:
            self.set_boundary_size(boundary, mesh_size)
        return self

    def set_boundary_size(self, boundary: str, size: float) -> Geometry2D:
        """Helyi cél-elemméretet rendel egy elnevezett peremhez."""

        self._validate_boundary_names([boundary])
        if not isfinite(size) or size <= 0.0:
            raise ValueError("boundary mesh size must be positive and finite")
        self.boundary_sizes[boundary] = float(size)
        return self

    @property
    def boundary_names(self) -> tuple[str, ...]:
        """A geometriában szereplő peremcsoportok rendezett nevei."""

        return tuple(sorted({curve.boundary for curve in self.curves}))

    def validate(self) -> None:
        """Ellenőrzi, hogy van pontosan egy külső, zárt kontúr."""

        outer_count = sum(not loop.hole for loop in self.loops)
        if outer_count != 1:
            raise ValueError("Geometry2D needs exactly one outer loop")

    def _add_point(self, coordinates: tuple[float, float]) -> int:
        x, y = coordinates
        if not isfinite(x) or not isfinite(y):
            raise ValueError("geometry coordinates must be finite")
        self.points.append(GeometryPoint(x, y))
        return len(self.points) - 1

    def _find_or_add_point(self, coordinates: tuple[float, float]) -> int:
        """Közös szegmensvégpontot közös topológiai pontként őriz meg."""

        x, y = map(float, coordinates)
        for index, point in enumerate(self.points):
            if point.x == x and point.y == y:
                return index
        return self._add_point((x, y))

    def _add_curve(self, start: int, end: int, boundary: str, center: int | None = None) -> int:
        self.curves.append(GeometryCurve(start, end, boundary, center))
        return len(self.curves) - 1

    def _require_loop_slot(self, hole: bool) -> None:
        if not hole and any(not loop.hole for loop in self.loops):
            raise ValueError("Geometry2D can contain only one outer loop")
        if hole and not any(not loop.hole for loop in self.loops):
            raise ValueError("add the outer loop before adding holes")

    @staticmethod
    def _validate_boundary_names(names: Iterable[str]) -> None:
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("boundary names must be non-empty strings")


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Két geometriai pont euklideszi távolsága külső függőség nélkül."""

    dx = float(first[0]) - float(second[0])
    dy = float(first[1]) - float(second[1])
    return (dx * dx + dy * dy) ** 0.5
