"""Terhelési esetek egy közös végeselemes modellen.

A :class:`LoadCase` ugyanazt a jól olvasható terhelési és peremfeltétel-API-t
adja, mint a :class:`primfem.model.Model`, de a hálót, anyagot és vastagságot nem
másolja. Így egy geometriai modellhez több egymástól független statikai eset
definiálható, majd közös merevségi mátrixszal oldható meg.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .model import Model
    from .result import AnalysisResult
    from .solver import SolverMethod, SolverOptions


class LoadCase:
    """Egy modell önálló terhelés- és peremfeltétel-készlete.

    A példányt általában a :meth:`Model.load_case` metódus készíti. Az
    ``inherit=True`` alapérték lemásolja a modellen már megadott támaszokat,
    terheket és testgyorsulást. A későbbi módosítások egyik irányban sem
    szivárognak át, ezért a terhelési esetek biztonságosan függetlenek.
    """

    __slots__ = ("_definition", "model", "name")

    def __init__(self, model: Model, name: str, *, inherit: bool = True) -> None:
        from .model import Model

        if not isinstance(name, str) or not name.strip():
            raise ValueError("load-case name must be a non-empty string")
        self.model = model
        self.name = name.strip()
        self._definition = Model(
            model.mesh,
            model.material,
            thickness=model.thickness,
            condition=model.condition,
            name=f"{model.name} — {self.name}",
        )
        if inherit:
            self._definition._loads[:] = model._loads
            self._definition._prescribed.update(model._prescribed)
            self._definition._body_acceleration[:] = model._body_acceleration

    @property
    def mesh(self):
        """A közös, megváltoztathatatlan háló."""

        return self.model.mesh

    @property
    def force_vector(self) -> np.ndarray:
        """A terhelési eset koncentrált és integrált csomóponti erői."""

        return self._definition.force_vector

    @property
    def prescribed_displacements(self) -> np.ndarray:
        """A terhelési eset előírt elmozdulásainak áttekintő tömbje."""

        return self._definition.prescribed_displacements

    def add_nodal_load(self, node: int, *, fx: float = 0.0, fy: float = 0.0) -> LoadCase:
        self._definition.add_nodal_load(node, fx=fx, fy=fy)
        return self

    def add_nodal_loads(self, nodes: list[int], *, fx: float = 0.0, fy: float = 0.0) -> LoadCase:
        self._definition.add_nodal_loads(nodes, fx=fx, fy=fy)
        return self

    def set_body_acceleration(self, *, ax: float = 0.0, ay: float = 0.0) -> LoadCase:
        self._definition.set_body_acceleration(ax=ax, ay=ay)
        return self

    def prescribe(
        self,
        node: int,
        *,
        ux: float | None = None,
        uy: float | None = None,
    ) -> LoadCase:
        self._definition.prescribe(node, ux=ux, uy=uy)
        return self

    def fix_node(self, node: int, *, x: bool = True, y: bool = True) -> LoadCase:
        self._definition.fix_node(node, x=x, y=y)
        return self

    def fix_nodes(self, nodes: list[int], *, x: bool = True, y: bool = True) -> LoadCase:
        self._definition.fix_nodes(nodes, x=x, y=y)
        return self

    def fix_boundary(self, name: str, *, x: bool = True, y: bool = True) -> LoadCase:
        self._definition.fix_boundary(name, x=x, y=y)
        return self

    def prescribe_boundary(
        self,
        name: str,
        *,
        ux: float | None = None,
        uy: float | None = None,
    ) -> LoadCase:
        self._definition.prescribe_boundary(name, ux=ux, uy=uy)
        return self

    def add_boundary_traction(
        self,
        name: str,
        *,
        tx: float = 0.0,
        ty: float = 0.0,
    ) -> LoadCase:
        self._definition.add_boundary_traction(name, tx=tx, ty=ty)
        return self

    def add_boundary_pressure(self, name: str, pressure: float) -> LoadCase:
        self._definition.add_boundary_pressure(name, pressure)
        return self

    def plot_boundary_conditions(self, **kwargs):
        """Kirajzolja kizárólag ennek az esetnek a támaszait és terheit."""

        return self._definition.plot_boundary_conditions(**kwargs)

    def solve(
        self,
        solver: SolverOptions | SolverMethod | str | None = None,
    ) -> AnalysisResult:
        """Önállóan megoldja ezt az esetet faktorizáció-megosztás nélkül."""

        return self._definition.solve(solver)
