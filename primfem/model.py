"""A végeselemes modell felépítése, összeállítása és megoldása.

Ez a csomag központi modulja. A :class:`Model` összekapcsolja a hálót, az
anyagot, a vastagságot, a terheléseket és a peremfeltételeket. A ``solve()``
folyamata röviden:

1. a kötött és szabad szabadságfokok szétválasztása;
2. kizárólag a redukált ``K_ff`` ritka mátrix összeállítása;
3. a ``K_ff u_f = f_f - K_fc u_c`` egyenlet megoldása;
4. a teljes elmozdulásvektor és a reakcióerők visszaállítása;
5. Gauss-ponti, elemközépi és csomóponti eredmények képzése.

A normál megoldási út soha nem készít sűrű globális mátrixot. A teljes CSR
merevségi mátrix csak a ``stiffness_matrix()`` explicit kérésére épül fel.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .boundary import edge_quadrature, element_boundary_edges
from .elements import Element2D
from .material import LinearElasticMaterial, PlaneCondition
from .mesh import Mesh
from .result import (
    AnalysisResult,
    ElementResult,
    IntegrationPointResult,
    principal_values,
)
from .solver import (
    SolverInfo,
    SolverMethod,
    SolverOptions,
    SparseDirectFactorization,
    selected_solver_method,
    solve_factorized_system,
    solve_sparse_system,
)

if TYPE_CHECKING:
    from .loadcase import LoadCase


def _element_dofs(node_ids: np.ndarray) -> np.ndarray:
    """Csomópontindexekből az egymásba fűzött ``ux, uy`` indexek."""

    return np.column_stack((2 * node_ids, 2 * node_ids + 1)).ravel()


@dataclass(slots=True)
class Model:
    """Kétdimenziós, lineárisan rugalmas statikai végeselemes modell.

    A módosító metódusok visszaadják a modellt, ezért láncolhatók. Minden
    csomópontnak két szabadságfoka van, ``u_x`` és ``u_y``; a globális sorrend
    ``[u1, v1, u2, v2, ...]``. A csomópontindexek nullától indulnak.

    Args:
        mesh: Ellenőrzött :class:`Mesh` objektum.
        material: Egyetlen izotróp lineáris anyag.
        thickness: A kétdimenziós tartomány síkra merőleges vastagsága.
        condition: Síkfeszültségi vagy síkalakváltozási feltevés.
        name: Az elemzés és az ábrák megnevezése.
    """

    mesh: Mesh
    material: LinearElasticMaterial
    thickness: float = 1.0
    condition: PlaneCondition = PlaneCondition.STRESS
    name: str = "model"
    _loads: np.ndarray = field(init=False, repr=False)
    _prescribed: dict[int, float] = field(init=False, repr=False, default_factory=dict)
    _body_acceleration: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.condition = PlaneCondition(self.condition)
        if not np.isfinite(self.thickness) or self.thickness <= 0.0:
            raise ValueError("thickness must be positive and finite")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("model name must be a non-empty string")
        self._loads = np.zeros(2 * self.mesh.node_count, dtype=float)
        self._body_acceleration = np.zeros(2, dtype=float)

    def add_nodal_load(self, node: int, *, fx: float = 0.0, fy: float = 0.0) -> Model:
        """Koncentrált erőt ad egy csomóponthoz.

        A művelet összeadó jellegű: ugyanazon csomópont többszöri terhelése
        összegeződik. Visszaadja a modellt, ezért más API-hívással láncolható.
        """

        self._validate_node(node)
        if not np.isfinite(fx) or not np.isfinite(fy):
            raise ValueError("nodal load components must be finite")
        self._loads[2 * node : 2 * node + 2] += (fx, fy)
        return self

    def add_nodal_loads(
        self,
        nodes: Iterable[int],
        *,
        fx: float = 0.0,
        fy: float = 0.0,
    ) -> Model:
        """Ugyanazt az erővektort több csomóponthoz adja.

        Fontos: ``fx`` és ``fy`` csomópontonkénti erő. Egy ``F`` eredő terhet
        ``n`` csomóponton egyenletesen elosztva ``F/n`` értékkel kell megadni.
        """

        for node in nodes:
            self.add_nodal_load(node, fx=fx, fy=fy)
        return self

    def set_body_acceleration(self, *, ax: float = 0.0, ay: float = 0.0) -> Model:
        """Egyenletes testgyorsulást állít be, például ``ay=-9.81`` gravitációt.

        A tényleges csomóponti erőket konzisztens elemi tömegmátrix számítja.
        Hatása csak nem nulla anyagsűrűség mellett van.
        """

        if not np.isfinite(ax) or not np.isfinite(ay):
            raise ValueError("body acceleration components must be finite")
        self._body_acceleration[:] = (ax, ay)
        return self

    def prescribe(self, node: int, *, ux: float | None = None, uy: float | None = None) -> Model:
        """Egy vagy két elmozduláskomponenst ír elő egy csomóponton.

        A ``None`` szabad komponenst jelent. Nem nulla érték kényszerített
        elmozdulás, például támaszsüllyedés vagy illesztési elmozdulás lehet.
        """

        self._validate_node(node)
        if ux is None and uy is None:
            raise ValueError("at least one of ux or uy must be provided")
        if ux is not None and not np.isfinite(ux):
            raise ValueError("prescribed ux must be finite")
        if uy is not None and not np.isfinite(uy):
            raise ValueError("prescribed uy must be finite")
        if ux is not None:
            self._prescribed[2 * node] = float(ux)
        if uy is not None:
            self._prescribed[2 * node + 1] = float(uy)
        return self

    def fix_node(self, node: int, *, x: bool = True, y: bool = True) -> Model:
        """Nulla elmozdulást ír elő a kiválasztott irányokban."""

        return self.prescribe(node, ux=0.0 if x else None, uy=0.0 if y else None)

    def fix_nodes(self, nodes: Iterable[int], *, x: bool = True, y: bool = True) -> Model:
        """Több csomópont kiválasztott irányait egyszerre rögzíti."""

        for node in nodes:
            self.fix_node(node, x=x, y=y)
        return self

    def fix_boundary(self, name: str, *, x: bool = True, y: bool = True) -> Model:
        """Egy hálózáskor elnevezett teljes peremet rögzít.

        Ez a geometriai névhez kötött változat hálófinomítástól független:
        ugyanaz a ``"left"`` hívás működik akkor is, ha a peremen később több
        csomópont keletkezik.
        """

        return self.prescribe_boundary(
            name,
            ux=0.0 if x else None,
            uy=0.0 if y else None,
        )

    def prescribe_boundary(
        self,
        name: str,
        *,
        ux: float | None = None,
        uy: float | None = None,
    ) -> Model:
        """Állandó elmozdulást ír elő egy teljes, elnevezett peremen.

        ``None`` szabad irányt jelent. Ez a metódus támaszsüllyedést vagy
        vezérelt elmozdulást is kezel; a :meth:`fix_boundary` ennek nulla
        értékű, kényelmes rövidítése.
        """

        for node in self.mesh.boundary_nodes(name):
            self.prescribe(node, ux=ux, uy=uy)
        return self

    def add_boundary_traction(self, name: str, *, tx: float = 0.0, ty: float = 0.0) -> Model:
        """Állandó felületi megoszló terhet integrál egy elnevezett peremen.

        ``tx`` és ``ty`` erő/felület dimenziójú érték. Lineáris élen a szokásos
        ``traction * thickness * L/2`` végponti erő adódik. T6 élen hárompontos
        Gauss-integrálás kezeli a kvadratikus alakfüggvényeket és a görbült
        geometriát. A közös csomópontok hozzájárulásai automatikusan összeadódnak.
        """

        if not np.isfinite(tx) or not np.isfinite(ty):
            raise ValueError("boundary traction components must be finite")
        edges = self.mesh.boundary_edges(name)
        if not edges:
            raise ValueError(f"boundary {name!r} has no edges")
        traction = np.asarray([tx, ty], dtype=float)
        for edge in edges:
            for _point, tangent, shape, weight in edge_quadrature(edge, self.mesh.nodes):
                jacobian = float(np.linalg.norm(tangent))
                if jacobian <= np.finfo(float).eps:
                    raise ValueError(f"boundary {name!r} contains a degenerate edge")
                for node, shape_value in zip(edge, shape, strict=True):
                    self._loads[2 * node : 2 * node + 2] += (
                        traction * self.thickness * shape_value * jacobian * weight
                    )
        return self

    def add_boundary_force(self, name: str, *, fx: float = 0.0, fy: float = 0.0) -> Model:
        """Teljes eredő erőt oszt el konzisztensen egy elnevezett peremen.

        Az ``fx`` és ``fy`` a teljes peremre ható eredő erő komponensei,
        nem csomópontonkénti értékek. A metódus a perem hosszából és a
        modell vastagságából képezi az egyenletes traction értéket, majd a
        meglévő konzisztens élintegrálást használja. Emiatt az eredő nem
        változik a perem hálójának finomításakor.
        """

        if not np.isfinite(fx) or not np.isfinite(fy):
            raise ValueError("boundary force components must be finite")
        loaded_area = self.mesh.boundary_length(name) * self.thickness
        if loaded_area <= np.finfo(float).eps:
            raise ValueError(f"boundary {name!r} has zero loaded area")
        return self.add_boundary_traction(name, tx=fx / loaded_area, ty=fy / loaded_area)

    def add_boundary_pressure(self, name: str, pressure: float) -> Model:
        """Peremre merőleges, pozitív értéknél befelé ható nyomást ad meg.

        Az irányt minden peremélnél az egyetlen szomszédos elem középpontjából
        határozza meg, ezért külső kontúron és furat peremén is helyes. A
        nyomást élhosszal és vastagsággal integrált csomóponti erőkké alakítja.
        """

        if not np.isfinite(pressure):
            raise ValueError("pressure must be finite")
        adjacent_elements: dict[tuple[int, int], list] = {}
        for element in self.mesh.elements:
            for edge in element_boundary_edges(element):
                key = tuple(sorted((edge[0], edge[-1])))
                adjacent_elements.setdefault(key, []).append(element)
        for edge in self.mesh.boundary_edges(name):
            adjacent = adjacent_elements.get(tuple(sorted((edge[0], edge[-1]))), [])
            if len(adjacent) != 1:
                raise ValueError(f"boundary edge {edge} must have exactly one adjacent element")
            centroid = self.mesh.nodes[list(adjacent[0].node_ids)].mean(axis=0)
            for point, tangent, shape, weight in edge_quadrature(edge, self.mesh.nodes):
                jacobian = float(np.linalg.norm(tangent))
                if jacobian <= np.finfo(float).eps:
                    raise ValueError(f"boundary {name!r} contains a degenerate edge")
                outward = np.array([tangent[1], -tangent[0]], dtype=float) / jacobian
                if np.dot(outward, point - centroid) < 0.0:
                    outward *= -1.0
                # A pozitív mérnöki nyomás a test belseje felé hat.
                for node, shape_value in zip(edge, shape, strict=True):
                    self._loads[2 * node : 2 * node + 2] += (
                        -pressure * outward * self.thickness * shape_value * jacobian * weight
                    )
        return self

    @property
    def force_vector(self) -> np.ndarray:
        """A koncentrált külső csomóponti erővektor biztonságos másolata.

        A testgyorsulásból származó erő nincs benne; az csak megoldáskor kerül
        hozzáadásra, mert az anyag sűrűségétől és az elemek területétől függ.
        """

        return self._loads.copy()

    @property
    def prescribed_displacements(self) -> np.ndarray:
        """A csomóponti előírások ``(node_count, 2)`` alakú áttekintő tömbje.

        Az oszlopok ``u_x`` és ``u_y``. A ``nan`` szabad irányt, a nulla fix
        irányt, a más véges érték pedig előírt nem nulla elmozdulást jelent.
        A tulajdonság mindig új tömböt ad vissza, így biztonságosan módosítható.
        """

        values = np.full((self.mesh.node_count, 2), np.nan, dtype=float)
        for degree_of_freedom, value in self._prescribed.items():
            values[degree_of_freedom // 2, degree_of_freedom % 2] = value
        return values

    def plot_boundary_conditions(
        self,
        *,
        ax=None,
        show_mesh: bool = True,
        show_loads: bool = True,
        style=None,
    ):
        """Kirajzolja az x-, y- és kétirányú megtámasztásokat és a terheket."""

        from .plotting import plot_boundary_conditions

        return plot_boundary_conditions(
            self,
            ax=ax,
            show_mesh=show_mesh,
            show_loads=show_loads,
            style=style,
        )

    def load_case(self, name: str, *, inherit: bool = True) -> LoadCase:
        """Új, a hálót és anyagot megosztó terhelési esetet készít.

        Args:
            name: Az eset egyedi, ember által olvasható neve.
            inherit: Ha igaz, a modellen már megadott támaszokat, terheket és
                testgyorsulást kezdeti állapotként átmásolja.

        A visszaadott eset ezután függetlenül módosítható. Több eset hatékony
        közös megoldásához a :meth:`solve_cases` metódus használható.
        """

        from .loadcase import LoadCase

        return LoadCase(self, name, inherit=inherit)

    def solve_cases(
        self,
        cases: Iterable[LoadCase],
        solver: SolverOptions | SolverMethod | str | None = None,
        *,
        reuse_factorization: bool = True,
    ) -> dict[str, AnalysisResult]:
        """Több terhelési esetet old meg közös ritka merevségi mátrixszal.

        Az azonos kötött szabadságfokokat használó eseteket egy csoportba
        rendezi. Csoportonként a ``K_ff`` csak egyszer épül fel; direkt solver
        esetén az LU-faktorizáció is egyszer készül el, majd minden új jobb
        oldal újrafelhasználja. Eltérő támaszkészlet automatikusan külön csoport.

        Returns:
            A bemeneti sorrendet megtartó ``{case_name: AnalysisResult}`` szótár.
        """

        case_list, groups = self._group_load_cases(cases)
        options = self._solver_options(solver)

        solved: dict[str, AnalysisResult] = {}
        for group in groups:
            first = group[0]
            _, free, free_map, _ = first._dof_partition()
            matrix = first._assemble_reduced_stiffness(free, free_map)
            method = selected_solver_method(options, len(free))
            factorization = None
            if len(free) and reuse_factorization and method is SolverMethod.DIRECT:
                factorization = SparseDirectFactorization(matrix)

            for index, case in enumerate(group):
                _, case_free, case_free_map, prescribed = case._dof_partition()
                # A csoportosítás miatt ezek értéke azonos; az ellenőrzés a
                # későbbi refaktorálások ellen is védi a faktorizációt.
                if not np.array_equal(case_free, free) or not np.array_equal(
                    case_free_map, free_map
                ):
                    raise RuntimeError("incompatible load cases were grouped together")
                solved[case.name] = case._solve_reduced(
                    matrix,
                    free,
                    free_map,
                    prescribed,
                    options,
                    factorization=factorization,
                    factorization_reused=index > 0,
                )
        return {case.name: solved[case.name] for case in case_list}

    def _group_load_cases(
        self,
        cases: Iterable[LoadCase],
    ) -> tuple[tuple[LoadCase, ...], tuple[list[LoadCase], ...]]:
        """Ellenőrzi és a kötött szabadságfokok szerint csoportosítja az eseteket."""

        from .loadcase import LoadCase

        case_list = tuple(cases)
        if not case_list:
            raise ValueError("at least one load case is required")
        if any(not isinstance(case, LoadCase) for case in case_list):
            raise TypeError("solve_cases accepts LoadCase objects")
        if any(case.model is not self for case in case_list):
            raise ValueError("every load case must belong to this model")
        names = [case.name for case in case_list]
        if len(set(names)) != len(names):
            raise ValueError("load-case names must be unique")

        grouped: dict[tuple[int, ...], list[LoadCase]] = {}
        for case in case_list:
            if not case._prescribed:
                raise ValueError(f"load case {case.name!r} has no displacement boundary conditions")
            grouped.setdefault(tuple(sorted(case._prescribed)), []).append(case)
        return case_list, tuple(grouped.values())

    def solve(self, solver: SolverOptions | SolverMethod | str | None = None) -> AnalysisResult:
        """Összeállítja és megoldja a statikai egyenletrendszert.

        Args:
            solver: :class:`SolverOptions`, vagy röviden ``"auto"``,
                ``"direct"`` illetve ``"cg"``. ``None`` az automatikus mód.

        Returns:
            :class:`AnalysisResult` elmozdulással, reakcióval, feszültségekkel
            és solver-diagnosztikával.

        Megjegyzés:
            Ezen az útvonalon a teljes globális ``K`` nem épül fel. Az elemi
            mátrixokból közvetlenül csak ``K_ff`` készül, ezért a memóriaigény
            a nemnulla redukált bejegyzések számával arányos.
        """

        if not self._prescribed:
            raise ValueError("the model has no displacement boundary conditions")
        options = self._solver_options(solver)
        forces = self._loads + self._body_force_vector()
        # A free_map minden globális szabadságfokhoz megmondja a redukált
        # sorszámot. Kötött szabadságfoknál -1, így nincs szükség szótárakra.
        _, free, free_map, prescribed = self._dof_partition()
        reduced_stiffness = self._assemble_reduced_stiffness(free, free_map)
        return self._solve_reduced(
            reduced_stiffness,
            free,
            free_map,
            prescribed,
            options,
            forces=forces,
        )

    def _solve_reduced(
        self,
        matrix: csr_matrix,
        free: np.ndarray,
        free_map: np.ndarray,
        prescribed: np.ndarray,
        options: SolverOptions,
        *,
        forces: np.ndarray | None = None,
        factorization: SparseDirectFactorization | None = None,
        factorization_reused: bool = False,
    ) -> AnalysisResult:
        """Egy már összeállított redukált rendszer közös megoldási útja."""

        if forces is None:
            forces = self._loads + self._body_force_vector()
        right_hand_side = self._assemble_reduced_force(forces, free, free_map, prescribed)
        displacement = prescribed.copy()
        if not len(free):
            solver_info = self._empty_solver_info(options)
        elif factorization is None:
            displacement[free], solver_info = solve_sparse_system(matrix, right_hand_side, options)
        else:
            displacement[free], solver_info = solve_factorized_system(
                matrix,
                right_hand_side,
                factorization,
                reused=factorization_reused,
            )

        reaction = self._internal_force_vector(displacement) - forces
        return self._recover_results(displacement, reaction, solver_info)

    @staticmethod
    def _solver_options(
        solver: SolverOptions | SolverMethod | str | None,
    ) -> SolverOptions:
        """A kényelmes rövid solver-megadást teljes opcióobjektummá alakítja."""

        if solver is None:
            return SolverOptions()
        if isinstance(solver, SolverOptions):
            return solver
        return SolverOptions(method=solver)

    @staticmethod
    def _empty_solver_info(options: SolverOptions) -> SolverInfo:
        """Diagnosztika olyan modellhez, amelyben nincs szabad szabadságfok."""

        method = SolverMethod.DIRECT if options.method is SolverMethod.AUTO else options.method
        return SolverInfo(
            method=method,
            free_dofs=0,
            nonzero_entries=0,
            matrix_memory_bytes=0,
            iterations=None,
            residual_norm=0.0,
            relative_residual=0.0,
        )

    def _recover_results(
        self,
        displacement: np.ndarray,
        reaction: np.ndarray,
        solver_info: SolverInfo,
    ) -> AnalysisResult:
        """Az elmozdulásból minden elem- és csomóponti eredményt visszaállít."""

        constitutive = self.material.constitutive_matrix(self.condition)
        element_results = []
        for element in self.mesh.elements:
            node_ids = np.array(element.node_ids)
            coordinates = self.mesh.nodes[node_ids]
            element_displacement = np.column_stack(
                (displacement[2 * node_ids], displacement[2 * node_ids + 1])
            ).ravel()
            strain = element.strain_at_center(coordinates, element_displacement)
            stress = constitutive @ strain
            principal_stress, principal_angle = principal_values(stress)
            integration_results = []
            integration_strain = []
            integration_stress = []
            for natural_coordinates, b_matrix, _ in element.integration_data(coordinates):
                # epsilon = B u_e, majd sigma = D epsilon. Az integrációs súly
                # itt nem kell, mert már nem integrálunk, csak pontértéket kérünk.
                point_strain = b_matrix @ element_displacement
                point_stress = constitutive @ point_strain
                point_principal, point_angle = principal_values(point_stress)
                integration_strain.append(point_strain)
                integration_stress.append(point_stress)
                integration_results.append(
                    IntegrationPointResult(
                        natural_coordinates=natural_coordinates,
                        strain=point_strain,
                        stress=point_stress,
                        von_mises=self.material.von_mises(point_stress, self.condition),
                        principal_stress=point_principal,
                        principal_angle=float(point_angle),
                    )
                )
            element_results.append(
                ElementResult(
                    strain=strain,
                    stress=stress,
                    von_mises=self.material.von_mises(stress, self.condition),
                    principal_stress=principal_stress,
                    principal_angle=float(principal_angle),
                    integration_points=tuple(integration_results),
                    # Quad4: négy Gauss-pont -> négy elemcsomópont.
                    # Triangle3: az egyetlen állandó érték háromszor ismétlődik.
                    nodal_strain=element.extrapolate_to_nodes(np.vstack(integration_strain)),
                    nodal_stress=element.extrapolate_to_nodes(np.vstack(integration_stress)),
                )
            )
        return AnalysisResult(
            model=self,
            displacement=displacement.reshape((-1, 2)),
            reaction=reaction.reshape((-1, 2)),
            element_results=tuple(element_results),
            solver_info=solver_info,
        )

    def stiffness_matrix(self, *, reduced: bool = False) -> csr_matrix:
        """Teljes vagy peremfeltételekkel redukált CSR merevségi mátrixot ad.

        Args:
            reduced: ``False`` esetén a teljes ``K``, ``True`` esetén ``K_ff``.

        Returns:
            SciPy ``csr_matrix``. A duplikált elemi hozzájárulások a COO→CSR
            átalakításkor automatikusan összeadódnak.

        Ez az explicit API vizsgálathoz és ábrázoláshoz készült. A ``solve``
        ennél takarékosabb útvonalon közvetlenül redukált mátrixot állít össze.
        """

        if reduced:
            if not self._prescribed:
                raise ValueError("a reduced matrix requires boundary conditions")
            forces = self._loads + self._body_force_vector()
            _, free, free_map, prescribed = self._dof_partition()
            return self._assemble_reduced_system(forces, free, free_map, prescribed)[0]

        constitutive = self.material.constitutive_matrix(self.condition)
        return self._assemble_global_matrix(
            lambda element, coordinates: element.stiffness(
                coordinates, constitutive, self.thickness
            )
        )

    def plot_stiffness_matrix(
        self,
        *,
        kind: str = "sparsity",
        reduced: bool = False,
        max_points: int = 250_000,
        ax=None,
        style=None,
    ):
        """Kirajzolja a merevségi mátrix szerkezetét vagy koefficienseit.

        Args:
            kind: ``"sparsity"`` a nemnulla mintához; ``"magnitude"`` a
                ``log10(abs(Kij))`` színezéshez.
            reduced: A teljes ``K`` vagy a megtámasztások utáni ``K_ff``.
            max_points: Nagy mátrixnál legfeljebb ennyi pontot rajzol ki.
            ax: Opcionális Matplotlib tengely.

        Returns:
            A használt Matplotlib ``Axes``. A mátrix soha nem lesz sűrűvé
            alakítva az ábrázolás kedvéért.
        """

        from .plotting import plot_sparse_matrix

        matrix = self.stiffness_matrix(reduced=reduced)
        title = f"{self.name} — {'reduced ' if reduced else ''}stiffness matrix"
        return plot_sparse_matrix(
            matrix,
            kind=kind,
            max_points=max_points,
            title=title,
            ax=ax,
            style=style,
        )

    def mass_matrix(self):
        """Összeállítja a teljes konzisztens transzlációs tömegmátrixot.

        Ezt főként oktatáshoz, sajátérték- vagy későbbi dinamikai számításhoz
        érdemes lekérni. Az egyszerű statikai testsúly számításakor nem építjük
        fel, hanem az elemi ``Me @ acceleration`` vektorokat összegezzük.
        """

        return self._assemble_global_matrix(
            lambda element, coordinates: element.mass_matrix(
                coordinates, self.material.density, self.thickness
            )
        )

    def _assemble_global_matrix(
        self,
        element_matrix: Callable[[Element2D, np.ndarray], np.ndarray],
    ) -> csr_matrix:
        """Elemi mátrixokból teljes COO→CSR globális mátrixot épít.

        A merevségi és tömegmátrix ugyanazt a memóriatakarékos scatter
        algoritmust használja; csak az elemi mátrixot előállító függvény tér el.
        """

        entry_count = sum((2 * len(element.node_ids)) ** 2 for element in self.mesh.elements)
        rows = np.empty(entry_count, dtype=np.int64)
        columns = np.empty(entry_count, dtype=np.int64)
        values = np.empty(entry_count, dtype=float)
        cursor = 0
        for element in self.mesh.elements:
            node_ids = np.array(element.node_ids)
            coordinates = self.mesh.nodes[node_ids]
            local_matrix = element_matrix(element, coordinates)
            dofs = _element_dofs(node_ids)
            block_size = len(dofs) ** 2
            target = slice(cursor, cursor + block_size)
            rows[target] = np.repeat(dofs, len(dofs))
            columns[target] = np.tile(dofs, len(dofs))
            values[target] = local_matrix.ravel()
            cursor += block_size
        size = 2 * self.mesh.node_count
        matrix = coo_matrix((values, (rows, columns)), shape=(size, size)).tocsr()
        matrix.eliminate_zeros()
        return matrix

    def _body_force_vector(self) -> np.ndarray:
        """Testsűrűségből és gyorsulásból konzisztens csomóponti erőt készít."""
        if self.material.density == 0.0 or np.allclose(self._body_acceleration, 0.0):
            return np.zeros_like(self._loads)
        forces = np.zeros_like(self._loads)
        for element in self.mesh.elements:
            node_ids = np.asarray(element.node_ids)
            coordinates = self.mesh.nodes[node_ids]
            element_mass = element.mass_matrix(coordinates, self.material.density, self.thickness)
            element_acceleration = np.tile(self._body_acceleration, len(node_ids))
            element_force = element_mass @ element_acceleration
            dofs = np.column_stack((2 * node_ids, 2 * node_ids + 1)).ravel()
            forces[dofs] += element_force
        return forces

    def _dof_partition(self):
        """Elkészíti a kötött/szabad indexeket és a globális→redukált leképezést."""
        dof_count = 2 * self.mesh.node_count
        constrained = np.array(sorted(self._prescribed), dtype=int)
        free = np.setdiff1d(np.arange(dof_count), constrained, assume_unique=True)
        free_map = np.full(dof_count, -1, dtype=int)
        free_map[free] = np.arange(len(free))
        prescribed = np.zeros(dof_count, dtype=float)
        prescribed[constrained] = [self._prescribed[dof] for dof in constrained]
        return constrained, free, free_map, prescribed

    def _assemble_reduced_system(
        self,
        forces: np.ndarray,
        free: np.ndarray,
        free_map: np.ndarray,
        prescribed: np.ndarray,
    ) -> tuple[csr_matrix, np.ndarray]:
        """Közvetlenül összeállítja ``K_ff``-et és a redukált jobb oldalt.

        A kényszerített szabadságfokok oszlopai nem kerülnek a ritka mátrixba.
        Hatásukat elemenként levonjuk a jobb oldalból::

            rhs_f -= Ke[f, c] @ u_c

        Így a teljes ``K``, majd annak ``K[free][:, free]`` másolata sem foglal
        memóriát. Az összeállítás ideiglenes tárhelye is pontosan előre méretezett.
        """

        matrix = self._assemble_reduced_stiffness(free, free_map)
        right_hand_side = self._assemble_reduced_force(forces, free, free_map, prescribed)
        return matrix, right_hand_side

    def _assemble_reduced_stiffness(
        self,
        free: np.ndarray,
        free_map: np.ndarray,
    ) -> csr_matrix:
        """Közvetlenül összeállítja a szabad szabadságfokok ``K_ff`` mátrixát."""

        constitutive = self.material.constitutive_matrix(self.condition)
        # Első, olcsó topológiai menet: pontosan megszámoljuk, hány lokális
        # szabad-szabad bejegyzést kell tárolni a COO összeállításhoz.
        entry_count = 0
        # Második menet: elemi merevség, jobb oldal korrekció és COO tripletek.
        for element in self.mesh.elements:
            node_ids = np.asarray(element.node_ids)
            dofs = np.column_stack((2 * node_ids, 2 * node_ids + 1)).ravel()
            local_free_count = np.count_nonzero(free_map[dofs] >= 0)
            entry_count += local_free_count**2
        rows = np.empty(entry_count, dtype=np.int64)
        columns = np.empty(entry_count, dtype=np.int64)
        values = np.empty(entry_count, dtype=float)
        cursor = 0
        for element in self.mesh.elements:
            node_ids = np.asarray(element.node_ids)
            coordinates = self.mesh.nodes[node_ids]
            element_matrix = element.stiffness(coordinates, constitutive, self.thickness)
            dofs = np.column_stack((2 * node_ids, 2 * node_ids + 1)).ravel()
            mapped = free_map[dofs]
            local_free = np.flatnonzero(mapped >= 0)
            if not len(local_free):
                continue
            mapped_free = mapped[local_free]
            block = element_matrix[np.ix_(local_free, local_free)]
            block_size = len(local_free) ** 2
            target = slice(cursor, cursor + block_size)
            rows[target] = np.repeat(mapped_free, len(local_free))
            columns[target] = np.tile(mapped_free, len(local_free))
            values[target] = block.ravel()
            cursor += block_size
        matrix = coo_matrix(
            (values[:cursor], (rows[:cursor], columns[:cursor])),
            shape=(len(free), len(free)),
        ).tocsr()
        matrix.eliminate_zeros()
        return matrix

    def _assemble_reduced_force(
        self,
        forces: np.ndarray,
        free: np.ndarray,
        free_map: np.ndarray,
        prescribed: np.ndarray,
    ) -> np.ndarray:
        """Elkészíti a ``f_f - K_fc u_c`` redukált jobb oldalt.

        A gyakori, nulla értékű támaszoknál nincs szükség elemi mátrixra: a
        módszer egyszerűen a külső erő szabad komponenseit másolja. Nem nulla
        előírt elmozdulásnál csak a szükséges ``K_fc u_c`` tagokat számítja ki.
        """

        right_hand_side = forces[free].copy()
        if not np.any(prescribed):
            return right_hand_side
        constitutive = self.material.constitutive_matrix(self.condition)
        for element in self.mesh.elements:
            node_ids = np.asarray(element.node_ids)
            dofs = np.column_stack((2 * node_ids, 2 * node_ids + 1)).ravel()
            mapped = free_map[dofs]
            local_free = np.flatnonzero(mapped >= 0)
            local_constrained = np.flatnonzero(mapped < 0)
            if not len(local_free) or not len(local_constrained):
                continue
            constrained_values = prescribed[dofs[local_constrained]]
            if not np.any(constrained_values):
                continue
            coordinates = self.mesh.nodes[node_ids]
            element_matrix = element.stiffness(coordinates, constitutive, self.thickness)
            right_hand_side[mapped[local_free]] -= (
                element_matrix[np.ix_(local_free, local_constrained)] @ constrained_values
            )
        return right_hand_side

    def _internal_force_vector(self, displacement: np.ndarray) -> np.ndarray:
        """Elemenként összeállítja a ``K @ u`` belső csomóponti erővektort."""
        constitutive = self.material.constitutive_matrix(self.condition)
        internal_force = np.zeros_like(displacement)
        for element in self.mesh.elements:
            node_ids = np.asarray(element.node_ids)
            coordinates = self.mesh.nodes[node_ids]
            dofs = np.column_stack((2 * node_ids, 2 * node_ids + 1)).ravel()
            element_matrix = element.stiffness(coordinates, constitutive, self.thickness)
            internal_force[dofs] += element_matrix @ displacement[dofs]
        return internal_force

    def _validate_node(self, node: int) -> None:
        if not 0 <= node < self.mesh.node_count:
            raise IndexError(f"node index {node} is outside the mesh")
