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

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .material import LinearElasticMaterial, PlaneCondition
from .mesh import Mesh
from .result import (
    AnalysisResult,
    ElementResult,
    IntegrationPointResult,
    principal_values,
)
from .solver import SolverInfo, SolverMethod, SolverOptions, solve_sparse_system


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

    def add_nodal_loads(self, nodes: list[int], *, fx: float = 0.0, fy: float = 0.0) -> Model:
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

    def fix_nodes(self, nodes: list[int], *, x: bool = True, y: bool = True) -> Model:
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

        ``tx`` és ``ty`` erő/felület dimenziójú érték. Egy kétcsomópontos,
        ``L`` hosszú peremél mindkét végpontjára ``traction * thickness * L/2``
        konzisztens csomóponti erő jut. A közös csomópontok hozzájárulásai
        automatikusan összeadódnak.
        """

        if not np.isfinite(tx) or not np.isfinite(ty):
            raise ValueError("boundary traction components must be finite")
        edges = self.mesh.boundary_edges(name)
        if not edges:
            raise ValueError(f"boundary {name!r} has no edges")
        traction = np.asarray([tx, ty], dtype=float)
        for first, second in edges:
            length = float(np.linalg.norm(self.mesh.nodes[second] - self.mesh.nodes[first]))
            nodal_force = traction * self.thickness * length / 2.0
            self._loads[2 * first : 2 * first + 2] += nodal_force
            self._loads[2 * second : 2 * second + 2] += nodal_force
        return self

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
            ids = element.node_ids
            for index, first_node in enumerate(ids):
                second_node = ids[(index + 1) % len(ids)]
                key = tuple(sorted((first_node, second_node)))
                adjacent_elements.setdefault(key, []).append(element)
        for first, second in self.mesh.boundary_edges(name):
            first_point = self.mesh.nodes[first]
            second_point = self.mesh.nodes[second]
            tangent = second_point - first_point
            length = float(np.linalg.norm(tangent))
            if length <= np.finfo(float).eps:
                raise ValueError(f"boundary {name!r} contains a zero-length edge")
            midpoint = 0.5 * (first_point + second_point)
            adjacent = adjacent_elements.get(tuple(sorted((first, second))), [])
            if len(adjacent) != 1:
                raise ValueError(
                    f"boundary edge {(first, second)} must have exactly one adjacent element"
                )
            centroid = self.mesh.nodes[list(adjacent[0].node_ids)].mean(axis=0)
            outward = np.array([tangent[1], -tangent[0]], dtype=float) / length
            if np.dot(outward, midpoint - centroid) < 0.0:
                outward *= -1.0
            # A pozitív mérnöki nyomás a test belseje felé hat.
            nodal_force = -pressure * outward * self.thickness * length / 2.0
            self._loads[2 * first : 2 * first + 2] += nodal_force
            self._loads[2 * second : 2 * second + 2] += nodal_force
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
    ):
        """Kirajzolja az x-, y- és kétirányú megtámasztásokat és a terheket."""

        from .plotting import plot_boundary_conditions

        return plot_boundary_conditions(
            self,
            ax=ax,
            show_mesh=show_mesh,
            show_loads=show_loads,
        )

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
        if solver is None:
            options = SolverOptions()
        elif isinstance(solver, SolverOptions):
            options = solver
        else:
            options = SolverOptions(method=solver)
        forces = self._loads + self._body_force_vector()
        # A free_map minden globális szabadságfokhoz megmondja a redukált
        # sorszámot. Kötött szabadságfoknál -1, így nincs szükség szótárakra.
        _, free, free_map, prescribed = self._dof_partition()
        reduced_stiffness, reduced_force = self._assemble_reduced_system(
            forces, free, free_map, prescribed
        )
        displacement = prescribed.copy()
        if len(free):
            # A ritka solver csak K_ff-et látja. A kényszerített u_c értékek
            # hatása már az összeállított jobb oldalba került.
            displacement[free], solver_info = solve_sparse_system(
                reduced_stiffness, reduced_force, options
            )
        else:
            method = SolverMethod.DIRECT if options.method is SolverMethod.AUTO else options.method
            solver_info = SolverInfo(
                method=method,
                free_dofs=0,
                nonzero_entries=0,
                matrix_memory_bytes=0,
                iterations=None,
                residual_norm=0.0,
                relative_residual=0.0,
            )

        # Reakció = belső csomóponti erő - külső erő. A belső erőt újra
        # elemenként összegezzük, ezért a reakcióhoz sem kell teljes globális K.
        internal_force = self._internal_force_vector(displacement)
        reaction = internal_force - forces
        return self._recover_results(displacement, reaction, solver_info)

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
        # Előre ismert az összes lokális mátrixbejegyzés száma. A fix méretű
        # NumPy-tömbök jóval kisebbek, mint három, Python-objektumokból álló lista.
        entry_count = sum((2 * len(element.node_ids)) ** 2 for element in self.mesh.elements)
        rows = np.empty(entry_count, dtype=np.int64)
        columns = np.empty(entry_count, dtype=np.int64)
        values = np.empty(entry_count, dtype=float)
        cursor = 0
        for element in self.mesh.elements:
            node_ids = np.array(element.node_ids)
            coordinates = self.mesh.nodes[node_ids]
            element_matrix = element.stiffness(coordinates, constitutive, self.thickness)
            dofs = np.column_stack((2 * node_ids, 2 * node_ids + 1)).ravel()
            block_size = len(dofs) ** 2
            target = slice(cursor, cursor + block_size)
            rows[target] = np.repeat(dofs, len(dofs))
            columns[target] = np.tile(dofs, len(dofs))
            values[target] = element_matrix.ravel()
            cursor += block_size
        size = 2 * self.mesh.node_count
        # A COO formátum jól használható összeállításhoz, a CSR pedig gyors
        # mátrix-vektor szorzást, szeletelést és ritka megoldást biztosít.
        matrix = coo_matrix((values, (rows, columns)), shape=(size, size)).tocsr()
        matrix.eliminate_zeros()
        return matrix

    def plot_stiffness_matrix(
        self,
        *,
        kind: str = "sparsity",
        reduced: bool = False,
        max_points: int = 250_000,
        ax=None,
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
        )

    def mass_matrix(self):
        """Összeállítja a teljes konzisztens transzlációs tömegmátrixot.

        Ezt főként oktatáshoz, sajátérték- vagy későbbi dinamikai számításhoz
        érdemes lekérni. Az egyszerű statikai testsúly számításakor nem építjük
        fel, hanem az elemi ``Me @ acceleration`` vektorokat összegezzük.
        """

        entry_count = sum((2 * len(element.node_ids)) ** 2 for element in self.mesh.elements)
        rows = np.empty(entry_count, dtype=np.int64)
        columns = np.empty(entry_count, dtype=np.int64)
        values = np.empty(entry_count, dtype=float)
        cursor = 0
        for element in self.mesh.elements:
            node_ids = np.array(element.node_ids)
            coordinates = self.mesh.nodes[node_ids]
            element_matrix = element.mass_matrix(coordinates, self.material.density, self.thickness)
            dofs = np.column_stack((2 * node_ids, 2 * node_ids + 1)).ravel()
            block_size = len(dofs) ** 2
            target = slice(cursor, cursor + block_size)
            rows[target] = np.repeat(dofs, len(dofs))
            columns[target] = np.tile(dofs, len(dofs))
            values[target] = element_matrix.ravel()
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
        right_hand_side = forces[free].copy()
        cursor = 0
        for element in self.mesh.elements:
            node_ids = np.asarray(element.node_ids)
            coordinates = self.mesh.nodes[node_ids]
            element_matrix = element.stiffness(coordinates, constitutive, self.thickness)
            dofs = np.column_stack((2 * node_ids, 2 * node_ids + 1)).ravel()
            mapped = free_map[dofs]
            local_free = np.flatnonzero(mapped >= 0)
            local_constrained = np.flatnonzero(mapped < 0)
            if len(local_constrained) and len(local_free):
                right_hand_side[mapped[local_free]] -= (
                    element_matrix[np.ix_(local_free, local_constrained)]
                    @ prescribed[dofs[local_constrained]]
                )
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
        return matrix, right_hand_side

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
