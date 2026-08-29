"""Opcionális, többformátumú háló- és eredmény I/O a meshio segítségével.

A modul késleltetve importálja a ``meshio`` csomagot, ezért a PrimFEM alapvető
hálózása és megoldója az opcionális függőség nélkül is használható. A belső
``Mesh`` mindig saját, ellenőrzött T3/T6/Q4 elemeket és nullától induló
indexeket kap.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .elements import Quad4, Triangle3, Triangle6


class MeshioNotInstalledError(ImportError):
    """Akkor keletkezik, ha a többformátumú I/O függősége nincs telepítve."""


def _meshio():
    try:
        import meshio
    except ImportError as exc:
        raise MeshioNotInstalledError(
            "meshio support requires: python -m pip install 'primfem[io]'"
        ) from exc
    return meshio


def _positive_element(cell_type: str, node_ids: np.ndarray, points: np.ndarray):
    """A külső cellasorrendet pozitív PrimFEM-elemmé alakítja."""

    ids = tuple(map(int, node_ids))
    corners = points[list(ids[:4] if cell_type == "quad" else ids[:3])]
    if cell_type.startswith("triangle"):
        first = corners[1] - corners[0]
        second = corners[2] - corners[0]
        cross = first[0] * second[1] - first[1] * second[0]
        if cross < 0.0:
            ids = (
                (ids[0], ids[2], ids[1])
                if cell_type == "triangle"
                else (ids[0], ids[2], ids[1], ids[5], ids[4], ids[3])
            )
        return Triangle3(ids) if cell_type == "triangle" else Triangle6(ids)
    signed_twice_area = float(
        np.dot(corners[:, 0], np.roll(corners[:, 1], -1))
        - np.dot(corners[:, 1], np.roll(corners[:, 0], -1))
    )
    if signed_twice_area < 0.0:
        ids = (ids[0], ids[3], ids[2], ids[1])
    return Quad4(ids)


def _read_cell_blocks(source, points: np.ndarray):
    """A területi elemeket és a külön kezelt peremblokkokat olvassa ki."""

    elements = []
    boundary_blocks: dict[int, list[tuple[int, ...]]] = {}
    supported_area_types = {"triangle", "triangle6", "quad"}
    for block_index, block in enumerate(source.cells):
        if block.type in supported_area_types:
            elements.extend(
                _positive_element(block.type, row, points) for row in np.asarray(block.data)
            )
        elif block.type == "line":
            boundary_blocks[block_index] = [tuple(map(int, row)) for row in block.data]
        elif block.type == "line3":
            # meshio: (első végpont, második végpont, középcsomópont).
            boundary_blocks[block_index] = [
                (int(row[0]), int(row[2]), int(row[1])) for row in block.data
            ]
        elif getattr(block, "dim", None) == 2:
            raise ValueError(f"unsupported two-dimensional cell type: {block.type}")
    if not elements:
        raise ValueError("mesh file contains no supported Triangle3, Triangle6 or Quad4 cells")
    return elements, boundary_blocks


def _named_boundary_edges(source, boundary_blocks):
    """A cell-set, cell-data és Gmsh fizikai neveket egységesíti."""

    edge_sets: dict[str, list[tuple[int, ...]]] = defaultdict(list)
    for name, selections in source.cell_sets.items():
        for block_index, selected in enumerate(selections):
            if block_index in boundary_blocks and selected is not None:
                edges = boundary_blocks[block_index]
                edge_sets[name].extend(edges[int(index)] for index in np.asarray(selected).ravel())

    # A VTU egyes cellahalmazokat -1/0 jelölésű cellaadattá alakít.
    for name, block_values in source.cell_data.items():
        if name.startswith("gmsh:"):
            continue
        for block_index, values in enumerate(block_values):
            values = np.asarray(values)
            if (
                block_index in boundary_blocks
                and values.ndim == 1
                and np.issubdtype(values.dtype, np.integer)
            ):
                edges = boundary_blocks[block_index]
                edge_sets[name].extend(edges[int(index)] for index in np.flatnonzero(values >= 0))

    physical_data = source.cell_data.get("gmsh:physical")
    if physical_data is not None:
        _extend_physical_boundaries(edge_sets, source, boundary_blocks, physical_data)
    return edge_sets


def _extend_physical_boundaries(edge_sets, source, boundary_blocks, physical_data) -> None:
    """A Gmsh egydimenziós fizikai csoportjait név szerinti peremmé alakítja."""

    for name, definition in source.field_data.items():
        physical_tag, dimension = map(int, np.asarray(definition).ravel()[:2])
        if dimension != 1:
            continue
        for block_index, tags in enumerate(physical_data):
            if block_index in boundary_blocks:
                edges = boundary_blocks[block_index]
                selected = np.flatnonzero(np.asarray(tags) == physical_tag)
                edge_sets[name].extend(edges[int(index)] for index in selected)


def _compact_imported_mesh(points, elements, edge_sets, point_sets, point_data):
    """Eltávolítja a nem használt pontokat és újraszámozza a halmazokat."""

    from .mesh import Mesh

    used = sorted({node for element in elements for node in element.node_ids})
    old_to_new = {old: new for new, old in enumerate(used)}
    compact_elements = [
        type(element)(tuple(old_to_new[node] for node in element.node_ids)) for element in elements
    ]
    compact_edges = {
        name: [
            tuple(old_to_new[node] for node in edge)
            for edge in edges
            if all(node in old_to_new for node in edge)
        ]
        for name, edges in edge_sets.items()
    }
    node_sets = {
        name: [old_to_new[int(node)] for node in nodes if int(node) in old_to_new]
        for name, nodes in point_sets.items()
    }
    for name, raw_values in point_data.items():
        values = np.asarray(raw_values)
        if (
            values.ndim == 1
            and np.issubdtype(values.dtype, np.integer)
            and np.all(values >= -1)
            and np.any(values < 0)
        ):
            node_sets.setdefault(name, []).extend(
                old_to_new[int(node)]
                for node in np.flatnonzero(values >= 0)
                if int(node) in old_to_new
            )
    for name, edges in compact_edges.items():
        node_sets.setdefault(name, []).extend(node for edge in edges for node in edge)
    return Mesh(points[used], compact_elements, node_sets=node_sets, edge_sets=compact_edges)


def read_mesh(path: str | Path):
    """T3, T6 vagy Q4 kétdimenziós hálót olvas bármely meshio-formátumból."""

    meshio = _meshio()
    source = meshio.read(Path(path))
    points = np.asarray(source.points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("mesh file must contain at least x and y coordinates")
    if points.shape[1] > 2 and not np.allclose(points[:, 2:], points[0, 2:]):
        raise ValueError("mesh is not planar; PrimFEM currently supports 2D meshes")
    points_2d = points[:, :2]

    elements, boundary_blocks = _read_cell_blocks(source, points_2d)
    edge_sets = _named_boundary_edges(source, boundary_blocks)
    return _compact_imported_mesh(
        points_2d,
        elements,
        edge_sets,
        source.point_sets,
        source.point_data,
    )


def _cell_groups(mesh):
    """Meshio-cellablokkokat és az eredeti elemindexeket adja."""

    kinds = ((Triangle3, "triangle"), (Triangle6, "triangle6"), (Quad4, "quad"))
    cells = []
    indices = []
    for element_class, cell_type in kinds:
        selected = [
            index
            for index, element in enumerate(mesh.elements)
            if isinstance(element, element_class)
        ]
        if selected:
            cells.append(
                (cell_type, np.asarray([mesh.elements[index].node_ids for index in selected]))
            )
            indices.append(np.asarray(selected, dtype=int))
    return cells, indices


def write_mesh(mesh, path: str | Path) -> Path:
    """Hálót és névvel ellátott peremhalmazokat ír meshio-formátumba."""

    meshio = _meshio()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cells, _ = _cell_groups(mesh)
    volume_block_count = len(cells)
    boundary_rows: dict[str, list[tuple[int, ...]]] = {"line": [], "line3": []}
    boundary_index: dict[str, dict[tuple[int, ...], int]] = {"line": {}, "line3": {}}
    named_indices: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"line": [], "line3": []})
    for name, edges in mesh.edge_sets.items():
        for edge in edges:
            cell_type = "line" if len(edge) == 2 else "line3"
            # meshio a kvadratikus vonalat (első, utolsó, középső) sorrendben várja.
            row = edge if len(edge) == 2 else (edge[0], edge[2], edge[1])
            reverse = row[::-1] if len(row) == 2 else (row[1], row[0], row[2])
            lookup = boundary_index[cell_type]
            index = lookup.get(row, lookup.get(reverse))
            if index is None:
                index = len(boundary_rows[cell_type])
                boundary_rows[cell_type].append(row)
                lookup[row] = index
            named_indices[name][cell_type].append(index)
    boundary_types = [cell_type for cell_type in ("line", "line3") if boundary_rows[cell_type]]
    cells.extend(
        (cell_type, np.asarray(boundary_rows[cell_type], dtype=int)) for cell_type in boundary_types
    )
    cell_sets = {
        name: [np.array([], dtype=int) for _ in range(volume_block_count)]
        + [np.asarray(by_type[cell_type], dtype=int) for cell_type in boundary_types]
        for name, by_type in named_indices.items()
    }
    point_sets = {name: np.asarray(nodes, dtype=int) for name, nodes in mesh.node_sets.items()}
    points = np.column_stack((mesh.nodes, np.zeros(mesh.node_count)))
    meshio.write(
        output,
        meshio.Mesh(
            points=points,
            cells=cells,
            point_sets=point_sets,
            cell_sets=cell_sets,
        ),
    )
    return output


def write_result(result, path: str | Path) -> Path:
    """Teljes FEM-eredményt ír VTU, VTK, XDMF vagy más meshio-formátumba."""

    meshio = _meshio()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh = result.model.mesh
    cells, group_indices = _cell_groups(mesh)
    points = np.column_stack((mesh.nodes, np.zeros(mesh.node_count)))

    def vector3(values):
        values = np.asarray(values, dtype=float)
        return np.column_stack((values, np.zeros(len(values))))

    first_principal, second_principal = result.nodal_principal_vectors
    point_data = {
        "displacement": vector3(result.displacement),
        "reaction": vector3(result.reaction),
        "displacement_magnitude": result.displacement_magnitude,
        "strain": result.nodal_strain,
        "stress": result.nodal_stress,
        "von_mises": result.nodal_von_mises,
        "principal_stress": result.nodal_principal_stress,
        "principal_stress_1_vector": vector3(first_principal),
        "principal_stress_2_vector": vector3(second_principal),
    }
    element_fields = {
        "strain": result.strain,
        "stress": result.stress,
        "von_mises": result.von_mises,
        "principal_stress": result.principal_stress,
        "principal_angle": result.principal_angle,
    }
    cell_data = {
        name: [np.asarray(values)[indices] for indices in group_indices]
        for name, values in element_fields.items()
    }
    meshio.write(
        output,
        meshio.Mesh(
            points=points,
            cells=cells,
            point_data=point_data,
            cell_data=cell_data,
        ),
    )
    return output
