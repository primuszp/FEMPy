"""Eredményexport külső utófeldolgozó programokhoz.

A modul jelenleg a klasszikus ASCII VTK ``UNSTRUCTURED_GRID`` formátumot írja,
amelyet a ParaView közvetlenül megnyit. Az export nullától induló cellaindexeket,
csomóponti vektorokat, csomóponti skalárokat, elemmezőket és külön Gauss-ponti
cellamezőket tartalmaz.
"""

from __future__ import annotations

from pathlib import Path


def write_vtk(result, path: str | Path) -> Path:
    """ParaView által olvasható legacy VTK fájlt készít.

    Args:
        result: Megoldott :class:`AnalysisResult`.
        path: A célfájl. A hiányzó szülőkönyvtár automatikusan létrejön.

    Returns:
        A létrehozott fájl :class:`Path` objektuma.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh = result.model.mesh
    total_cell_entries = sum(len(element.node_ids) + 1 for element in mesh.elements)
    # A legacy VTK szöveges formátum sororientált. Először a geometria és a
    # topológia, utána a POINT_DATA, végül a CELL_DATA mezők következnek.
    lines = [
        "# vtk DataFile Version 2.0",
        result.model.name,
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {mesh.node_count} float",
    ]
    lines.extend(f"{x:.12g} {y:.12g} 0" for x, y in mesh.nodes)
    lines.append(f"CELLS {mesh.element_count} {total_cell_entries}")
    for element in mesh.elements:
        ids = " ".join(str(node) for node in element.node_ids)
        lines.append(f"{len(element.node_ids)} {ids}")
    lines.append(f"CELL_TYPES {mesh.element_count}")
    lines.extend(str(element.vtk_cell_type) for element in mesh.elements)

    def add_scalar(name, values):
        """Egy VTK skalármezőt fűz a kimeneti sorokhoz."""
        lines.append(f"SCALARS {name} float 1")
        lines.append("LOOKUP_TABLE default")
        lines.extend(f"{float(value):.12g}" for value in values)

    def add_vector(name, values):
        """Egy kétdimenziós mezőt z=0 komponenssel VTK vektorként ír."""
        lines.append(f"VECTORS {name} float")
        lines.extend(f"{float(x):.12g} {float(y):.12g} 0" for x, y in values)

    lines.append(f"POINT_DATA {mesh.node_count}")
    add_vector("displacement", result.displacement)
    add_vector("reaction", result.reaction)
    add_scalar("displacement_magnitude", result.displacement_magnitude)
    for name, values in (
        ("strain_x", result.nodal_strain[:, 0]),
        ("strain_y", result.nodal_strain[:, 1]),
        ("strain_xy", result.nodal_strain[:, 2]),
        ("stress_x", result.nodal_stress[:, 0]),
        ("stress_y", result.nodal_stress[:, 1]),
        ("stress_xy", result.nodal_stress[:, 2]),
        ("principal_stress_1", result.nodal_principal_stress[:, 0]),
        ("principal_stress_2", result.nodal_principal_stress[:, 1]),
        ("von_mises", result.nodal_von_mises),
        (
            "principal_angle_degrees",
            result.nodal_principal_angle * 180.0 / 3.141592653589793,
        ),
    ):
        add_scalar(name, values)
    first_vector, second_vector = result.nodal_principal_vectors
    add_vector("principal_stress_1_vector", first_vector)
    add_vector("principal_stress_2_vector", second_vector)

    lines.append(f"CELL_DATA {mesh.element_count}")
    for name, values in (
        ("strain_x", result.strain[:, 0]),
        ("strain_y", result.strain[:, 1]),
        ("strain_xy", result.strain[:, 2]),
        ("stress_x", result.stress[:, 0]),
        ("stress_y", result.stress[:, 1]),
        ("stress_xy", result.stress[:, 2]),
        ("principal_stress_1", result.principal_stress[:, 0]),
        ("principal_stress_2", result.principal_stress[:, 1]),
        ("principal_angle_degrees", result.principal_angle * 180.0 / 3.141592653589793),
        ("von_mises", result.von_mises),
    ):
        add_scalar(name, values)

    # VTK cell fields keep every Gauss point individually.  Elements with one
    # integration point (Triangle3) repeat their constant value in mixed meshes.
    integration_point_count = max(
        len(element_result.integration_points) for element_result in result.element_results
    )
    for point_index in range(integration_point_count):
        points = [
            element_result.integration_points[
                min(point_index, len(element_result.integration_points) - 1)
            ]
            for element_result in result.element_results
        ]
        suffix = f"gp{point_index + 1}"
        for name, values in (
            (f"strain_x_{suffix}", [point.strain[0] for point in points]),
            (f"strain_y_{suffix}", [point.strain[1] for point in points]),
            (f"strain_xy_{suffix}", [point.strain[2] for point in points]),
            (f"stress_x_{suffix}", [point.stress[0] for point in points]),
            (f"stress_y_{suffix}", [point.stress[1] for point in points]),
            (f"stress_xy_{suffix}", [point.stress[2] for point in points]),
            (
                f"principal_stress_1_{suffix}",
                [point.principal_stress[0] for point in points],
            ),
            (
                f"principal_stress_2_{suffix}",
                [point.principal_stress[1] for point in points],
            ),
            (f"von_mises_{suffix}", [point.von_mises for point in points]),
        ):
            add_scalar(name, values)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
