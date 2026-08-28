"""Matplotlib-alapú, oktatási célú végeselemes vizualizációk.

A modul hálót, deformált eredménymezőt, főfeszültségi irányt és ritka mátrixot
rajzol. A csomóponti mezők belső háromszög-felbontáson Gouraud-színezést kapnak;
az elemmezők elemenként állandó színűek. A színskála mindig pontosan a
rajztéglalap magasságát követi.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse import issparse


def plot_mesh(mesh, *, ax=None, show_node_ids: bool = False):
    """Deformálatlan hálót rajzol, opcionális csomópontszámokkal."""

    if ax is None:
        _, ax = plt.subplots()
    polygons = [mesh.nodes[list(element.node_ids)] for element in mesh.elements]
    collection = PolyCollection(polygons, facecolors="none", edgecolors="#34495e", linewidths=0.8)
    ax.add_collection(collection)
    if show_node_ids:
        for index, (x, y) in enumerate(mesh.nodes):
            ax.text(x, y, str(index), fontsize=8, color="#c0392b")
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def plot_boundaries(
    mesh,
    *,
    names=None,
    ax=None,
    show_mesh: bool = True,
    show_labels: bool = True,
    linewidth: float = 3.0,
    cmap: str = "tab10",
):
    """Színesen kirajzolja a háló névvel ellátott peremcsoportjait.

    Args:
        mesh: Peremhalmazokat tartalmazó :class:`fempy.mesh.Mesh`.
        names: Kirajzolandó peremnevek. ``None`` minden peremet jelent.
        ax: Opcionális Matplotlib tengely.
        show_mesh: Látszódjon-e halványan a teljes végeselemes háló.
        show_labels: Kerüljön-e a perem neve annak közepe közelébe.
        linewidth: A színes peremvonal vastagsága.
        cmap: Diszkrét Matplotlib-színtérkép neve.

    Returns:
        A használt Matplotlib tengely.
    """

    if ax is None:
        _, ax = plt.subplots()
    selected = list(mesh.boundary_names if names is None else names)
    if not selected:
        raise ValueError("the mesh has no named boundaries to plot")
    unknown = [name for name in selected if name not in mesh.boundary_names]
    if unknown:
        raise KeyError(f"unknown boundaries {unknown}; available: {', '.join(mesh.boundary_names)}")
    if show_mesh:
        polygons = [mesh.nodes[list(element.node_ids)] for element in mesh.elements]
        ax.add_collection(
            PolyCollection(
                polygons,
                facecolors="#f5f7fa",
                edgecolors="#c5ccd3",
                linewidths=0.5,
            )
        )

    colour_map = plt.get_cmap(cmap, len(selected))
    for index, name in enumerate(selected):
        edges = mesh.boundary_edges(name)
        segments = [mesh.nodes[list(edge)] for edge in edges]
        colour = colour_map(index)
        ax.add_collection(
            LineCollection(
                segments,
                colors=[colour],
                linewidths=linewidth,
                label=name,
                zorder=3,
            )
        )
        if show_labels:
            nodes = mesh.boundary_nodes(name)
            center = mesh.nodes[nodes].mean(axis=0)
            ax.annotate(
                name,
                center,
                xytext=(5, 5),
                textcoords="offset points",
                color=colour,
                fontsize=9,
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                    "pad": 1.5,
                },
                zorder=4,
            )
    ax.autoscale()
    ax.margins(0.05)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(title="Peremek", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return ax


def plot_boundary_conditions(
    model,
    *,
    ax=None,
    show_mesh: bool = True,
    show_loads: bool = True,
):
    """A modell kinematikai peremfeltételeit könnyen olvashatóan ábrázolja.

    A narancssárga jobbra mutató háromszög az x, a kék felfelé mutató
    háromszög az y irányú rögzítést jelenti. A piros négyzet mindkét irány
    rögzítését mutatja. Nem nulla előírt elmozdulás lila rombuszként jelenik
    meg. Az opcionális fekete nyilak a koncentrált csomóponti terhek.
    """

    if ax is None:
        _, ax = plt.subplots()
    if show_mesh:
        polygons = [model.mesh.nodes[list(element.node_ids)] for element in model.mesh.elements]
        ax.add_collection(
            PolyCollection(
                polygons,
                facecolors="#f7f9fb",
                edgecolors="#c5ccd3",
                linewidths=0.55,
            )
        )

    prescribed = model.prescribed_displacements
    fixed_x = np.isclose(prescribed[:, 0], 0.0, equal_nan=False)
    fixed_y = np.isclose(prescribed[:, 1], 0.0, equal_nan=False)
    nonzero = np.any(
        np.isfinite(prescribed) & ~np.isclose(prescribed, 0.0, equal_nan=False), axis=1
    )
    categories = (
        (fixed_x & ~fixed_y & ~nonzero, ">", "#e67e22", "x irányban fix"),
        (fixed_y & ~fixed_x & ~nonzero, "^", "#2471a3", "y irányban fix"),
        (fixed_x & fixed_y & ~nonzero, "s", "#c0392b", "x és y irányban fix"),
        (nonzero, "D", "#7d3c98", "előírt nem nulla elmozdulás"),
    )
    for mask, marker, colour, label in categories:
        if np.any(mask):
            points = model.mesh.nodes[mask]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                marker=marker,
                s=42,
                facecolors=colour,
                edgecolors="white",
                linewidths=0.45,
                label=label,
                zorder=4,
            )

    if show_loads:
        loads = model.force_vector.reshape((-1, 2))
        loaded = np.linalg.norm(loads, axis=1) > 0.0
        if np.any(loaded):
            vectors = loads[loaded]
            maximum = np.linalg.norm(vectors, axis=1).max()
            span = np.ptp(model.mesh.nodes, axis=0).max()
            scaled = vectors * (0.08 * span / maximum)
            points = model.mesh.nodes[loaded]
            ax.quiver(
                points[:, 0],
                points[:, 1],
                scaled[:, 0],
                scaled[:, 1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="#17202a",
                width=0.0035,
                label="csomóponti terhelés",
                zorder=5,
            )

    ax.autoscale()
    ax.margins(0.08)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return ax


def _triangles(mesh):
    """A megjelenítéshez minden elemet háromszöglistává alakít."""
    triangles = []
    for element in mesh.elements:
        ids = element.node_ids
        if len(ids) == 3:
            triangles.append(ids)
        elif len(ids) == 4:
            triangles.extend(((ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])))
        else:
            raise ValueError("only Triangle3 and Quad4 elements can be plotted")
    return np.asarray(triangles, dtype=int)


def _draw_undeformed(mesh, ax):
    """Halvány, szaggatott eredeti hálót rajzol referencia-geometriaként."""
    polygons = [mesh.nodes[list(element.node_ids)] for element in mesh.elements]
    ax.add_collection(
        PolyCollection(
            polygons,
            facecolors="none",
            edgecolors="#95a5a6",
            linewidths=0.55,
            linestyles="dashed",
        )
    )


def _add_colour_bar(ax, artist, label: str):
    """A rajztéglalappal pontosan azonos magasságú színskálát készít.

    Az ``axes_grid1`` külön tengelyt fűz a diagram jobb oldalához. Emiatt a
    színskála nem a teljes subplot-, hanem a tényleges axes-magasságot követi,
    még ``aspect='equal'`` esetén is.
    """

    divider = make_axes_locatable(ax)
    colour_axis = divider.append_axes("right", size="3.5%", pad=0.08)
    return ax.figure.colorbar(artist, cax=colour_axis, label=label)


def plot_sparse_matrix(
    matrix,
    *,
    kind: str = "sparsity",
    max_points: int = 250_000,
    title: str = "sparse matrix",
    ax=None,
):
    """Ritka mátrixot jelenít meg sűrűvé alakítás nélkül.

    Args:
        matrix: Tetszőleges SciPy ritka mátrix.
        kind: ``sparsity`` a nemnulla helyekhez, ``magnitude`` a
            ``log10(abs(value))`` színezéshez.
        max_points: A kirajzolható pontok felső korlátja.
        title: Ábracím.
        ax: Opcionális Matplotlib tengely.

    Nagy mátrixnál determinisztikus mintavételezés korlátozza a Matplotlib
    memóriaigényét. A számítási mátrix ettől nem változik meg.
    """

    if not issparse(matrix):
        raise TypeError("matrix must be a SciPy sparse matrix")
    if kind not in {"sparsity", "magnitude"}:
        raise ValueError("kind must be 'sparsity' or 'magnitude'")
    if max_points < 1:
        raise ValueError("max_points must be positive")
    if ax is None:
        _, ax = plt.subplots()
    # A COO sor/oszlop tömbjei közvetlenül használhatók szórásdiagramként;
    # nincs N×N képtömb és nincs toarray() hívás.
    coordinate = matrix.tocoo(copy=False)
    nonzero = coordinate.data != 0.0
    rows = coordinate.row[nonzero]
    columns = coordinate.col[nonzero]
    data = coordinate.data[nonzero]
    original_count = len(data)
    if original_count > max_points:
        selection = np.linspace(0, original_count - 1, max_points, dtype=int)
        rows = rows[selection]
        columns = columns[selection]
        data = data[selection]
    marker_size = max(0.15, min(8.0, 500.0 / np.sqrt(max(matrix.shape))))
    if kind == "sparsity":
        ax.scatter(columns, rows, s=marker_size, c="#17202a", marker="s", linewidths=0)
    else:
        magnitude = np.log10(np.abs(data))
        artist = ax.scatter(
            columns,
            rows,
            s=marker_size,
            c=magnitude,
            cmap="viridis",
            marker="s",
            linewidths=0,
        )
        _add_colour_bar(ax, artist, "log10 |Kij|")
    row_count, column_count = matrix.shape
    ax.set_xlim(-0.5, column_count - 0.5)
    ax.set_ylim(row_count - 0.5, -0.5)
    ax.set_aspect("equal")
    density = matrix.nnz / max(row_count * column_count, 1)
    sampling = f", showing {len(data):,}" if len(data) < original_count else ""
    ax.set_title(
        f"{title}\n{row_count:,} × {column_count:,}, "
        f"nnz={matrix.nnz:,}{sampling}, density={density:.3%}",
        fontsize=10,
    )
    ax.set_xlabel("column degree of freedom")
    ax.set_ylabel("row degree of freedom")
    return ax


def plot_result(
    result,
    *,
    scale: float = 1.0,
    field: str = "von_mises",
    cmap: str = "viridis",
    show_undeformed: bool = True,
    ax=None,
):
    """Színezett eredménymezőt rajzol a skálázott deformált hálóra.

    Az elemmezők cellánként állandó színt kapnak. A csomóponti mezők az
    automatikus háromszög-megjelenítési hálón sima Gouraud-színezést használnak.
    A ``scale`` kizárólag vizuális nagyítás, a számítást nem módosítja.
    """

    cell_fields = {
        "von_mises": result.von_mises,
        "stress_x": result.stress[:, 0],
        "stress_y": result.stress[:, 1],
        "stress_xy": result.stress[:, 2],
    }
    nodal_fields = {
        "displacement_magnitude": result.displacement_magnitude,
        "displacement_x": result.displacement[:, 0],
        "displacement_y": result.displacement[:, 1],
        "nodal_von_mises": result.nodal_von_mises,
        "nodal_stress_x": result.nodal_stress[:, 0],
        "nodal_stress_y": result.nodal_stress[:, 1],
        "nodal_stress_xy": result.nodal_stress[:, 2],
        "principal_stress_1": result.nodal_principal_stress[:, 0],
        "principal_stress_2": result.nodal_principal_stress[:, 1],
    }
    available = sorted(cell_fields | nodal_fields)
    if field not in cell_fields and field not in nodal_fields:
        raise ValueError(f"unknown field {field!r}; choose one of {available}")
    if ax is None:
        _, ax = plt.subplots()
    nodes = result.displaced_nodes(scale)
    if show_undeformed and scale != 0.0:
        _draw_undeformed(result.model.mesh, ax)
    if field in cell_fields:
        polygons = [nodes[list(element.node_ids)] for element in result.model.mesh.elements]
        artist = PolyCollection(
            polygons,
            array=np.asarray(cell_fields[field]),
            cmap=cmap,
            edgecolors="#263238",
            linewidths=0.25,
        )
        ax.add_collection(artist)
    else:
        triangulation = Triangulation(nodes[:, 0], nodes[:, 1], _triangles(result.model.mesh))
        artist = ax.tripcolor(
            triangulation,
            np.asarray(nodal_fields[field]),
            shading="gouraud",
            cmap=cmap,
        )
        polygons = [nodes[list(element.node_ids)] for element in result.model.mesh.elements]
        ax.add_collection(
            PolyCollection(
                polygons,
                facecolors="none",
                edgecolors="#263238",
                linewidths=0.2,
            )
        )
    ax.autoscale()
    ax.set_aspect("equal")
    readable_field = field.replace("_", " ").title()
    ax.set_title(f"{readable_field}\n(displacement ×{scale:g})", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    _add_colour_bar(ax, artist, field)
    return ax


def plot_principal_directions(
    result,
    *,
    scale: float = 1.0,
    stride: int = 1,
    cmap: str = "coolwarm",
    ax=None,
):
    """Az első főfeszültség irányát és előjeles nagyságát nyilakkal rajzolja.

    A ``stride`` minden n-edik csomópont nyilát tartja meg, így sűrű hálón is
    olvasható marad az ábra.
    """

    if stride < 1:
        raise ValueError("stride must be at least 1")
    if ax is None:
        _, ax = plt.subplots()
    if scale != 0.0:
        _draw_undeformed(result.model.mesh, ax)
    all_nodes = result.displaced_nodes(scale)
    polygons = [all_nodes[list(element.node_ids)] for element in result.model.mesh.elements]
    ax.add_collection(
        PolyCollection(
            polygons,
            facecolors="#ecf0f1",
            edgecolors="#7f8c8d",
            linewidths=0.35,
        )
    )
    nodes = all_nodes[::stride]
    angle = result.nodal_principal_angle[::stride]
    magnitude = result.nodal_principal_stress[::stride, 0]
    direction_x = np.cos(angle)
    direction_y = np.sin(angle)
    arrows = ax.quiver(
        nodes[:, 0],
        nodes[:, 1],
        direction_x,
        direction_y,
        magnitude,
        cmap=cmap,
        angles="xy",
        scale_units="xy",
        scale=None,
        pivot="middle",
        width=0.004,
    )
    _add_colour_bar(ax, arrows, "principal_stress_1")
    ax.set_title("First Principal Stress Directions", fontsize=10)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax
