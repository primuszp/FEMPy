"""Matplotlib-alapú, oktatási célú végeselemes vizualizációk.

A modul hálót, deformált eredménymezőt, főfeszültségi irányt és ritka mátrixot
rajzol. A csomóponti mezők belső háromszög-felbontáson Gouraud-színezést kapnak;
az elemmezők elemenként állandó színűek. A színskála mindig pontosan a
rajztéglalap magasságát követi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse import issparse

from .elements import Triangle6


@dataclass(frozen=True, slots=True)
class PlotStyle:
    """Egységes nyelvi és mérnöki megjelenítési beállítás.

    Args:
        language: ``"hu"`` esetén magyar felirat és tizedesvessző,
            ``"en"`` esetén angol felirat és tizedespont.
        length_unit: A koordináták mértékegysége, például ``"mm"``.
        displacement_unit: Az elmozdulás mértékegysége; alapértelmezetten a
            ``length_unit`` értékét örökli.
        stress_unit: A feszültség mértékegysége, például ``"MPa"``.
        precision: Jelentős számjegyek száma a tengelyeken és színskálákon.
        engineering_scaling: Nagy vagy kis mezőértékek közös ``10^(3n)``
            mérnöki kitevővel jelenjenek meg.
    """

    language: Literal["hu", "en"] = "hu"
    length_unit: str | None = None
    displacement_unit: str | None = None
    stress_unit: str | None = None
    precision: int = 4
    engineering_scaling: bool = True

    def __post_init__(self) -> None:
        if self.language not in ("hu", "en"):
            raise ValueError("plot language must be 'hu' or 'en'")
        if not 2 <= self.precision <= 10:
            raise ValueError("plot precision must be between 2 and 10")

    @property
    def effective_displacement_unit(self) -> str | None:
        return self.displacement_unit or self.length_unit

    def text(self, hungarian: str, english: str) -> str:
        """A beállított nyelvnek megfelelő szöveget választja ki."""

        return hungarian if self.language == "hu" else english

    def number(self, value: float) -> str:
        """Lebegőpontos számot lokalizált, szükség esetén tudományos alakban ír."""

        if np.isclose(value, 0.0, atol=10.0 ** (-self.precision - 2)):
            value = 0.0
        text = f"{value:.{self.precision}g}"
        if "e" in text:
            mantissa, exponent = text.split("e")
            mantissa = mantissa.replace(".", "{,}") if self.language == "hu" else mantissa
            return rf"${mantissa}\times10^{{{int(exponent)}}}$"
        return text.replace(".", ",") if self.language == "hu" else text

    def integer(self, value: int) -> str:
        """Egész számot nyelvhelyes ezres tagolással formáz."""

        separator = " " if self.language == "hu" else ","
        return f"{value:,}".replace(",", separator)


DEFAULT_PLOT_STYLE = PlotStyle()


def _style(style: PlotStyle | None) -> PlotStyle:
    return DEFAULT_PLOT_STYLE if style is None else style


def _format_axes(ax, style: PlotStyle, *, equal: bool = True) -> None:
    """Egységes tipográfiát, rácsot és lokalizált tengelyszámokat alkalmaz."""

    formatter = FuncFormatter(lambda value, _position: style.number(value))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_facecolor("#fbfcfe")
    ax.grid(True, color="#d9e0e7", linewidth=0.55, alpha=0.55, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#5d6d7e")
        spine.set_linewidth(0.8)
    if equal:
        ax.set_aspect("equal")
    coordinate_unit = f" [{style.length_unit}]" if style.length_unit else ""
    ax.set_xlabel(f"x{coordinate_unit}")
    ax.set_ylabel(f"y{coordinate_unit}")


def _outline_node_ids(element) -> tuple[int, ...]:
    """Rajzolási körüljárást ad; a T6 tárolási sorrendje önmagában nem poligon."""

    ids = element.node_ids
    if isinstance(element, Triangle6):
        return (ids[0], ids[3], ids[1], ids[4], ids[2], ids[5])
    return ids


def _polygons(mesh, nodes=None):
    coordinates = mesh.nodes if nodes is None else nodes
    return [coordinates[list(_outline_node_ids(element))] for element in mesh.elements]


def plot_mesh(mesh, *, ax=None, show_node_ids: bool = False, style: PlotStyle | None = None):
    """Deformálatlan hálót rajzol, opcionális csomópontszámokkal."""

    if ax is None:
        _, ax = plt.subplots()
    style = _style(style)
    polygons = _polygons(mesh)
    collection = PolyCollection(polygons, facecolors="none", edgecolors="#34495e", linewidths=0.8)
    ax.add_collection(collection)
    if show_node_ids:
        for index, (x, y) in enumerate(mesh.nodes):
            ax.text(x, y, str(index), fontsize=8, color="#c0392b")
    ax.autoscale()
    ax.margins(0.05)
    _format_axes(ax, style)
    ax.set_title(style.text("Végeselemes háló", "Finite element mesh"), fontweight="semibold")
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
    style: PlotStyle | None = None,
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
    style = _style(style)
    selected = list(mesh.boundary_names if names is None else names)
    if not selected:
        raise ValueError("the mesh has no named boundaries to plot")
    unknown = [name for name in selected if name not in mesh.boundary_names]
    if unknown:
        raise KeyError(f"unknown boundaries {unknown}; available: {', '.join(mesh.boundary_names)}")
    if show_mesh:
        polygons = _polygons(mesh)
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
    _format_axes(ax, style)
    ax.set_title(style.text("Elnevezett peremek", "Named boundaries"), fontweight="semibold")
    ax.legend(
        title=style.text("Peremek", "Boundaries"),
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
    )
    return ax


def plot_boundary_conditions(
    model,
    *,
    ax=None,
    show_mesh: bool = True,
    show_loads: bool = True,
    style: PlotStyle | None = None,
):
    """A modell kinematikai peremfeltételeit könnyen olvashatóan ábrázolja.

    A narancssárga jobbra mutató háromszög az x, a kék felfelé mutató
    háromszög az y irányú rögzítést jelenti. A piros négyzet mindkét irány
    rögzítését mutatja. Nem nulla előírt elmozdulás lila rombuszként jelenik
    meg. Az opcionális fekete nyilak a koncentrált csomóponti terhek.
    """

    if ax is None:
        _, ax = plt.subplots()
    style = _style(style)
    if show_mesh:
        polygons = _polygons(model.mesh)
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
        (fixed_x & ~fixed_y & ~nonzero, ">", "#e67e22", style.text("x irányban fix", "fixed in x")),
        (fixed_y & ~fixed_x & ~nonzero, "^", "#2471a3", style.text("y irányban fix", "fixed in y")),
        (
            fixed_x & fixed_y & ~nonzero,
            "s",
            "#c0392b",
            style.text("x és y irányban fix", "fixed in x and y"),
        ),
        (
            nonzero,
            "D",
            "#7d3c98",
            style.text("előírt nem nulla elmozdulás", "prescribed nonzero displacement"),
        ),
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
                label=style.text("csomóponti terhelés", "nodal load"),
                zorder=5,
            )

    ax.autoscale()
    ax.margins(0.08)
    _format_axes(ax, style)
    ax.set_title(
        style.text("Peremfeltételek és terhelések", "Boundary conditions and loads"),
        fontweight="semibold",
    )
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
        elif isinstance(element, Triangle6):
            # Négy lineáris megjelenítési háromszög használja mind a hat T6
            # csomópontot; így a kvadratikus mező középcsomópontjai sem vesznek el.
            triangles.extend(
                (
                    (ids[0], ids[3], ids[5]),
                    (ids[3], ids[1], ids[4]),
                    (ids[5], ids[4], ids[2]),
                    (ids[3], ids[4], ids[5]),
                )
            )
        elif len(ids) == 4:
            triangles.extend(((ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])))
        else:
            raise ValueError("only Triangle3, Triangle6 and Quad4 elements can be plotted")
    return np.asarray(triangles, dtype=int)


def _draw_undeformed(mesh, ax):
    """Halvány, szaggatott eredeti hálót rajzol referencia-geometriaként."""
    polygons = _polygons(mesh)
    ax.add_collection(
        PolyCollection(
            polygons,
            facecolors="none",
            edgecolors="#95a5a6",
            linewidths=0.55,
            linestyles="dashed",
        )
    )


def _add_colour_bar(ax, artist, label: str, style: PlotStyle):
    """A rajztéglalappal pontosan azonos magasságú színskálát készít.

    Az ``axes_grid1`` külön tengelyt fűz a diagram jobb oldalához. Emiatt a
    színskála nem a teljes subplot-, hanem a tényleges axes-magasságot követi,
    még ``aspect='equal'`` esetén is.
    """

    divider = make_axes_locatable(ax)
    colour_axis = divider.append_axes("right", size="3.5%", pad=0.08)
    colour_bar = ax.figure.colorbar(artist, cax=colour_axis)
    colour_bar.set_label(label, fontweight="semibold")
    colour_bar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: style.number(value))
    )
    return colour_bar


def plot_sparse_matrix(
    matrix,
    *,
    kind: str = "sparsity",
    max_points: int = 250_000,
    title: str = "sparse matrix",
    ax=None,
    style: PlotStyle | None = None,
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
    style = _style(style)
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
        _add_colour_bar(ax, artist, r"$\log_{10}|K_{ij}|$", style)
    row_count, column_count = matrix.shape
    ax.set_xlim(-0.5, column_count - 0.5)
    ax.set_ylim(row_count - 0.5, -0.5)
    ax.set_aspect("equal")
    density = matrix.nnz / max(row_count * column_count, 1)
    sampling = (
        style.text(
            f", megjelenítve: {style.integer(len(data))}",
            f", showing {style.integer(len(data))}",
        )
        if len(data) < original_count
        else ""
    )
    density_text = style.number(100.0 * density)
    ax.set_title(
        f"{title}\n{style.integer(row_count)} × {style.integer(column_count)}, "
        f"nnz={style.integer(matrix.nnz)}{sampling}, "
        f"{style.text('kitöltöttség', 'density')}={density_text}%",
        fontsize=10,
        fontweight="semibold",
    )
    _format_axes(ax, style, equal=True)
    ax.set_xlabel(style.text("oszlop szabadságfok", "column degree of freedom"))
    ax.set_ylabel(style.text("sor szabadságfok", "row degree of freedom"))
    return ax


_FIELD_INFO = {
    "von_mises": (
        "von Mises-egyenértékfeszültség",
        "von Mises equivalent stress",
        r"$\sigma_\mathrm{vM}$",
        "stress",
        False,
    ),
    "stress_x": ("x irányú normálfeszültség", "x normal stress", r"$\sigma_x$", "stress", True),
    "stress_y": ("y irányú normálfeszültség", "y normal stress", r"$\sigma_y$", "stress", True),
    "stress_xy": (
        "síkbeli nyírófeszültség",
        "in-plane shear stress",
        r"$\tau_{xy}$",
        "stress",
        True,
    ),
    "principal_stress_1": (
        "első főfeszültség",
        "first principal stress",
        r"$\sigma_1$",
        "stress",
        True,
    ),
    "principal_stress_2": (
        "második főfeszültség",
        "second principal stress",
        r"$\sigma_2$",
        "stress",
        True,
    ),
    "displacement_magnitude": (
        "elmozdulás nagysága",
        "displacement magnitude",
        r"$|\mathbf{u}|$",
        "displacement",
        False,
    ),
    "displacement_x": ("x irányú elmozdulás", "x displacement", r"$u_x$", "displacement", True),
    "displacement_y": ("y irányú elmozdulás", "y displacement", r"$u_y$", "displacement", True),
}


def _field_info(field: str):
    """A csomóponti előtagtól független tudományos mezőmetaadatot ad."""

    return _FIELD_INFO[field.removeprefix("nodal_")]


def _scaled_field(values, field: str, style: PlotStyle):
    """Közös mérnöki kitevőre skálázott mezőt és szakszerű címkét készít."""

    array = np.asarray(values, dtype=float)
    hungarian, english, symbol, quantity, signed = _field_info(field)
    maximum = float(np.max(np.abs(array), initial=0.0))
    exponent = 0
    if style.engineering_scaling and maximum > 0.0 and (maximum >= 1.0e4 or maximum < 1.0e-2):
        exponent = 3 * int(np.floor(np.log10(maximum) / 3.0))
    displayed = array / (10.0**exponent)
    unit = style.stress_unit if quantity == "stress" else style.effective_displacement_unit
    multiplier = rf"$\times 10^{{{exponent}}}$ " if exponent else ""
    unit_text = f"{multiplier}{unit}" if unit else multiplier.rstrip()
    colour_label = f"{symbol} [{unit_text}]" if unit_text else symbol
    return displayed, style.text(hungarian, english), colour_label, signed


def _field_norm(values: np.ndarray, signed: bool):
    """Előjeles mezőnél nullaközepű, egyébként adatvezérelt normálást ad."""

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if np.isclose(minimum, maximum):
        padding = max(abs(minimum), 1.0) * 1.0e-9
        return Normalize(vmin=minimum - padding, vmax=maximum + padding)
    if signed and minimum < 0.0 < maximum:
        limit = max(abs(minimum), abs(maximum))
        return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    return Normalize(vmin=minimum, vmax=maximum)


def plot_result(
    result,
    *,
    scale: float = 1.0,
    field: str = "von_mises",
    cmap: str | None = None,
    show_undeformed: bool = True,
    ax=None,
    style: PlotStyle | None = None,
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
    style = _style(style)
    nodes = result.displaced_nodes(scale)
    if show_undeformed and scale != 0.0:
        _draw_undeformed(result.model.mesh, ax)
    if field in cell_fields:
        values, readable_field, colour_label, signed = _scaled_field(
            cell_fields[field], field, style
        )
        polygons = _polygons(result.model.mesh, nodes)
        selected_cmap = cmap or ("coolwarm" if signed else "viridis")
        artist = PolyCollection(
            polygons,
            array=values,
            cmap=selected_cmap,
            norm=_field_norm(values, signed),
            edgecolors="#263238",
            linewidths=0.25,
        )
        ax.add_collection(artist)
    else:
        values, readable_field, colour_label, signed = _scaled_field(
            nodal_fields[field], field, style
        )
        selected_cmap = cmap or ("coolwarm" if signed else "viridis")
        triangulation = Triangulation(nodes[:, 0], nodes[:, 1], _triangles(result.model.mesh))
        artist = ax.tripcolor(
            triangulation,
            values,
            shading="gouraud",
            cmap=selected_cmap,
            norm=_field_norm(values, signed),
        )
        polygons = _polygons(result.model.mesh, nodes)
        ax.add_collection(
            PolyCollection(
                polygons,
                facecolors="none",
                edgecolors="#263238",
                linewidths=0.2,
            )
        )
    ax.autoscale()
    ax.margins(0.04)
    scale_text = style.number(scale)
    subtitle = style.text(
        f"deformáció nagyítása: {scale_text}×",
        f"deformation scale: {scale_text}×",
    )
    ax.set_title(f"{readable_field}\n{subtitle}", fontsize=10, fontweight="semibold")
    _format_axes(ax, style)
    _add_colour_bar(ax, artist, colour_label, style)
    return ax


def plot_principal_directions(
    result,
    *,
    scale: float = 1.0,
    stride: int = 1,
    cmap: str = "coolwarm",
    ax=None,
    style: PlotStyle | None = None,
):
    """Az első főfeszültség irányát és előjeles nagyságát nyilakkal rajzolja.

    A ``stride`` minden n-edik csomópont nyilát tartja meg, így sűrű hálón is
    olvasható marad az ábra.
    """

    if stride < 1:
        raise ValueError("stride must be at least 1")
    if ax is None:
        _, ax = plt.subplots()
    style = _style(style)
    if scale != 0.0:
        _draw_undeformed(result.model.mesh, ax)
    all_nodes = result.displaced_nodes(scale)
    polygons = _polygons(result.model.mesh, all_nodes)
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
    magnitude, _, colour_label, signed = _scaled_field(
        result.nodal_principal_stress[::stride, 0], "principal_stress_1", style
    )
    direction_x = np.cos(angle)
    direction_y = np.sin(angle)
    arrows = ax.quiver(
        nodes[:, 0],
        nodes[:, 1],
        direction_x,
        direction_y,
        magnitude,
        cmap=cmap,
        norm=_field_norm(magnitude, signed),
        angles="xy",
        scale_units="xy",
        scale=None,
        pivot="middle",
        width=0.004,
    )
    _add_colour_bar(ax, arrows, colour_label, style)
    ax.set_title(
        style.text("Első főfeszültségi irányok", "First principal stress directions"),
        fontsize=10,
        fontweight="semibold",
    )
    ax.autoscale()
    _format_axes(ax, style)
    return ax
