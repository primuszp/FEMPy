"""Adapterek a korábbi PROTUS, Myfem és FEMaster2D szövegformátumokhoz.

Az örökölt fájlok egytől számozzák a csomópontokat, míg az új Python API
nullától. Az importerek ezt automatikusan átalakítják, ellenőrzik a fejlécben
megadott darabszámokat, majd szabályos :class:`primfem.model.Model` objektumot
hoznak létre. A forrásfájlokat soha nem módosítják.
"""

from __future__ import annotations

from pathlib import Path

from .elements import Quad4, Triangle3
from .material import LinearElasticMaterial, PlaneCondition
from .mesh import Mesh
from .model import Model


def _section(lines: list[str], marker: str) -> int:
    """Egy pontos, kis/nagybetű-független komment-szakaszcímet keres."""
    for index, line in enumerate(lines):
        normalized = line.strip().lstrip("#%- ").strip().lower()
        if normalized == marker.lower():
            return index
    raise ValueError(f"missing section marker: {marker}")


def _section_any(lines: list[str], *markers: str) -> int:
    """Több történelmileg használt szakasznév közül megkeresi az elsőt."""
    for marker in markers:
        try:
            return _section(lines, marker)
        except ValueError:
            pass
    raise ValueError(f"missing section marker; expected one of: {', '.join(markers)}")


def _values(lines: list[str], start: int, end: int, comment: str) -> list[str]:
    """Kommentet és üres sorokat eltávolítva visszaadja a tényleges adatsorokat."""
    values = []
    for line in lines[start:end]:
        value = line.split(comment, 1)[0].strip()
        if value:
            values.append(value)
    return values


def load_protus(path: str | Path) -> Model:
    """Beolvassa az eredeti ``INPUT_FEA_PROTUS.txt`` formátumot.

    Megőrzi az anyagot, vastagságot, gravitációt, megtámasztásokat és
    koncentrált terheket. Az elemkapcsolatok egytől induló számait nullától
    induló :class:`Quad4` indexekké alakítja.
    """

    source = Path(path)
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    node_section = _section(lines, "NODE DATA")
    element_section = _section(lines, "ELEMENT DATA")
    boundary_section = _section(lines, "BC, 1-Fixed or 0-Free")
    load_section = _section(lines, "LOAD AND BC")

    # A PROTUS fejléc 11 skalárból áll. A fájl szabad szöveges megjegyzéseit
    # eldobjuk, de az adatok sorrendjét a kompatibilitás miatt megtartjuk.
    header = _values(lines, 0, node_section, "#")
    if len(header) < 11:
        raise ValueError("incomplete PROTUS material/model header")
    analysis_type = int(header[0])
    young_modulus = float(header[1])
    poisson_ratio = float(header[2])
    density = float(header[3])
    thickness = float(header[4])
    acceleration_x = float(header[5])
    acceleration_y = float(header[6])
    expected_nodes = int(header[7])
    expected_elements = int(header[8])
    expected_boundaries = int(header[9])
    expected_loads = int(header[10])

    node_rows = _values(lines, node_section + 1, element_section, "#")
    element_rows = _values(lines, element_section + 1, boundary_section, "#")
    boundary_rows = _values(lines, boundary_section + 1, load_section, "#")
    load_rows = _values(lines, load_section + 1, len(lines), "#")
    # A fejléc és a tényleges szakaszok összevetése korán jelzi a csonka vagy
    # kézzel hibásan módosított bemeneti fájlokat.
    actual = (len(node_rows), len(element_rows), len(boundary_rows), len(load_rows))
    expected = (expected_nodes, expected_elements, expected_boundaries, expected_loads)
    if actual != expected:
        raise ValueError(f"PROTUS row counts {actual} do not match header {expected}")

    nodes = []
    for expected_id, row in enumerate(node_rows, start=1):
        node_id, x, y = (part.strip() for part in row.split(","))
        if int(float(node_id)) != expected_id:
            raise ValueError("PROTUS nodes must be ordered and consecutively numbered")
        nodes.append((float(x), float(y)))
    elements = []
    for row in element_rows:
        values = [int(part.strip()) for part in row.split(",")]
        if len(values) != 5:
            raise ValueError(f"invalid PROTUS Quad4 row: {row}")
        elements.append(Quad4(tuple(node - 1 for node in values[1:])))

    material = LinearElasticMaterial(
        young_modulus,
        poisson_ratio,
        density=density,
        name="PROTUS material",
    )
    model = Model(
        Mesh(nodes, elements),
        material,
        thickness=thickness,
        condition=PlaneCondition.STRESS if analysis_type == 1 else PlaneCondition.STRAIN,
        name=source.stem,
    )
    model.set_body_acceleration(ax=acceleration_x, ay=acceleration_y)
    for row in boundary_rows:
        node, fixed_x, fixed_y = [int(part.strip()) for part in row.split(",")]
        if fixed_x or fixed_y:
            model.fix_node(node - 1, x=bool(fixed_x), y=bool(fixed_y))
    for row in load_rows:
        node, force_x, force_y = [float(part.strip()) for part in row.split(",")]
        model.add_nodal_load(int(node) - 1, fx=force_x, fy=force_y)
    return model


def load_myfem(path: str | Path) -> Model:
    """Beolvassa a Myfem/FEMaster2D háromszög-elemes ``.fem`` formátumát.

    A formátumnak két történelmi terhelésszakasz-neve létezik: ``point loads``
    és ``Point Load Definition``. Mindkettő támogatott, ezért a teljes átvett
    FEMaster2D mintakészlet változtatás nélkül használható.
    """

    source = Path(path)
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    node_section = _section(lines, "node definition")
    element_section = _section(lines, "element definition")
    load_section = _section_any(lines, "point loads", "point load definition")
    support_section = _section(lines, "supports")
    header_rows = _values(lines, 0, node_section, "%")
    if len(header_rows) != 1:
        raise ValueError("Myfem header must contain exactly one data row")
    header = [float(part.strip()) for part in header_rows[0].split(",")]
    if len(header) != 7:
        raise ValueError("Myfem header must contain nn, ne, np, ns, E, Nu, Thk")
    node_count, element_count, load_count, support_count = map(int, header[:4])
    node_rows = _values(lines, node_section + 1, element_section, "%")
    element_rows = _values(lines, element_section + 1, load_section, "%")
    load_rows = _values(lines, load_section + 1, support_section, "%")
    support_rows = _values(lines, support_section + 1, len(lines), "%")
    if (len(node_rows), len(element_rows), len(load_rows), len(support_rows)) != (
        node_count,
        element_count,
        load_count,
        support_count,
    ):
        raise ValueError("Myfem section row counts do not match the header")

    nodes = []
    for expected_id, row in enumerate(node_rows, start=1):
        node_id, x, y = [float(part.strip()) for part in row.split(",")]
        if int(node_id) != expected_id:
            raise ValueError("Myfem nodes must be ordered and consecutively numbered")
        nodes.append((x, y))
    elements = []
    for row in element_rows:
        _, n1, n2, n3 = [int(part.strip()) for part in row.split(",")]
        elements.append(Triangle3((n1 - 1, n2 - 1, n3 - 1)))
    model = Model(
        Mesh(nodes, elements),
        LinearElasticMaterial(header[4], header[5], name="Myfem material"),
        thickness=header[6],
        name=source.stem,
    )
    for row in load_rows:
        _, node, force_x, force_y = [float(part.strip()) for part in row.split(",")]
        model.add_nodal_load(int(node) - 1, fx=force_x, fy=force_y)
    for row in support_rows:
        _, node, fixed_x, fixed_y = [int(part.strip()) for part in row.split(",")]
        model.fix_node(node - 1, x=bool(fixed_x), y=bool(fixed_y))
    return model
