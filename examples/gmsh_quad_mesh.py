"""Tiszta Quad4 háló szemléltetése kör alakú furattal."""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection

from primfem import Geometry2D, GmshMesher

geometry = (
    Geometry2D("quad_plate_with_hole")
    .add_rectangle(width=100.0, height=50.0)
    .add_circle(
        center=(50.0, 25.0),
        radius=8.0,
        boundary="hole",
        mesh_size=2.0,
    )
)

mesh = GmshMesher(
    element_size=5.0,
    element_shape="quad",
).generate(geometry)

counts = Counter(type(element).__name__ for element in mesh.elements)
polygons = [mesh.nodes[list(element.node_ids)] for element in mesh.elements]
colours = ["#dceeff" if index % 2 == 0 else "#c6e2f5" for index in range(mesh.element_count)]

figure, axis = plt.subplots(figsize=(11, 5.8))
axis.add_collection(
    PolyCollection(
        polygons,
        facecolors=colours,
        edgecolors="#274c77",
        linewidths=0.85,
    )
)

# A furat név szerint visszanyert pereméleit külön színnel emeljük ki.
hole_edges = [mesh.nodes[list(edge)] for edge in mesh.boundary_edges("hole")]
axis.add_collection(LineCollection(hole_edges, colors="#d1495b", linewidths=2.2))

axis.autoscale()
axis.margins(0.04)
axis.set_aspect("equal")
axis.set_xlabel("x")
axis.set_ylabel("y")
axis.set_title(
    "Gmsh négyszögháló kör alakú furattal\n"
    f"{mesh.node_count} csomópont · {counts.get('Quad4', 0)} Quad4 · "
    f"{counts.get('Triangle3', 0)} Triangle3"
)
axis.grid(False)
figure.tight_layout()

output = Path(__file__).with_suffix(".png")
figure.savefig(output, dpi=190, bbox_inches="tight")

boundary_figure, boundary_axis = plt.subplots(figsize=(11, 5.8))
mesh.plot_boundaries(ax=boundary_axis)
boundary_axis.set_title("A geometriában elnevezett peremek a kész hálón")
boundary_figure.tight_layout()
boundary_output = Path(__file__).with_name("gmsh_quad_boundaries.png")
boundary_figure.savefig(boundary_output, dpi=190, bbox_inches="tight")

print(f"Elemeloszlás: {dict(counts)}")
print(f"Ábra: {output}")
print(f"Peremábra: {boundary_output}")
