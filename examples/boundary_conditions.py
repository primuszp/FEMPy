"""Egy- és kétirányú megtámasztások vizualizálása névvel ellátott peremeken."""

from pathlib import Path

import matplotlib.pyplot as plt

from fempy import Geometry2D, GmshMesher, LinearElasticMaterial, Model

geometry = Geometry2D("supports").add_rectangle(width=100.0, height=50.0)
mesh = GmshMesher(element_size=7.0, element_shape="quad").generate(geometry)
model = Model(mesh, LinearElasticMaterial(210_000.0, 0.3), thickness=5.0)

# Bal: kétirányú befogás. Alul: csak y, jobb oldalon: csak x irány fix.
model.fix_boundary("left", x=True, y=True)
model.fix_boundary("bottom", x=False, y=True)
model.fix_boundary("right", x=True, y=False)

# Egy koncentrált erő, hogy a terhelésnyíl megjelenítése is látható legyen.
top_nodes = mesh.boundary_nodes("top")
model.add_nodal_load(top_nodes[len(top_nodes) // 2], fy=-1000.0)

figure, axis = plt.subplots(figsize=(11, 5.8))
model.plot_boundary_conditions(ax=axis)
axis.set_title("Kinematikai peremfeltételek és csomóponti terhelés")
figure.tight_layout()
output = Path(__file__).with_suffix(".png")
figure.savefig(output, dpi=190, bbox_inches="tight")
print(f"Ábra: {output}")
