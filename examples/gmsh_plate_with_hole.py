"""Lyukas lemez modern Gmsh-hálóval és név szerinti peremfeltételekkel."""

from pathlib import Path

import matplotlib.pyplot as plt

from fempy import Geometry2D, GmshMesher, LinearElasticMaterial, Model

# A négy téglalapoldal automatikusan bottom/right/top/left nevet kap. A kör
# külön "hole" perem, amelyhez kisebb helyi elemméretet kérünk.
geometry = (
    Geometry2D("plate_with_hole")
    .add_rectangle(width=100.0, height=50.0)
    .add_circle(center=(50.0, 25.0), radius=8.0, boundary="hole", mesh_size=1.2)
)

# A Gmsh csak opcionális hálógenerátor; telepítés: pip install -e .[gmsh]
mesh = GmshMesher(element_size=4.0, element_shape="triangle").generate(geometry)

steel = LinearElasticMaterial(young_modulus=210_000.0, poisson_ratio=0.3)
model = Model(mesh, steel, thickness=5.0, name="lyukas lemez")

# A peremnevek hálósűrűségtől függetlenek. A jobb oldali megoszló terhet a
# modell konzisztens csomóponti erőkké integrálja.
model.fix_boundary("left")
model.add_boundary_traction("right", ty=-2.0)
result = model.solve()

figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
mesh.plot(ax=axes[0])
axes[0].set_title(f"Gmsh háló: {mesh.node_count} csomópont, {mesh.element_count} elem")
result.plot(
    field="nodal_von_mises",
    scale=result.suggested_deformation_scale(),
    ax=axes[1],
)
axes[1].set_title("von Mises-feszültség")
figure.tight_layout()
output = Path(__file__).with_suffix(".png")
figure.savefig(output, dpi=180)
print(result.summary())
print(f"Ábra: {output}")
