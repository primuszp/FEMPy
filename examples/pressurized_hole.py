"""Furat belső nyomása: hálózás, peremfeltétel, megoldás és kiértékelés."""

from pathlib import Path

import matplotlib.pyplot as plt

from primfem import Geometry2D, GmshMesher, LinearElasticMaterial, Model, PlotStyle

geometry = (
    Geometry2D("pressurized_plate")
    .add_rectangle(width=100.0, height=50.0)
    .add_circle((50.0, 25.0), radius=8.0, boundary="hole", mesh_size=1.2)
)
mesh = GmshMesher(element_size=4.0, element_shape="triangle").generate(geometry)

model = Model(
    mesh,
    LinearElasticMaterial(young_modulus=210_000.0, poisson_ratio=0.3, name="steel"),
    thickness=5.0,
    name="pressurized hole",
)
model.fix_boundary("left")
model.add_boundary_pressure("hole", pressure=10.0)

result = model.solve()
scale = result.suggested_deformation_scale(fraction=0.06)

style = PlotStyle(language="hu", length_unit="mm", stress_unit="MPa")
figure, axes = plt.subplots(1, 2, figsize=(13, 5.4))
figure.subplots_adjust(left=0.07, right=0.94, bottom=0.13, top=0.86, wspace=0.43)
model.plot_boundary_conditions(ax=axes[0], style=style)
axes[0].set_title("Megtámasztás és a nyomás csomóponti erői")
result.plot(field="nodal_von_mises", scale=scale, ax=axes[1], style=style)
output = Path(__file__).with_suffix(".png")
figure.savefig(output, dpi=180, bbox_inches="tight")
result.write_vtk(Path(__file__).with_suffix(".vtk"))
print(result.summary())
print(f"Ábra: {output}")
