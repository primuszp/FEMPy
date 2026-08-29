"""T6 háló és egységes magyar/angol tudományos eredményábrák.

A példa ugyanazt a megoldást két nyelvi stílussal rajzolja ki. Magyar módban
a tengelyek és a színskála tizedesvesszőt, angol módban tizedespontot használ.
A feszültségmező szabványos jelölést és explicit MPa mértékegységet kap.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from primfem import Geometry2D, GmshMesher, LinearElasticMaterial, Model, PlotStyle

geometry = (
    Geometry2D("T6 lyukas lemez")
    .add_rectangle(100.0, 50.0)
    .add_circle((50.0, 25.0), 10.0, boundary="hole", mesh_size=3.0)
)
mesh = GmshMesher(element_size=6.0, element_shape="triangle", order=2).generate(geometry)

steel = LinearElasticMaterial(young_modulus=210_000.0, poisson_ratio=0.3)
model = Model(mesh, steel, thickness=5.0, name="T6 plate with a pressurized hole")
model.fix_boundary("left")
model.add_boundary_pressure("hole", pressure=10.0)
result = model.solve()
deformation_scale = result.suggested_deformation_scale(fraction=0.08)


def save_report(style: PlotStyle, filename: str) -> None:
    """Hárompaneles, egységes stílusú mérnöki ábrát ment."""

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(19, 5.8),
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.12)},
    )
    figure.subplots_adjust(left=0.05, right=0.94, bottom=0.12, top=0.82, wspace=0.46)
    mesh.plot(ax=axes[0], style=style)
    model.plot_boundary_conditions(ax=axes[1], style=style)
    axes[1].legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    result.plot(
        field="nodal_von_mises",
        scale=deformation_scale,
        ax=axes[2],
        style=style,
    )
    for axis in axes:
        axis.set_yticks((0.0, 12.5, 25.0, 37.5, 50.0))
    title = style.text(
        "Másodrendű T6 végeselemes elemzés",
        "Second-order T6 finite element analysis",
    )
    figure.suptitle(title, fontsize=16, fontweight="semibold")
    output = Path(__file__).with_name(filename)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    print(f"Saved: {output}")


save_report(
    PlotStyle(language="hu", length_unit="mm", stress_unit="MPa"),
    "t6_report_hu.png",
)
save_report(
    PlotStyle(language="en", length_unit="mm", stress_unit="MPa"),
    "t6_report_en.png",
)
print(result.summary())
