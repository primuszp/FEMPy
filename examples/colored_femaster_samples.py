"""Coloured gallery of the six imported FEMaster2D sample models."""

from pathlib import Path

import matplotlib.pyplot as plt

from fempy import PlotStyle, load_myfem

sample_directory = Path(__file__).with_name("femaster_samples")
sample_names = (
    "btfemexample.fem",
    "notchedspecimen.fem",
    "plhole.fem",
    "coarse.fem",
    "smallrect.fem",
    "rect.fem",
)

figure, axes = plt.subplots(
    2,
    3,
    figsize=(20, 10),
    gridspec_kw={"height_ratios": (2.0, 1.0)},
)
figure.subplots_adjust(
    left=0.05,
    right=0.98,
    bottom=0.07,
    top=0.89,
    wspace=0.40,
    hspace=0.42,
)
style = PlotStyle(language="hu")
for ax, sample_name in zip(axes.flat, sample_names, strict=True):
    model = load_myfem(sample_directory / sample_name)
    result = model.solve()
    display_scale = result.suggested_deformation_scale(fraction=0.06)
    result.plot(
        scale=display_scale,
        field="nodal_von_mises",
        ax=ax,
        style=style,
    )
    ax.set_title(
        f"{Path(sample_name).stem}\n"
        f"{model.mesh.node_count} csomópont · {model.mesh.element_count} háromszög",
        fontsize=10,
    )
    print(result.summary())

figure.suptitle("FEMaster2D minták egységes FEMPy kiértékelése", fontsize=16)
output = Path(__file__).with_name("colored_femaster_samples.png")
figure.savefig(output, dpi=170)
print(f"Saved: {output}")
plt.show()
