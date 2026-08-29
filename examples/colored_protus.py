"""Coloured post-processing of the original PROTUS input model."""

from pathlib import Path

import matplotlib.pyplot as plt

from primfem import PlotStyle, load_protus

root = Path(__file__).resolve().parents[1]
model = load_protus(root / "tests" / "data" / "protus" / "INPUT_FEA_PROTUS.txt")
result = model.solve()

style = PlotStyle(language="hu")
figure, axes = plt.subplots(2, 2, figsize=(14, 10))
figure.subplots_adjust(left=0.07, right=0.95, bottom=0.07, top=0.88, wspace=0.38, hspace=0.46)
result.plot(scale=30.0, field="displacement_magnitude", ax=axes[0, 0], style=style)
result.plot(scale=30.0, field="nodal_von_mises", ax=axes[0, 1], style=style)
result.plot(scale=30.0, field="principal_stress_1", ax=axes[1, 0], style=style)
result.plot_principal_directions(scale=30.0, stride=4, ax=axes[1, 1], style=style)
figure.suptitle("Az eredeti PROTUS modell egységes PrimFEM ábrái", fontsize=16)
output = Path(__file__).with_name("colored_protus.png")
figure.savefig(output, dpi=180)
print(result.summary())
print(f"Saved: {output}")
plt.show()
