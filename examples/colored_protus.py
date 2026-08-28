"""Coloured post-processing of the original PROTUS input model."""

from pathlib import Path

import matplotlib.pyplot as plt

from fempy import load_protus

root = Path(__file__).resolve().parents[1]
model = load_protus(root / "tests" / "data" / "protus" / "INPUT_FEA_PROTUS.txt")
result = model.solve()

figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
result.plot(scale=30.0, field="displacement_magnitude", cmap="magma", ax=axes[0, 0])
result.plot(scale=30.0, field="nodal_von_mises", cmap="turbo", ax=axes[0, 1])
result.plot(scale=30.0, field="principal_stress_1", cmap="coolwarm", ax=axes[1, 0])
result.plot_principal_directions(scale=30.0, stride=4, ax=axes[1, 1])
figure.suptitle("Original PROTUS model — FEMPy coloured results", fontsize=16)
output = Path(__file__).with_name("colored_protus.png")
figure.savefig(output, dpi=180)
print(result.summary())
print(f"Saved: {output}")
plt.show()
