"""A PrimFEM teljes klasszikus verifikációs csomagjának futtatása."""

from pathlib import Path

import matplotlib.pyplot as plt

from primfem import run_classic_validations

directory = Path(__file__).resolve().parent
report = run_classic_validations()
print(report.summary())

report.write_markdown(directory / "classic_validation_results.md")
figure, axis = plt.subplots(figsize=(10.5, 6.0))
report.plot(ax=axis)
figure.tight_layout()
figure.savefig(directory / "classic_validation_convergence.png", dpi=190)

if not report.passed:
    raise SystemExit("A klasszikus FEM validáció legalább egy esete megbukott.")
