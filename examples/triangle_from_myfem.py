"""Run the original Myfem triangular input through the unified solver."""

from pathlib import Path

from fempy import load_myfem

root = Path(__file__).resolve().parents[1]
model = load_myfem(root / "tests" / "data" / "myfem" / "coarse.fem")
result = model.solve()
print(result.summary())
result.write_vtk(Path(__file__).with_name("myfem_result.vtk"))
