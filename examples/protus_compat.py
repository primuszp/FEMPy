"""Read and solve the original PROTUS model with the unified API."""

from pathlib import Path

from fempy import load_protus

root = Path(__file__).resolve().parents[1]
model = load_protus(root / "tests" / "data" / "protus" / "INPUT_FEA_PROTUS.txt")
result = model.solve()
print(result.summary())
result.write_vtk(Path(__file__).with_name("protus_result.vtk"))
