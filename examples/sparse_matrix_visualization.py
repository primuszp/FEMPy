"""Merevségimátrix- és memóriajelentés egy nagy FEMaster2D modellen.

A példa szándékosan külön kéri le a teljes és a redukált CSR-mátrixot, mert
azokat ábrázolja. A normál ``solve()`` ezek közül csak a redukált mátrixot
állítaná össze, közvetlenül és kisebb csúcsmemóriával.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from primfem import PlotStyle, SolverOptions, load_myfem, plot_sparse_matrix
from primfem.solver import sparse_memory_bytes

sample = Path(__file__).with_name("femaster_samples") / "plhole.fem"
model = load_myfem(sample)

# Teljes K az összes szabadságfokkal, Kff pedig csak a szabad szabadságfokokkal.
full_matrix = model.stiffness_matrix()
reduced_matrix = model.stiffness_matrix(reduced=True)

# A CG nem készít ritka LU/Cholesky faktorizációt. A Jacobi-előkondicionáló
# csak a merevségi mátrix diagonálisának inverzét tárolja.
result = model.solve(
    SolverOptions(
        method="cg",
        relative_tolerance=1e-8,
        max_iterations=50_000,
    )
)
style = PlotStyle(language="en")

figure, axes = plt.subplot_mosaic(
    [["full", "reduced"], ["result", "information"]],
    figsize=(16, 12),
    gridspec_kw={"width_ratios": (1.2, 1.0)},
)
figure.subplots_adjust(
    left=0.06,
    right=0.97,
    bottom=0.06,
    top=0.91,
    wspace=0.34,
    hspace=0.32,
)
# A sparsity nézet kizárólag a nemnulla pozíciókat mutatja.
plot_sparse_matrix(
    full_matrix,
    kind="sparsity",
    title="Full stiffness matrix K",
    ax=axes["full"],
    style=style,
)
# A magnitude nézet a nemnulla együtthatók nagyságrendjét színezi.
plot_sparse_matrix(
    reduced_matrix,
    kind="magnitude",
    title="Reduced stiffness matrix Kff",
    ax=axes["reduced"],
    style=style,
)
result.plot(
    scale=result.suggested_deformation_scale(),
    field="nodal_von_mises",
    ax=axes["result"],
    style=style,
)
# Elméleti összehasonlítás: float64 sűrű N×N tömb kontra a CSR három tömbje.
dense_bytes = full_matrix.shape[0] ** 2 * 8
csr_bytes = sparse_memory_bytes(full_matrix)
info = result.solver_info
information = (
    "Sparse system report\n\n"
    f"Degrees of freedom: {full_matrix.shape[0]:,}\n"
    f"Free degrees of freedom: {info.free_dofs:,}\n"
    f"Full matrix nonzeros: {full_matrix.nnz:,}\n"
    f"Full CSR storage: {csr_bytes / 1024**2:.2f} MiB\n"
    f"Equivalent dense storage: {dense_bytes / 1024**2:.2f} MiB\n"
    f"Storage reduction: {dense_bytes / csr_bytes:,.0f}x\n\n"
    f"Selected solver: {info.method.value}\n"
    f"Reduced CSR storage: {info.matrix_memory_megabytes:.2f} MiB\n"
    f"Relative residual: {info.relative_residual:.3e}\n"
    f"Iterations: {info.iterations if info.iterations is not None else 'direct'}"
)
axes["information"].axis("off")
axes["information"].text(
    0.10,
    0.95,
    information,
    transform=axes["information"].transAxes,
    va="top",
    ha="left",
    family="monospace",
    fontsize=12,
    bbox={"boxstyle": "round,pad=0.8", "facecolor": "#f4f6f7", "edgecolor": "#7f8c8d"},
)
figure.suptitle("PrimFEM — sparse stiffness matrix diagnostics", fontsize=16)
output = Path(__file__).with_name("sparse_matrix_visualization.png")
figure.savefig(output, dpi=180)
print(result.summary())
print(information)
print(f"Saved: {output}")
plt.show()
