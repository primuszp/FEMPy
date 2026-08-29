"""Ritka lineáris egyenletmegoldók beállításai és diagnosztikája.

A végeselemes redukált merevségi mátrix szimmetrikus pozitív definit, ha a
modell megfelelően meg van támasztva. Kis és közepes rendszernél a SciPy ritka
direkt megoldója robusztus. Nagy rendszernél a faktorizáció kitöltése sok
memóriát igényelhet, ezért rendelkezésre áll a konjugáltgradiens-módszer (CG)
Jacobi-előkondicionálóval. A CG csak a CSR-mátrixot és néhány vektort tárol.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, MatrixRankWarning, cg, splu


class SolverMethod(str, Enum):
    """A redukált egyenletrendszer választható megoldási módszerei."""

    AUTO = "auto"
    DIRECT = "direct"
    CONJUGATE_GRADIENT = "cg"


class Preconditioner(str, Enum):
    """Az iteratív solver memóriatakarékos előkondicionálói."""

    NONE = "none"
    JACOBI = "jacobi"


@dataclass(slots=True)
class SolverOptions:
    """A ritka egyenletmegoldás felhasználói beállításai.

    Args:
        method: ``auto``, ``direct`` vagy ``cg``.
        relative_tolerance: A maradék jobb-oldalhoz viszonyított tűrése.
        absolute_tolerance: Abszolút maradéktűrés; általában nulla maradhat.
        max_iterations: CG maximális iterációszáma. ``None`` a SciPy alapértéke.
        preconditioner: ``jacobi`` vagy ``none``.
        direct_dof_limit: ``auto`` módban e szabadságfokszám fölött CG indul.

    Az automatikus mód közepes rendszernél ritka direkt megoldást, a küszöb
    fölött előkondicionált CG-t választ. Ez a küszöb projektenként hangolható.
    """

    method: SolverMethod | str = SolverMethod.AUTO
    relative_tolerance: float = 1e-10
    absolute_tolerance: float = 0.0
    max_iterations: int | None = None
    preconditioner: Preconditioner | str = Preconditioner.JACOBI
    direct_dof_limit: int = 20_000

    def __post_init__(self) -> None:
        self.method = SolverMethod(self.method)
        self.preconditioner = Preconditioner(self.preconditioner)
        if not np.isfinite(self.relative_tolerance) or self.relative_tolerance <= 0.0:
            raise ValueError("relative_tolerance must be positive and finite")
        if not np.isfinite(self.absolute_tolerance) or self.absolute_tolerance < 0.0:
            raise ValueError("absolute_tolerance must be non-negative and finite")
        if self.max_iterations is not None and self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.direct_dof_limit < 1:
            raise ValueError("direct_dof_limit must be positive")


@dataclass(frozen=True, slots=True)
class SolverInfo:
    """Egy befejezett megoldás megváltoztathatatlan diagnosztikai adatai.

    A ``matrix_memory_bytes`` kizárólag a CSR három tömbjének tényleges mérete.
    Direkt solver esetén a belső faktorizáció további memóriát használhat; CG
    esetén ilyen nagy faktorizáció nincs.
    """

    method: SolverMethod
    free_dofs: int
    nonzero_entries: int
    matrix_memory_bytes: int
    iterations: int | None
    residual_norm: float
    relative_residual: float
    factorization_reused: bool = False

    @property
    def matrix_memory_megabytes(self) -> float:
        """A redukált CSR-mátrix tényleges tárigénye bináris MiB egységben."""
        return self.matrix_memory_bytes / (1024.0**2)


class SolverConvergenceError(RuntimeError):
    """Akkor keletkezik, ha az iteratív solver nem konvergál."""


def sparse_memory_bytes(matrix: csr_matrix) -> int:
    """Megadja a CSR ``data``, ``indices`` és ``indptr`` tömbjeinek összméretét."""

    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


class SparseDirectFactorization:
    """Újrafelhasználható ritka LU-faktorizáció több jobb oldalhoz.

    Az objektum egyetlen, változatlan redukált merevségi mátrixhoz tartozik.
    Tipikus felhasználása több olyan terhelési eset megoldása, amelyek hálója,
    anyaga, vastagsága és kötött szabadságfokai azonosak.
    """

    def __init__(self, matrix: csr_matrix) -> None:
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("a sparse factorization requires a square matrix")
        try:
            # A SuperLU oszloporientált CSC formátumot vár. Az átalakítás csak
            # egyszer történik, utána tetszőleges számú jobb oldal oldható meg.
            self._factor = splu(matrix.tocsc())
        except RuntimeError as exc:
            raise ValueError(
                "the stiffness matrix is singular; check supports and connectivity"
            ) from exc
        self.shape = matrix.shape

    def solve(self, right_hand_side: np.ndarray) -> np.ndarray:
        """Megoldja a korábban faktorizált rendszert egy új jobb oldallal."""

        values = np.asarray(right_hand_side, dtype=float)
        if values.shape != (self.shape[0],):
            raise ValueError("right-hand side has an incompatible shape")
        solution = np.asarray(self._factor.solve(values), dtype=float)
        if not np.all(np.isfinite(solution)):
            raise ValueError("the sparse solution contains non-finite values")
        return solution


def _solver_info(
    matrix: csr_matrix,
    right_hand_side: np.ndarray,
    solution: np.ndarray,
    method: SolverMethod,
    *,
    iterations: int | None = None,
    factorization_reused: bool = False,
) -> SolverInfo:
    """Egységes solverdiagnosztikát készít bármely megoldási út eredményéhez."""

    residual = np.asarray(matrix @ solution - right_hand_side)
    residual_norm = float(np.linalg.norm(residual))
    rhs_norm = float(np.linalg.norm(right_hand_side))
    relative_residual = residual_norm / rhs_norm if rhs_norm > 0.0 else residual_norm
    return SolverInfo(
        method=method,
        free_dofs=matrix.shape[0],
        nonzero_entries=matrix.nnz,
        matrix_memory_bytes=sparse_memory_bytes(matrix),
        iterations=iterations,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        factorization_reused=factorization_reused,
    )


def selected_solver_method(options: SolverOptions, dof_count: int) -> SolverMethod:
    """Feloldja az ``auto`` beállítást a tényleges megoldási módszerre."""

    if options.method is not SolverMethod.AUTO:
        return options.method
    return (
        SolverMethod.DIRECT
        if dof_count <= options.direct_dof_limit
        else SolverMethod.CONJUGATE_GRADIENT
    )


def solve_factorized_system(
    matrix: csr_matrix,
    right_hand_side: np.ndarray,
    factorization: SparseDirectFactorization,
    *,
    reused: bool = False,
) -> tuple[np.ndarray, SolverInfo]:
    """Megold egy jobb oldalt egy már elkészített ritka faktorizációval."""

    if factorization.shape != matrix.shape:
        raise ValueError("factorization and matrix shapes do not match")
    solution = factorization.solve(right_hand_side)
    return solution, _solver_info(
        matrix,
        right_hand_side,
        solution,
        SolverMethod.DIRECT,
        factorization_reused=reused,
    )


def solve_sparse_system(
    matrix: csr_matrix,
    right_hand_side: np.ndarray,
    options: SolverOptions,
) -> tuple[np.ndarray, SolverInfo]:
    """Megold egy szimmetrikus pozitív definit redukált FEM-rendszert.

    Returns:
        Kételemű tuple: a megoldásvektor és a :class:`SolverInfo`.

    Raises:
        ValueError: Szinguláris mátrix vagy hibás Jacobi-diagonál esetén.
        SolverConvergenceError: Ha a CG nem éri el a kért toleranciát.
    """

    dof_count = matrix.shape[0]
    method = selected_solver_method(options, dof_count)

    iterations: int | None = None
    if method is SolverMethod.DIRECT:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            factorization = SparseDirectFactorization(matrix)
            solution = factorization.solve(right_hand_side)
    else:
        preconditioner = None
        if options.preconditioner is Preconditioner.JACOBI:
            # M^-1 = diag(K)^-1. A LinearOperator miatt még diagonális ritka
            # mátrixot sem kell létrehozni; csak egy vektorszorzás történik.
            diagonal = matrix.diagonal()
            threshold = np.finfo(float).eps * max(float(np.abs(diagonal).max()), 1.0)
            if np.any(np.abs(diagonal) <= threshold):
                raise ValueError("cannot build Jacobi preconditioner from a zero diagonal")
            inverse_diagonal = 1.0 / diagonal
            preconditioner = LinearOperator(
                matrix.shape,
                matvec=lambda vector: inverse_diagonal * vector,
                dtype=float,
            )
        iteration_count = 0

        def count_iteration(_):
            """SciPy callback: kizárólag az elvégzett CG-iterációkat számlálja."""
            nonlocal iteration_count
            iteration_count += 1

        solution, status = cg(
            matrix,
            right_hand_side,
            rtol=options.relative_tolerance,
            atol=options.absolute_tolerance,
            maxiter=options.max_iterations,
            M=preconditioner,
            callback=count_iteration,
        )
        iterations = iteration_count
        if status != 0:
            reason = (
                f"did not converge in {status} iterations"
                if status > 0
                else "failed because of an illegal input or numerical breakdown"
            )
            raise SolverConvergenceError(f"conjugate-gradient solver {reason}")

    if not np.all(np.isfinite(solution)):
        raise ValueError("the sparse solution contains non-finite values")
    return solution, _solver_info(
        matrix,
        right_hand_side,
        solution,
        method,
        iterations=iterations,
    )
