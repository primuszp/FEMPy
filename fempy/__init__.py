"""A FEMPy Edu kétdimenziós, oktatási célú végeselemes csomagja.

A csomag gyökere szándékosan csak a lineárisan rugalmas elemzéshez szükséges,
magas szintű fogalmakat exportálja. Így a felhasználónak nem kell belső
modulútvonalakat ismernie; a részletes implementáció az almodulokban marad.
"""

from .elements import Quad4, Triangle3
from .geometry import CircularArc2D, Geometry2D, LineSegment2D
from .gmsh import GmshMesher, GmshNotInstalledError
from .legacy import load_myfem, load_protus
from .material import LinearElasticMaterial, PlaneCondition
from .mesh import Mesh, rectangular_quad_mesh, rectangular_tri_mesh
from .model import Model
from .plotting import (
    plot_boundaries,
    plot_boundary_conditions,
    plot_mesh,
    plot_sparse_matrix,
)
from .result import AnalysisResult, ElementResult, IntegrationPointResult
from .solver import (
    Preconditioner,
    SolverConvergenceError,
    SolverInfo,
    SolverMethod,
    SolverOptions,
)
from .validation import (
    ConvergenceSample,
    ValidationCase,
    ValidationReport,
    run_classic_validations,
)

__all__ = [
    "AnalysisResult",
    "CircularArc2D",
    "ConvergenceSample",
    "ElementResult",
    "Geometry2D",
    "GmshMesher",
    "GmshNotInstalledError",
    "IntegrationPointResult",
    "LineSegment2D",
    "LinearElasticMaterial",
    "Mesh",
    "Model",
    "PlaneCondition",
    "Preconditioner",
    "Quad4",
    "SolverConvergenceError",
    "SolverInfo",
    "SolverMethod",
    "SolverOptions",
    "Triangle3",
    "ValidationCase",
    "ValidationReport",
    "load_myfem",
    "load_protus",
    "plot_boundaries",
    "plot_boundary_conditions",
    "plot_mesh",
    "plot_sparse_matrix",
    "rectangular_quad_mesh",
    "rectangular_tri_mesh",
    "run_classic_validations",
]

__version__ = "1.0.0"
