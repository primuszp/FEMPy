"""A PrimFEM olvasható, validált kétdimenziós végeselemes csomagja.

A csomag gyökere szándékosan csak a lineárisan rugalmas elemzéshez szükséges,
magas szintű fogalmakat exportálja. Így a felhasználónak nem kell belső
modulútvonalakat ismernie; a részletes implementáció az almodulokban marad.
"""

from .element_checks import ElementCheckReport, verify_element, verify_supported_elements
from .elements import Quad4, Triangle3, Triangle6
from .geometry import CircularArc2D, Geometry2D, LineSegment2D
from .gmsh import GmshMesher, GmshNotInstalledError
from .legacy import load_myfem, load_protus
from .loadcase import LoadCase
from .material import LinearElasticMaterial, PlaneCondition
from .mesh import (
    Mesh,
    rectangular_quad_mesh,
    rectangular_t6_mesh,
    rectangular_tri_mesh,
    to_quadratic_tri_mesh,
)
from .meshio_adapter import MeshioNotInstalledError
from .model import Model
from .plotting import (
    PlotStyle,
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
    "ElementCheckReport",
    "ElementResult",
    "Geometry2D",
    "GmshMesher",
    "GmshNotInstalledError",
    "IntegrationPointResult",
    "LineSegment2D",
    "LinearElasticMaterial",
    "LoadCase",
    "Mesh",
    "MeshioNotInstalledError",
    "Model",
    "PlaneCondition",
    "PlotStyle",
    "Preconditioner",
    "Quad4",
    "SolverConvergenceError",
    "SolverInfo",
    "SolverMethod",
    "SolverOptions",
    "Triangle3",
    "Triangle6",
    "ValidationCase",
    "ValidationReport",
    "load_myfem",
    "load_protus",
    "plot_boundaries",
    "plot_boundary_conditions",
    "plot_mesh",
    "plot_sparse_matrix",
    "rectangular_quad_mesh",
    "rectangular_t6_mesh",
    "rectangular_tri_mesh",
    "run_classic_validations",
    "to_quadratic_tri_mesh",
    "verify_element",
    "verify_supported_elements",
]

__version__ = "1.3.0"
