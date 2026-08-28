import unittest
from pathlib import Path

import numpy as np

from fempy import (
    CircularArc2D,
    Geometry2D,
    GmshMesher,
    LinearElasticMaterial,
    LineSegment2D,
    Mesh,
    Model,
    PlaneCondition,
    PlotStyle,
    Quad4,
    SolverMethod,
    SolverOptions,
    Triangle3,
    Triangle6,
    load_myfem,
    load_protus,
    rectangular_quad_mesh,
    rectangular_t6_mesh,
    run_classic_validations,
)

ROOT = Path(__file__).resolve().parents[1]


class MaterialTests(unittest.TestCase):
    def test_plane_stress_matrix(self):
        material = LinearElasticMaterial(200.0, 0.25)
        matrix = material.constitutive_matrix(PlaneCondition.STRESS)
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.all(np.linalg.eigvalsh(matrix) > 0.0))
        self.assertAlmostEqual(matrix[0, 0], 200.0 / (1.0 - 0.25**2))

    def test_invalid_material_is_rejected(self):
        with self.assertRaises(ValueError):
            LinearElasticMaterial(-1.0, 0.3)
        with self.assertRaises(ValueError):
            LinearElasticMaterial(1.0, 0.5)
        with self.assertRaises(ValueError):
            LinearElasticMaterial(float("nan"), 0.3)


class ElementTests(unittest.TestCase):
    def test_inverted_quad_is_rejected_when_mesh_is_created(self):
        with self.assertRaises(ValueError):
            Mesh(
                [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)],
                [Quad4((0, 1, 2, 3))],
            )

    def test_mesh_does_not_change_the_input_array_writeability(self):
        nodes = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
        mesh = Mesh(nodes, [Triangle3((0, 1, 2))])
        self.assertTrue(nodes.flags.writeable)
        self.assertFalse(mesh.nodes.flags.writeable)

    def test_quad_patch_recovers_affine_strain(self):
        mesh = Mesh(
            [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)],
            [Quad4((0, 1, 2, 3))],
        )
        model = Model(mesh, LinearElasticMaterial(1000.0, 0.2))
        # u = [0.01*x, -0.02*y] is an exact constant-strain field.
        for node, (x, y) in enumerate(mesh.nodes):
            model.prescribe(node, ux=0.01 * x, uy=-0.02 * y)
        result = model.solve()
        self.assertTrue(np.allclose(result.strain[0], [0.01, -0.02, 0.0], atol=1e-12))
        self.assertEqual(len(result.element_results[0].integration_points), 4)
        self.assertTrue(
            np.allclose(
                result.integration_point_strain[0],
                np.tile([0.01, -0.02, 0.0], (4, 1)),
                atol=1e-12,
            )
        )
        self.assertTrue(
            np.allclose(
                result.nodal_strain,
                np.tile([0.01, -0.02, 0.0], (4, 1)),
                atol=1e-12,
            )
        )

    def test_triangle_patch_recovers_affine_strain(self):
        mesh = Mesh(
            [(0.0, 0.0), (2.0, 0.0), (0.0, 1.0)],
            [Triangle3((0, 1, 2))],
        )
        model = Model(mesh, LinearElasticMaterial(1000.0, 0.2))
        for node, (x, y) in enumerate(mesh.nodes):
            model.prescribe(node, ux=0.01 * x + 0.03 * y, uy=-0.02 * y)
        result = model.solve()
        self.assertTrue(np.allclose(result.strain[0], [0.01, -0.02, 0.03], atol=1e-12))

    def test_t6_patch_recovers_affine_strain_at_all_seven_integration_points(self):
        mesh = rectangular_t6_mesh(1, 1, 2.0, 1.0)
        model = Model(mesh, LinearElasticMaterial(1000.0, 0.2))
        for node, (x, y) in enumerate(mesh.nodes):
            model.prescribe(node, ux=0.01 * x + 0.03 * y, uy=-0.02 * y)
        result = model.solve()
        expected = np.array([0.01, -0.02, 0.03])
        self.assertTrue(all(isinstance(element, Triangle6) for element in mesh.elements))
        self.assertTrue(
            all(
                np.allclose(point.strain, expected, atol=1e-12)
                for element in result.element_results
                for point in element.integration_points
            )
        )
        self.assertTrue(np.allclose(result.nodal_strain, expected, atol=1e-12))

    def test_t6_consistent_mass_has_correct_total_mass_in_each_direction(self):
        mesh = rectangular_t6_mesh(1, 1, 2.0, 1.0)
        density = 3.0
        thickness = 0.5
        matrix = Model(
            mesh,
            LinearElasticMaterial(1000.0, 0.2, density=density),
            thickness=thickness,
        ).mass_matrix()
        expected_mass = density * thickness * 2.0
        self.assertAlmostEqual(matrix[0::2, 0::2].sum(), expected_mass, places=11)
        self.assertAlmostEqual(matrix[1::2, 1::2].sum(), expected_mass, places=11)

    def test_t6_edge_traction_uses_consistent_one_four_one_load_ratio(self):
        nodes = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]
        mesh = Mesh(
            nodes,
            [Triangle6((0, 1, 2, 3, 4, 5))],
            node_sets={"bottom": (0, 3, 1)},
            edge_sets={"bottom": ((0, 3, 1),)},
        )
        model = Model(mesh, LinearElasticMaterial(1000.0, 0.2))
        model.add_boundary_traction("bottom", tx=6.0)
        force_x = model.force_vector[0::2]
        self.assertTrue(np.allclose(force_x[[0, 3, 1]], [1.0, 4.0, 1.0], atol=1e-12))
        self.assertAlmostEqual(force_x.sum(), 6.0, places=12)


class ModelTests(unittest.TestCase):
    def test_positive_pressure_acts_inward_and_prescribed_boundary_is_vectorized(self):
        mesh = Mesh(
            [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)],
            [Quad4((0, 1, 2, 3))],
            node_sets={"top": [2, 3]},
            edge_sets={"top": [(2, 3)]},
        )
        model = Model(mesh, LinearElasticMaterial(1000.0, 0.3), thickness=4.0)
        model.prescribe_boundary("top", ux=0.1).add_boundary_pressure("top", 3.0)
        self.assertTrue(np.allclose(model.prescribed_displacements[[2, 3], 0], 0.1))
        self.assertAlmostEqual(model.force_vector[1::2].sum(), -24.0)

    def test_non_finite_model_inputs_are_rejected(self):
        mesh = rectangular_quad_mesh(1, 1, 1.0, 1.0)
        material = LinearElasticMaterial(1000.0, 0.3)
        with self.assertRaises(ValueError):
            Model(mesh, material, thickness=float("nan"))
        model = Model(mesh, material)
        with self.assertRaises(ValueError):
            model.add_nodal_load(0, fx=float("nan"))
        with self.assertRaises(ValueError):
            model.prescribe(0, ux=float("inf"))

    def test_load_and_reaction_are_in_equilibrium(self):
        mesh = rectangular_quad_mesh(4, 2, 4.0, 1.0)
        model = Model(mesh, LinearElasticMaterial(10_000.0, 0.3))
        model.fix_nodes(mesh.nodes_where(x=0.0))
        loaded = mesh.nodes_where(x=4.0)
        model.add_nodal_loads(loaded, fy=-30.0 / len(loaded))
        result = model.solve()
        support_reaction = result.reaction[mesh.nodes_where(x=0.0)].sum(axis=0)
        self.assertAlmostEqual(support_reaction[1], 30.0, places=8)
        self.assertGreater(result.displacement_magnitude.max(), 0.0)

    def test_vtk_export_contains_expected_fields(self):
        mesh = rectangular_quad_mesh(1, 1, 1.0, 1.0)
        model = Model(mesh, LinearElasticMaterial(1000.0, 0.3))
        for node in range(mesh.node_count):
            model.prescribe(node, ux=0.0, uy=0.0)
        output = ROOT / "tests" / "_temporary_result.vtk"
        try:
            result = model.solve()
            result.write_vtk(output)
            text = output.read_text(encoding="utf-8")
            self.assertIn("VECTORS displacement", text)
            self.assertIn("SCALARS von_mises", text)
            self.assertIn("VECTORS principal_stress_1_vector", text)
            self.assertIn("SCALARS stress_x_gp4", text)
        finally:
            output.unlink(missing_ok=True)

    def test_consistent_mass_preserves_total_mass(self):
        mesh = rectangular_quad_mesh(1, 1, 2.0, 3.0)
        material = LinearElasticMaterial(1000.0, 0.3, density=4.0)
        model = Model(mesh, material, thickness=0.5)
        mass = model.mass_matrix()
        total_mass = 2.0 * 3.0 * 0.5 * 4.0
        acceleration_x = np.tile([1.0, 0.0], mesh.node_count)
        force = np.asarray(mass @ acceleration_x).reshape((-1, 2))
        self.assertAlmostEqual(force[:, 0].sum(), total_mass, places=12)
        self.assertAlmostEqual(force[:, 1].sum(), 0.0, places=12)

    def test_principal_stress_and_direction(self):
        mesh = rectangular_quad_mesh(1, 1, 1.0, 1.0)
        poisson = 0.25
        model = Model(mesh, LinearElasticMaterial(1000.0, poisson))
        # ey=-nu*ex produces uniaxial sigma_x in plane stress.
        for node, (x, y) in enumerate(mesh.nodes):
            model.prescribe(node, ux=0.01 * x, uy=-poisson * 0.01 * y)
        result = model.solve()
        self.assertTrue(np.allclose(result.nodal_principal_stress[:, 1], 0.0, atol=1e-12))
        self.assertTrue(np.allclose(result.nodal_principal_angle, 0.0, atol=1e-12))

    def test_suggested_deformation_scale_is_geometry_relative(self):
        mesh = rectangular_quad_mesh(2, 1, 10.0, 2.0)
        model = Model(mesh, LinearElasticMaterial(1000.0, 0.3))
        model.fix_nodes(mesh.nodes_where(x=0.0))
        model.add_nodal_load(mesh.nodes_where(x=10.0)[-1], fy=-1.0)
        result = model.solve()
        scale = result.suggested_deformation_scale(fraction=0.05)
        displayed_maximum = scale * result.displacement_magnitude.max()
        self.assertAlmostEqual(displayed_maximum, 0.05 * 10.0, places=12)

    def test_colour_bar_matches_chart_rectangle_height(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        mesh = rectangular_quad_mesh(1, 1, 1.0, 1.0)
        model = Model(mesh, LinearElasticMaterial(1000.0, 0.3))
        for node in range(mesh.node_count):
            model.prescribe(node, ux=0.001 * mesh.nodes[node, 0], uy=0.0)
        result = model.solve()
        figure = Figure(figsize=(5, 3))
        canvas = FigureCanvasAgg(figure)
        chart_axis = figure.subplots()
        result.plot(field="nodal_von_mises", ax=chart_axis)
        canvas.draw()
        colour_axis = figure.axes[-1]
        self.assertAlmostEqual(
            chart_axis.get_window_extent().height,
            colour_axis.get_window_extent().height,
            places=8,
        )

    def test_hungarian_plot_uses_decimal_comma_and_scientific_stress_notation(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        mesh = rectangular_t6_mesh(1, 1, 1.0, 1.0)
        model = Model(mesh, LinearElasticMaterial(210_000.0, 0.3))
        for node, (x, _y) in enumerate(mesh.nodes):
            model.prescribe(node, ux=0.1 * x, uy=0.0)
        result = model.solve()
        figure = Figure(figsize=(5, 3))
        canvas = FigureCanvasAgg(figure)
        axis = figure.subplots()
        style = PlotStyle(language="hu", length_unit="mm", stress_unit="MPa")
        result.plot(field="nodal_stress_x", ax=axis, style=style)
        canvas.draw()
        self.assertEqual(axis.xaxis.get_major_formatter()(1.5, 0), "1,5")
        self.assertEqual(PlotStyle(language="en").number(1.5), "1.5")
        self.assertIn("normálfeszültség", axis.get_title())
        self.assertIn(r"$\sigma_x$", figure.axes[-1].get_ylabel())
        self.assertIn(r"\times 10^{", figure.axes[-1].get_ylabel())

    def test_sparse_stiffness_is_symmetric(self):
        mesh = rectangular_quad_mesh(4, 2, 4.0, 2.0)
        model = Model(mesh, LinearElasticMaterial(1000.0, 0.3))
        matrix = model.stiffness_matrix()
        difference = matrix - matrix.T
        self.assertLess(
            np.max(np.abs(difference.data)) if difference.nnz else 0.0,
            1e-10,
        )
        dense_bytes = matrix.shape[0] ** 2 * 8
        sparse_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        self.assertLess(sparse_bytes, dense_bytes)

    def test_conjugate_gradient_matches_sparse_direct_solver(self):
        mesh = rectangular_quad_mesh(10, 3, 10.0, 3.0)
        model = Model(mesh, LinearElasticMaterial(10_000.0, 0.3))
        model.fix_nodes(mesh.nodes_where(x=0.0))
        loaded = mesh.nodes_where(x=10.0)
        model.add_nodal_loads(loaded, fy=-10.0 / len(loaded))
        direct = model.solve("direct")
        iterative = model.solve(
            SolverOptions(
                method="cg",
                relative_tolerance=1e-11,
                max_iterations=10_000,
            )
        )
        self.assertEqual(iterative.solver_info.method, SolverMethod.CONJUGATE_GRADIENT)
        self.assertIsNotNone(iterative.solver_info.iterations)
        self.assertLess(iterative.solver_info.relative_residual, 1e-10)
        self.assertTrue(
            np.allclose(iterative.displacement, direct.displacement, rtol=1e-8, atol=1e-11)
        )

    def test_auto_solver_can_switch_to_iterative_mode(self):
        mesh = rectangular_quad_mesh(3, 1, 3.0, 1.0)
        model = Model(mesh, LinearElasticMaterial(1000.0, 0.25))
        model.fix_nodes(mesh.nodes_where(x=0.0))
        model.add_nodal_load(mesh.nodes_where(x=3.0)[-1], fy=-1.0)
        result = model.solve(SolverOptions(direct_dof_limit=1, max_iterations=5000))
        self.assertEqual(result.solver_info.method, SolverMethod.CONJUGATE_GRADIENT)


class GeometryAndMeshingTests(unittest.TestCase):
    def test_geometry_keeps_named_boundaries_and_local_sizes(self):
        geometry = (
            Geometry2D("plate")
            .add_rectangle(10.0, 5.0)
            .add_circle((5.0, 2.5), 1.0, boundary="hole", mesh_size=0.2)
        )
        geometry.validate()
        self.assertEqual(
            geometry.boundary_names,
            ("bottom", "hole", "left", "right", "top"),
        )
        self.assertEqual(geometry.boundary_sizes["hole"], 0.2)

    def test_general_line_and_arc_loop(self):
        geometry = Geometry2D("arched").add_loop(
            [
                LineSegment2D((0.0, 0.0), (2.0, 0.0), "bottom"),
                LineSegment2D((2.0, 0.0), (2.0, 1.0), "right"),
                CircularArc2D((2.0, 1.0), (1.0, 1.0), (1.0, 2.0), "arc"),
                LineSegment2D((1.0, 2.0), (0.0, 2.0), "top"),
                LineSegment2D((0.0, 2.0), (0.0, 0.0), "left"),
            ]
        )
        geometry.validate()
        self.assertEqual(len(geometry.loops), 1)
        self.assertIn("arc", geometry.boundary_names)

    def test_mesh_boundary_sets_are_immutable_and_easy_to_query(self):
        mesh = Mesh(
            [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
            [Triangle3((0, 1, 2))],
            node_sets={"bottom": [1, 0, 1]},
            edge_sets={"bottom": [(0, 1)]},
        )
        self.assertEqual(mesh.boundary_names, ("bottom",))
        self.assertEqual(mesh.boundary_nodes("bottom"), [0, 1])
        self.assertEqual(mesh.boundary_edges("bottom"), [(0, 1)])
        with self.assertRaises(TypeError):
            mesh.node_sets["new"] = (0,)  # type: ignore[index]

    def test_named_boundary_support_and_traction(self):
        mesh = Mesh(
            [(0.0, 0.0), (2.0, 0.0), (0.0, 1.0)],
            [Triangle3((0, 1, 2))],
            node_sets={"left": [0, 2], "bottom": [0, 1]},
            edge_sets={"left": [(2, 0)], "bottom": [(0, 1)]},
        )
        model = Model(mesh, LinearElasticMaterial(1000.0, 0.3), thickness=3.0)
        model.fix_boundary("left").add_boundary_traction("bottom", ty=-4.0)
        self.assertAlmostEqual(model.force_vector[1::2].sum(), -24.0)

    def test_named_boundaries_can_be_plotted(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        mesh = Mesh(
            [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
            [Triangle3((0, 1, 2))],
            node_sets={"bottom": [0, 1], "left": [0, 2]},
            edge_sets={"bottom": [(0, 1)], "left": [(2, 0)]},
        )
        figure = Figure(figsize=(4, 3))
        canvas = FigureCanvasAgg(figure)
        axis = figure.subplots()
        mesh.plot_boundaries(ax=axis)
        canvas.draw()
        self.assertEqual({text.get_text() for text in axis.texts}, {"bottom", "left"})
        self.assertIsNotNone(axis.get_legend())

    def test_directional_constraints_can_be_plotted(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        mesh = rectangular_quad_mesh(2, 1, 2.0, 1.0)
        model = Model(mesh, LinearElasticMaterial(1000.0, 0.3))
        model.fix_node(0, x=True, y=False)
        model.fix_node(3, x=False, y=True)
        model.fix_node(2)
        model.add_nodal_load(5, fy=-1.0)
        figure = Figure(figsize=(5, 3))
        canvas = FigureCanvasAgg(figure)
        axis = figure.subplots()
        model.plot_boundary_conditions(ax=axis)
        canvas.draw()
        labels = set(axis.get_legend_handles_labels()[1])
        self.assertIn("x irányban fix", labels)
        self.assertIn("y irányban fix", labels)
        self.assertIn("x és y irányban fix", labels)
        self.assertIn("csomóponti terhelés", labels)

    def test_real_gmsh_plate_with_hole_when_available(self):
        try:
            import gmsh  # noqa: F401
        except (ImportError, OSError):
            self.skipTest("optional gmsh package is not installed")
        geometry = (
            Geometry2D("test_plate")
            .add_rectangle(4.0, 2.0)
            .add_circle((2.0, 1.0), 0.35, boundary="hole", mesh_size=0.12)
        )
        mesh = GmshMesher(0.35).generate(geometry)
        self.assertGreater(mesh.node_count, 20)
        self.assertGreater(mesh.element_count, 20)
        self.assertEqual(set(mesh.boundary_names), set(geometry.boundary_names))
        self.assertGreater(len(mesh.boundary_nodes("hole")), 8)
        self.assertTrue(all(isinstance(element, Triangle3) for element in mesh.elements))

    def test_real_gmsh_generates_t6_and_keeps_midside_boundary_nodes(self):
        try:
            import gmsh  # noqa: F401
        except (ImportError, OSError):
            self.skipTest("optional gmsh package is not installed")
        geometry = Geometry2D("t6_test").add_rectangle(2.0, 1.0)
        mesh = GmshMesher(0.4, order=2).generate(geometry)
        self.assertTrue(all(isinstance(element, Triangle6) for element in mesh.elements))
        self.assertTrue(all(len(element.node_ids) == 6 for element in mesh.elements))
        self.assertGreater(len(mesh.boundary_nodes("bottom")), 5)
        self.assertGreater(len(mesh.boundary_edges("bottom")), 4)


class LegacyTests(unittest.TestCase):
    def test_myfem_import_and_solve(self):
        model = load_myfem(ROOT / "tests" / "data" / "myfem" / "coarse.fem")
        self.assertEqual(model.mesh.node_count, 8)
        self.assertEqual(model.mesh.element_count, 6)
        result = model.solve()
        self.assertTrue(np.all(np.isfinite(result.displacement)))

    def test_protus_import(self):
        model = load_protus(ROOT / "tests" / "data" / "protus" / "INPUT_FEA_PROTUS.txt")
        self.assertEqual(model.mesh.node_count, 292)
        self.assertEqual(model.mesh.element_count, 231)
        self.assertEqual(model.condition, PlaneCondition.STRESS)
        self.assertEqual(np.count_nonzero(model.force_vector), 5)

    def test_all_copied_femaster_samples_import(self):
        expected = {
            "btfemexample.fem": (453, 803),
            "coarse.fem": (8, 6),
            "notchedspecimen.fem": (2102, 4036),
            "plhole.fem": (2996, 5700),
            "rect.fem": (1279, 2336),
            "smallrect.fem": (106, 154),
        }
        directory = ROOT / "examples" / "femaster_samples"
        for name, counts in expected.items():
            with self.subTest(sample=name):
                model = load_myfem(directory / name)
                self.assertEqual((model.mesh.node_count, model.mesh.element_count), counts)


class ClassicValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_classic_validations()

    def test_all_classic_benchmarks_pass_and_converge(self):
        report = self.report
        failures = [case.name + " / " + case.element for case in report.cases if not case.passed]
        self.assertTrue(report.passed, f"failed validation cases: {failures}")
        self.assertEqual(len(report.cases), 9)
        self.assertTrue(all(case.converges for case in report.cases if len(case.samples) > 1))

        t6_cases = [case for case in report.cases if case.element == "Triangle6"]
        self.assertEqual(len(t6_cases), 3)
        self.assertTrue(all(case.passed for case in t6_cases))

    def test_cook_quad4_matches_published_coarse_mesh_value(self):
        report = self.report
        cook = next(
            case for case in report.cases if case.name == "Cook-membrán" and case.element == "Quad4"
        )
        self.assertAlmostEqual(cook.samples[0].value, 11.85, delta=0.02)
        self.assertLess(cook.final_error, 0.01)


if __name__ == "__main__":
    unittest.main()
