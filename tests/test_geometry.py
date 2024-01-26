"""Geometry methods tests"""
import unittest

import numpy as np

from gamma_ray_tracking import geometry as geo


class test_cos_act(unittest.TestCase):
    """
    Tests scattering angle cosine (angle is between vectors p1 -> p2 and p2 -> p3)
    """

    def test_right_triangle(self):
        """Test with a 3-4-5 right triangle"""
        points = np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0]])

        expected_cos = -0.6  # negative because scattering behind
        computed_cos = geo.cos_act(points[0], points[1], points[2])
        self.assertAlmostEqual(computed_cos, expected_cos, places=4)

        expected_cos = 0  # 90 degrees
        computed_cos = geo.cos_act(points[2], points[0], points[1])
        self.assertAlmostEqual(computed_cos, expected_cos, places=4)

        expected_cos = -0.8  # negative because scattering behind
        computed_cos = geo.cos_act(points[1], points[2], points[0])
        self.assertAlmostEqual(computed_cos, expected_cos, places=4)

    def test_equilateral_triangle(self):
        """Test with an equilateral triangle"""
        # All angles should be 60 degrees
        points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.5 * np.sqrt(3), 0]])

        expected_cos = -0.5  # Cos 120 degrees (-60 degree scatter)
        computed_cos = geo.cos_act(points[0], points[1], points[2])
        self.assertAlmostEqual(computed_cos, expected_cos, places=4)

        computed_cos = geo.cos_act(points[1], points[2], points[0])
        self.assertAlmostEqual(computed_cos, expected_cos, places=4)

        computed_cos = geo.cos_act(points[2], points[0], points[1])
        self.assertAlmostEqual(computed_cos, expected_cos, places=4)


class test_cosine_vec(unittest.TestCase):
    """
    Tests scattering angle cosine (angle is between vectors p1 -> p2 and p2 -> p3)
    """

    def test_right_triangle(self):
        """Test with a 3-4-5 right triangle"""
        points = np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0]])

        expected_cos = np.array([0.0])
        computed_cos = geo.cosine_vec(points[1:], center=points[0])
        np.testing.assert_allclose(computed_cos, expected_cos, atol=1e-4)

        expected_cos = np.array([0.6])
        computed_cos = geo.cosine_vec(points[[0, 2]], center=points[1])
        np.testing.assert_allclose(computed_cos, expected_cos, atol=1e-4)

        expected_cos = np.array([0.8])
        computed_cos = geo.cosine_vec(points[:-1], center=points[2])
        np.testing.assert_allclose(computed_cos, expected_cos, atol=1e-4)

    def test_equilateral_triangle(self):
        """Test with an equilateral triangle"""
        # All angles should be 60 degrees
        points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.5 * np.sqrt(3), 0]])

        expected_cos = np.array([0.5])
        computed_cos = geo.cosine_vec(points[1:], center=points[0])
        np.testing.assert_allclose(computed_cos, expected_cos, atol=1e-4)

        expected_cos = np.array([0.5])
        computed_cos = geo.cosine_vec(points[[0, 2]], center=points[1])
        np.testing.assert_allclose(computed_cos, expected_cos, atol=1e-4)

        expected_cos = np.array([0.5])
        computed_cos = geo.cosine_vec(points[:-1], center=points[2])
        np.testing.assert_allclose(computed_cos, expected_cos, atol=1e-4)


class test_one_minus_cosine_ijk(unittest.TestCase):
    """
    Tests scattering angle cosine (angle is between vectors p1 -> p2 and p2 -> p3)
    """

    def test_right_triangle(self):
        """Test with a 3-4-5 right triangle"""
        points = np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0]])

        expected_cos = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.6], [0.0, 1.8, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.8, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.6, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        )
        computed_cos = geo.one_minus_cosine_ijk(points)
        np.testing.assert_allclose(computed_cos, expected_cos, atol=1e-4)

    def test_equilateral_triangle(self):
        """Test with an equilateral triangle"""
        # All angles should be 60 degrees
        points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.5 * np.sqrt(3), 0]])

        expected_cos = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [0.0, 1.5, 0.0]],
                [[0.0, 0.0, 1.5], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
                [[0.0, 1.5, 0.0], [1.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        )
        computed_cos = geo.one_minus_cosine_ijk(points)
        np.testing.assert_allclose(computed_cos, expected_cos, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
