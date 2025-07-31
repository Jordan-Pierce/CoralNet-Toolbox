#!/usr/bin/env python

"""Tests for `coralnet_toolbox` package."""

import pytest
import numpy as np
from shapely.geometry import Polygon

from coralnet_toolbox.utilities import simplify_polygon


class TestSimplifyPolygon:
    """Test cases for the simplify_polygon function."""

    def test_simple_square(self):
        """Test simplification of a simple square - should remain unchanged."""
        square_points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        result = simplify_polygon(square_points, simplify_tolerance=0.1)
        
        # Should return the same points (maybe in different order)
        assert len(result) == 4
        # Verify it's still a valid polygon
        polygon = Polygon(result)
        assert polygon.is_valid
        assert abs(polygon.area - 1.0) < 1e-10

    def test_complex_polygon_simplification(self):
        """Test that a complex polygon gets simplified."""
        # Create a polygon with many points that should be simplified
        # A rectangle with extra points along the edges
        complex_points = [
            (0, 0), (0.25, 0), (0.5, 0), (0.75, 0), (1, 0),
            (1, 0.25), (1, 0.5), (1, 0.75), (1, 1),
            (0.75, 1), (0.5, 1), (0.25, 1), (0, 1),
            (0, 0.75), (0, 0.5), (0, 0.25)
        ]
        
        result = simplify_polygon(complex_points, simplify_tolerance=0.1)
        
        # Should be simplified to fewer points
        assert len(result) < len(complex_points)
        # Should still be a valid polygon
        polygon = Polygon(result)
        assert polygon.is_valid
        # Area should be approximately preserved
        original_polygon = Polygon(complex_points)
        assert abs(polygon.area - original_polygon.area) < 0.1

    def test_different_tolerance_values(self):
        """Test that different tolerance values produce different levels of simplification."""
        # Create a slightly wavy line that can be simplified
        wavy_square = [
            (0, 0), (0.1, 0.05), (0.5, 0), (0.9, 0.05), (1, 0),
            (1.05, 0.1), (1, 0.5), (1.05, 0.9), (1, 1),
            (0.9, 1.05), (0.5, 1), (0.1, 1.05), (0, 1),
            (0.05, 0.9), (0, 0.5), (0.05, 0.1)
        ]
        
        result_low_tolerance = simplify_polygon(wavy_square, simplify_tolerance=0.01)
        result_high_tolerance = simplify_polygon(wavy_square, simplify_tolerance=0.2)
        
        # Higher tolerance should result in fewer points
        assert len(result_high_tolerance) <= len(result_low_tolerance)
        
        # Both should be valid polygons
        assert Polygon(result_low_tolerance).is_valid
        assert Polygon(result_high_tolerance).is_valid

    def test_triangle_unchanged(self):
        """Test that a triangle (minimal polygon) remains unchanged."""
        triangle = [(0, 0), (1, 0), (0.5, 1)]
        result = simplify_polygon(triangle, simplify_tolerance=0.1)
        
        # Should still have 3 points
        assert len(result) == 3
        # Should be a valid polygon
        polygon = Polygon(result)
        assert polygon.is_valid

    def test_numpy_array_input(self):
        """Test that function works with numpy array input."""
        square_points = np.array([(0, 0), (1, 0), (1, 1), (0, 1)])
        result = simplify_polygon(square_points, simplify_tolerance=0.1)
        
        # Should work and return a list
        assert isinstance(result, list)
        assert len(result) == 4
        polygon = Polygon(result)
        assert polygon.is_valid

    def test_self_intersecting_polygon(self):
        """Test handling of self-intersecting (invalid) polygons."""
        # Create a bow-tie shape (self-intersecting)
        bowtie = [(0, 0), (1, 1), (1, 0), (0, 1)]
        
        # Should not raise an exception
        result = simplify_polygon(bowtie, simplify_tolerance=0.1)
        
        # Should return some result (the function handles invalid polygons)
        assert isinstance(result, list)
        assert len(result) >= 3  # At least a triangle

    def test_multipolygon_handling(self):
        """Test that when simplification creates multiple polygons, largest is returned."""
        # Create points that might result in multiple polygons after processing
        # This is a complex case that tests the MultiPolygon handling
        complex_shape = [
            (0, 0), (2, 0), (2, 0.1), (1, 0.1), (1, 0.9), (2, 0.9), (2, 1),
            (0, 1), (0, 0.9), (1, 0.9), (1, 0.1), (0, 0.1)
        ]
        
        result = simplify_polygon(complex_shape, simplify_tolerance=0.05)
        
        # Should return a single polygon (the largest one)
        assert isinstance(result, list)
        assert len(result) >= 3
        polygon = Polygon(result)
        assert polygon.is_valid

    def test_edge_cases_and_invalid_inputs(self):
        """Test edge cases with empty, minimal, and invalid inputs."""
        # Test with empty list - should return empty list
        assert simplify_polygon([], simplify_tolerance=0.1) == []

        # Test with minimal valid polygon (triangle)
        minimal = [(0, 0), (1, 0), (0, 1)]
        result = simplify_polygon(minimal, simplify_tolerance=0.1)
        assert len(result) == 3
        
        # Test with duplicate points (should be handled gracefully)
        with_duplicates = [(0, 0), (0, 0), (1, 0), (1, 1), (0, 1)]
        result = simplify_polygon(with_duplicates, simplify_tolerance=0.1)
        assert len(result) == 4

    def test_large_tolerance(self):
        """Test behavior with very large tolerance values."""
        square_points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        result = simplify_polygon(square_points, simplify_tolerance=10.0)
        
        # Even with large tolerance, should maintain basic polygon structure
        assert len(result) >= 3  # At least a triangle
        polygon = Polygon(result)
        assert polygon.is_valid

    def test_zero_tolerance(self):
        """Test behavior with zero tolerance (no simplification)."""
        complex_points = [(0, 0), (0.1, 0), (0.2, 0), (1, 0), (1, 1), (0, 1)]
        result = simplify_polygon(complex_points, simplify_tolerance=0.0)
        
        # With zero tolerance, should preserve more detail
        assert len(result) >= 4  # Should preserve most points
        polygon = Polygon(result)
        assert polygon.is_valid
