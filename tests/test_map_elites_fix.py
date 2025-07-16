"""
Test cases for MAP-Elites algorithm fix

These tests verify that the MAP-Elites algorithm is properly implemented and
respects the feature map structure during population limit enforcement.
"""

import tempfile
import unittest
from unittest.mock import Mock

from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase


class TestMapElitesFix(unittest.TestCase):
    """Test cases for MAP-Elites algorithm implementation"""

    def setUp(self):
        """Set up test database"""
        self.config = DatabaseConfig(
            population_size=10,
            archive_size=5,
            num_islands=2,
            feature_dimensions=["complexity", "score"],
            feature_bins=3,
            exploration_ratio=0.3,
            exploitation_ratio=0.4,
            elite_selection_ratio=0.2,
            db_path=None,
        )
        self.db = ProgramDatabase(self.config)

    def test_map_elites_replacement_basic(self):
        """Test that MAP-Elites properly replaces programs in feature cells"""
        # Create two programs that will map to the same feature cell
        program1 = Program(
            id="prog1",
            code="def func1():\n    return 1",
            metrics={"score": 0.5},
        )
        program2 = Program(
            id="prog2", 
            code="def func2():\n    return 2",
            metrics={"score": 0.8},  # Better score
        )
        
        # Add first program
        self.db.add(program1)
        
        # Verify program1 is in the database
        self.assertIn("prog1", self.db.programs)
        
        # Calculate feature coords to verify they're the same
        coords1 = self.db._calculate_feature_coords(program1)
        coords2 = self.db._calculate_feature_coords(program2)
        
        # They should have the same coordinates (same feature cell)
        self.assertEqual(coords1, coords2)
        
        # Add second program (should replace first due to better score)
        self.db.add(program2)
        
        # Verify program2 is in the database
        self.assertIn("prog2", self.db.programs)
        
        # Verify program1 was removed (replaced in feature cell)
        self.assertNotIn("prog1", self.db.programs)
        
        # Verify feature map contains program2
        feature_key = self.db._feature_coords_to_key(coords2)
        self.assertEqual(self.db.feature_map[feature_key], "prog2")

    def test_map_elites_population_limit_respects_diversity(self):
        """Test that population limit enforcement respects MAP-Elites diversity"""
        # Create programs that will occupy different feature cells
        programs = []
        for i in range(15):  # More than population_size (10)
            program = Program(
                id=f"prog{i}",
                code=f"def func{i}():\n    return {'x' * (i * 100)}",  # Different complexity
                metrics={"score": 0.1 + (i * 0.05)},  # Different scores
            )
            programs.append(program)
        
        # Add all programs
        for program in programs:
            self.db.add(program)
        
        # Verify population limit was enforced
        self.assertEqual(len(self.db.programs), self.config.population_size)
        
        # Verify that programs in feature_map are preserved
        feature_map_programs = set(self.db.feature_map.values())
        for program_id in feature_map_programs:
            self.assertIn(program_id, self.db.programs)
        
        # Verify that the feature map structure is maintained
        self.assertGreater(len(self.db.feature_map), 0)

    def test_map_elites_best_program_protection(self):
        """Test that the best program is never removed during replacement or population limit"""
        # Create a clearly best program
        best_program = Program(
            id="best_prog",
            code="def best():\n    return 'best'",
            metrics={"score": 0.99},
        )
        
        # Add best program
        self.db.add(best_program)
        
        # Add many other programs that will trigger population limit
        for i in range(15):
            program = Program(
                id=f"prog{i}",
                code=f"def func{i}():\n    return {'x' * (i * 50)}",
                metrics={"score": 0.1 + (i * 0.02)},
            )
            self.db.add(program)
        
        # Verify best program is still in database
        self.assertIn("best_prog", self.db.programs)
        
        # Verify it's tracked as the best program
        self.assertEqual(self.db.best_program_id, "best_prog")

    def test_map_elites_feature_map_consistency(self):
        """Test that feature_map stays consistent with the actual database state"""
        # Add programs to different feature cells
        programs = []
        for i in range(8):
            program = Program(
                id=f"prog{i}",
                code=f"def func{i}():\n    return {'x' * (i * 200)}",  # Different complexity
                metrics={"score": 0.2 + (i * 0.1)},
            )
            programs.append(program)
            self.db.add(program)
        
        # Verify all programs in feature_map exist in database
        for program_id in self.db.feature_map.values():
            self.assertIn(program_id, self.db.programs)
        
        # Verify no stale references in feature_map
        for key, program_id in self.db.feature_map.items():
            self.assertIn(program_id, self.db.programs)
        
        # Force population limit enforcement
        for i in range(10):
            extra_program = Program(
                id=f"extra{i}",
                code=f"def extra{i}():\n    return {i}",
                metrics={"score": 0.01},  # Low score
            )
            self.db.add(extra_program)
        
        # Verify feature_map is still consistent
        for program_id in self.db.feature_map.values():
            self.assertIn(program_id, self.db.programs)

    def test_remove_program_from_database_method(self):
        """Test the _remove_program_from_database method works correctly"""
        # Create and add a program
        program = Program(
            id="test_prog",
            code="def test():\n    return 'test'",
            metrics={"score": 0.5},
        )
        self.db.add(program)
        
        # Verify program is in all relevant structures
        self.assertIn("test_prog", self.db.programs)
        
        # Find feature key
        coords = self.db._calculate_feature_coords(program)
        feature_key = self.db._feature_coords_to_key(coords)
        if feature_key in self.db.feature_map:
            self.assertEqual(self.db.feature_map[feature_key], "test_prog")
        
        # Remove the program
        self.db._remove_program_from_database("test_prog")
        
        # Verify program is removed from all structures
        self.assertNotIn("test_prog", self.db.programs)
        
        # Verify feature_map is cleaned up
        for program_id in self.db.feature_map.values():
            self.assertNotEqual(program_id, "test_prog")
        
        # Verify islands are cleaned up
        for island in self.db.islands:
            self.assertNotIn("test_prog", island)
        
        # Verify archive is cleaned up
        self.assertNotIn("test_prog", self.db.archive)

    def test_map_elites_non_elite_program_removal_priority(self):
        """Test that non-elite programs are removed before elite programs"""
        # Create programs that will be in feature cells (elite)
        elite_programs = []
        for i in range(4):
            program = Program(
                id=f"elite{i}",
                code=f"def elite{i}():\n    return {'x' * (i * 300)}",  # Different complexity
                metrics={"score": 0.5 + (i * 0.1)},
            )
            elite_programs.append(program)
            self.db.add(program)
        
        # Create programs that won't be in feature cells (non-elite)
        non_elite_programs = []
        for i in range(8):
            program = Program(
                id=f"non_elite{i}",
                code="def non_elite():\n    return 'same'",  # Same code = same feature cell
                metrics={"score": 0.1 + (i * 0.01)},  # Lower scores
            )
            non_elite_programs.append(program)
            self.db.add(program)
        
        # Get the feature map programs (should be elite programs)
        feature_map_programs = set(self.db.feature_map.values())
        
        # Verify elite programs are in feature map
        for program in elite_programs:
            if program.id in self.db.programs:  # Some might have been replaced
                # Check if this program's feature cell is occupied
                coords = self.db._calculate_feature_coords(program)
                feature_key = self.db._feature_coords_to_key(coords)
                if feature_key in self.db.feature_map:
                    # This program or a better one in the same cell should be in the feature map
                    self.assertIn(self.db.feature_map[feature_key], self.db.programs)
        
        # Population should be limited to config.population_size
        self.assertEqual(len(self.db.programs), self.config.population_size)
        
        # Most programs in feature_map should still exist (diversity preserved)
        remaining_feature_programs = [
            pid for pid in feature_map_programs if pid in self.db.programs
        ]
        self.assertGreater(len(remaining_feature_programs), 0)


if __name__ == "__main__":
    unittest.main()