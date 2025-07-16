#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase

def test_map_elites_replacement_basic():
    """Test that MAP-Elites properly replaces programs in feature cells"""
    
    # Create test configuration
    config = DatabaseConfig(
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
    
    # Create database
    db = ProgramDatabase(config)
    
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
    db.add(program1)
    
    # Verify program1 is in the database
    assert "prog1" in db.programs, "Program1 should be in database after adding"
    
    # Calculate feature coords to verify they're the same
    coords1 = db._calculate_feature_coords(program1)
    coords2 = db._calculate_feature_coords(program2)
    
    print(f"Program1 coords: {coords1}")
    print(f"Program2 coords: {coords2}")
    
    # They should have the same coordinates (same feature cell)
    assert coords1 == coords2, f"Programs should have same coordinates: {coords1} != {coords2}"
    
    # Add second program (should replace first due to better score)
    db.add(program2)
    
    # Verify program2 is in the database
    assert "prog2" in db.programs, "Program2 should be in database after adding"
    
    # Verify program1 was removed (replaced in feature cell)
    assert "prog1" not in db.programs, "Program1 should be removed from database"
    
    # Verify feature map contains program2
    feature_key = db._feature_coords_to_key(coords2)
    assert db.feature_map[feature_key] == "prog2", f"Feature map should contain prog2, got {db.feature_map.get(feature_key)}"
    
    print("TEST PASSED!")
    return True

if __name__ == "__main__":
    try:
        test_map_elites_replacement_basic()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)