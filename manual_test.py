#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase

def test_deterministic_behavior():
    """Test that our deterministic fixes work as expected"""
    print("Testing deterministic behavior of MAP-Elites fixes...")
    print("=" * 60)
    
    # Create the same configuration as in the test
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
        random_seed=42
    )
    
    # Test 1: Basic MAP-Elites replacement test (same as in test_map_elites_replacement_basic)
    print("1. Testing MAP-Elites replacement basic behavior...")
    
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
    print(f"   Added prog1, database has {len(db.programs)} programs")
    
    # Verify program1 is in the database
    assert "prog1" in db.programs, "prog1 should be in database"
    print("   ✅ prog1 is in database")
    
    # Calculate feature coords to verify they're the same
    coords1 = db._calculate_feature_coords(program1)
    coords2 = db._calculate_feature_coords(program2)
    
    print(f"   prog1 coords: {coords1}")
    print(f"   prog2 coords: {coords2}")
    
    # They should have the same coordinates (same feature cell)
    assert coords1 == coords2, f"Coordinates should be the same: {coords1} != {coords2}"
    print("   ✅ Programs have same coordinates (same feature cell)")
    
    # Add second program (should replace first due to better score)
    db.add(program2)
    print(f"   Added prog2, database has {len(db.programs)} programs")
    
    # Verify program2 is in the database
    assert "prog2" in db.programs, "prog2 should be in database"
    print("   ✅ prog2 is in database")
    
    # Verify program1 was removed (replaced in feature cell)
    assert "prog1" not in db.programs, "prog1 should be removed from database"
    print("   ✅ prog1 was removed from database")
    
    # Verify feature map contains program2
    feature_key = db._feature_coords_to_key(coords2)
    assert feature_key in db.feature_map, f"Feature key {feature_key} should be in feature_map"
    assert db.feature_map[feature_key] == "prog2", f"Feature map should contain prog2: {db.feature_map[feature_key]}"
    print("   ✅ Feature map correctly contains prog2")
    
    print("   ✅ PASS: MAP-Elites replacement basic test")
    
    # Test 2: Test deterministic coordinate calculation across multiple runs
    print("\n2. Testing deterministic coordinate calculation...")
    
    # Create multiple databases and add same programs
    databases = []
    for i in range(3):
        db = ProgramDatabase(config)
        for j in range(5):
            program = Program(
                id=f"test_prog_{j}",
                code=f"def test_func_{j}():\n    return {'x' * (j * 50)}",
                metrics={"score": 0.1 + (j * 0.1)},
            )
            db.add(program)
        databases.append(db)
    
    # Check that all databases produce the same coordinates
    for j in range(5):
        program = Program(
            id=f"test_prog_{j}",
            code=f"def test_func_{j}():\n    return {'x' * (j * 50)}",
            metrics={"score": 0.1 + (j * 0.1)},
        )
        
        coords_list = []
        for db in databases:
            coords = db._calculate_feature_coords(program)
            coords_list.append(coords)
        
        # All coordinates should be the same
        first_coords = coords_list[0]
        for i, coords in enumerate(coords_list[1:], 1):
            assert coords == first_coords, f"DB{i} coords {coords} != DB0 coords {first_coords} for program {program.id}"
        
        print(f"   ✅ Program {program.id}: consistent coords {first_coords}")
    
    print("   ✅ PASS: Deterministic coordinate calculation test")
    
    # Test 3: Population limit enforcement test
    print("\n3. Testing population limit enforcement...")
    
    # Create database with small population limit
    small_config = DatabaseConfig(
        population_size=5,
        archive_size=3,
        num_islands=2,
        feature_dimensions=["complexity", "score"],
        feature_bins=3,
        exploration_ratio=0.3,
        exploitation_ratio=0.4,
        elite_selection_ratio=0.2,
        db_path=None,
        random_seed=42
    )
    
    small_db = ProgramDatabase(small_config)
    
    # Add more programs than the limit
    for i in range(8):
        program = Program(
            id=f"pop_test_{i}",
            code=f"def pop_func_{i}():\n    return {'y' * (i * 100)}",
            metrics={"score": 0.2 + (i * 0.1)},
        )
        small_db.add(program)
    
    print(f"   Added 8 programs, database has {len(small_db.programs)} programs")
    print(f"   Population limit is {small_config.population_size}")
    
    # Verify population limit was enforced
    assert len(small_db.programs) == small_config.population_size, f"Population should be limited to {small_config.population_size}"
    print("   ✅ Population limit correctly enforced")
    
    # Verify that programs in feature_map are preserved
    feature_map_programs = set(small_db.feature_map.values())
    for program_id in feature_map_programs:
        assert program_id in small_db.programs, f"Feature map program {program_id} should be in database"
    print("   ✅ All feature map programs are preserved in database")
    
    # Verify that the feature map structure is maintained
    assert len(small_db.feature_map) > 0, "Feature map should not be empty"
    print("   ✅ Feature map structure is maintained")
    
    print("   ✅ PASS: Population limit enforcement test")
    
    print("\n" + "=" * 60)
    print("🎉 All manual tests passed!")
    print("The deterministic fixes are working correctly!")
    
    return True

if __name__ == "__main__":
    try:
        success = test_deterministic_behavior()
        print("\n✅ SUCCESS: All deterministic fixes verified!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ FAILURE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)