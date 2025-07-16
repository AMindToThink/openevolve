#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase

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

print("Initial state:")
print(f"Database programs: {list(db.programs.keys())}")
print(f"Feature map: {db.feature_map}")

# Add first program
db.add(program1)
print("\nAfter adding program1:")
print(f"Database programs: {list(db.programs.keys())}")
print(f"Feature map: {db.feature_map}")

# Calculate feature coords to verify they're the same
coords1 = db._calculate_feature_coords(program1)
coords2 = db._calculate_feature_coords(program2)
print(f"\nFeature coordinates:")
print(f"Program1 coords: {coords1}")
print(f"Program2 coords: {coords2}")
print(f"Are coordinates the same? {coords1 == coords2}")

# Add second program (should replace first due to better score)
db.add(program2)
print("\nAfter adding program2:")
print(f"Database programs: {list(db.programs.keys())}")
print(f"Feature map: {db.feature_map}")

# Check test conditions
print(f"\nTest results:")
print(f"prog2 in database: {'prog2' in db.programs}")
print(f"prog1 in database: {'prog1' in db.programs}")

# Check feature map
feature_key = db._feature_coords_to_key(coords2)
print(f"Feature key: {feature_key}")
print(f"Feature map contains prog2: {db.feature_map.get(feature_key) == 'prog2'}")

# Test passed?
test_passed = (
    "prog2" in db.programs and
    "prog1" not in db.programs and
    db.feature_map.get(feature_key) == "prog2"
)

print(f"\nTEST PASSED: {test_passed}")