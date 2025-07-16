#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# Check if the deterministic changes work
from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase

def test_deterministic_feature_coords():
    """Test that feature coordinate calculation is deterministic"""
    print("Testing deterministic feature coordinate calculation...")
    
    config = DatabaseConfig(
        population_size=10,
        archive_size=5,
        num_islands=2,
        feature_dimensions=["complexity", "diversity"],
        feature_bins=3,
        exploration_ratio=0.3,
        exploitation_ratio=0.4,
        elite_selection_ratio=0.2,
        db_path=None,
        random_seed=42  # Set seed for reproducibility
    )
    
    # Create multiple program databases
    db1 = ProgramDatabase(config)
    db2 = ProgramDatabase(config)
    
    # Add the same programs to both databases
    programs = []
    for i in range(10):
        program = Program(
            id=f"prog{i}",
            code=f"def func{i}():\n    return {'x' * (i * 100)}",
            metrics={"score": 0.1 + (i * 0.05)},
        )
        programs.append(program)
    
    # Add programs to both databases
    for program in programs:
        db1.add(program)
        db2.add(program)
    
    # Check that feature coordinates are the same
    all_coords_match = True
    for program in programs:
        coords1 = db1._calculate_feature_coords(program)
        coords2 = db2._calculate_feature_coords(program)
        
        if coords1 != coords2:
            print(f"  FAIL: Program {program.id} has different coordinates:")
            print(f"    DB1: {coords1}")
            print(f"    DB2: {coords2}")
            all_coords_match = False
    
    if all_coords_match:
        print("  PASS: All feature coordinates are deterministic")
    else:
        print("  FAIL: Feature coordinates are not deterministic")
    
    return all_coords_match

def test_deterministic_diversity_bin():
    """Test that diversity binning is deterministic"""
    print("Testing deterministic diversity binning...")
    
    config = DatabaseConfig(
        population_size=10,
        archive_size=5,
        num_islands=2,
        feature_dimensions=["diversity"],
        feature_bins=3,
        exploration_ratio=0.3,
        exploitation_ratio=0.4,
        elite_selection_ratio=0.2,
        db_path=None,
        random_seed=42
    )
    
    # Create multiple program databases
    db1 = ProgramDatabase(config)
    db2 = ProgramDatabase(config)
    
    # Add the same programs to both databases
    programs = []
    for i in range(10):
        program = Program(
            id=f"prog{i}",
            code=f"def func{i}():\n    return {'x' * (i * 50)}",
            metrics={"score": 0.1 + (i * 0.05)},
        )
        programs.append(program)
    
    # Add programs to both databases
    for program in programs:
        db1.add(program)
        db2.add(program)
    
    # Check that diversity bins are the same
    all_bins_match = True
    for program in programs:
        # Calculate diversity for this program
        diversity1 = 0
        diversity2 = 0
        
        if len(db1.programs) >= 2:
            # Get sorted programs for deterministic sampling
            sorted_programs1 = sorted(db1.programs.values(), key=lambda p: p.id)
            sample_programs1 = sorted_programs1[:min(5, len(sorted_programs1))]
            diversity1 = sum(
                db1._fast_code_diversity(program.code, other.code)
                for other in sample_programs1
            ) / len(sample_programs1)
        
        if len(db2.programs) >= 2:
            sorted_programs2 = sorted(db2.programs.values(), key=lambda p: p.id)
            sample_programs2 = sorted_programs2[:min(5, len(sorted_programs2))]
            diversity2 = sum(
                db2._fast_code_diversity(program.code, other.code)
                for other in sample_programs2
            ) / len(sample_programs2)
        
        bin1 = db1._calculate_diversity_bin(diversity1)
        bin2 = db2._calculate_diversity_bin(diversity2)
        
        if bin1 != bin2:
            print(f"  FAIL: Program {program.id} has different diversity bins:")
            print(f"    DB1: {bin1} (diversity: {diversity1})")
            print(f"    DB2: {bin2} (diversity: {diversity2})")
            all_bins_match = False
    
    if all_bins_match:
        print("  PASS: All diversity bins are deterministic")
    else:
        print("  FAIL: Diversity bins are not deterministic")
    
    return all_bins_match

def main():
    """Run all verification tests"""
    print("Running verification tests for deterministic fixes...")
    print("=" * 60)
    
    # Test 1: Deterministic feature coordinates
    test1_passed = test_deterministic_feature_coords()
    
    print()
    
    # Test 2: Deterministic diversity binning
    test2_passed = test_deterministic_diversity_bin()
    
    print()
    print("=" * 60)
    print("Test Results:")
    print(f"  Feature coordinates deterministic: {'PASS' if test1_passed else 'FAIL'}")
    print(f"  Diversity binning deterministic: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n✅ All tests passed! The deterministic fixes are working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed! The deterministic fixes need more work.")
        return 1

if __name__ == "__main__":
    sys.exit(main())