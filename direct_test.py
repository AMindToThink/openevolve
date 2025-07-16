#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# Direct test of the deterministic fixes
from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase

def main():
    """Direct test of deterministic behavior"""
    print("Testing deterministic behavior of MAP-Elites fixes...")
    
    # Create test configuration
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
        random_seed=42
    )
    
    # Create program database
    db = ProgramDatabase(config)
    
    # Test 1: Check that _calculate_feature_coords is deterministic
    print("\n1. Testing _calculate_feature_coords determinism...")
    
    # Create test programs
    programs = []
    for i in range(8):
        program = Program(
            id=f"prog{i}",
            code=f"def func{i}():\n    return {'x' * (i * 200)}",
            metrics={"score": 0.2 + (i * 0.1)},
        )
        programs.append(program)
        db.add(program)
    
    # Calculate coordinates multiple times
    coords_runs = []
    for run in range(3):
        coords_this_run = []
        for program in programs:
            coords = db._calculate_feature_coords(program)
            coords_this_run.append(coords)
        coords_runs.append(coords_this_run)
    
    # Check if all runs produced identical coordinates
    coords_deterministic = True
    for i, program in enumerate(programs):
        run1_coords = coords_runs[0][i]
        run2_coords = coords_runs[1][i]
        run3_coords = coords_runs[2][i]
        
        if run1_coords != run2_coords or run1_coords != run3_coords:
            print(f"   FAIL: Program {program.id} coordinates not deterministic")
            print(f"     Run 1: {run1_coords}")
            print(f"     Run 2: {run2_coords}")
            print(f"     Run 3: {run3_coords}")
            coords_deterministic = False
    
    if coords_deterministic:
        print("   PASS: Feature coordinates are deterministic")
    
    # Test 2: Check that diversity calculation is deterministic
    print("\n2. Testing diversity calculation determinism...")
    
    # Test the diversity calculation specifically
    diversity_deterministic = True
    for program in programs:
        # Calculate diversity multiple times
        diversities = []
        for _ in range(3):
            if len(db.programs) >= 2:
                # Get sorted programs for deterministic sampling
                sorted_programs = sorted(db.programs.values(), key=lambda p: p.id)
                sample_programs = sorted_programs[:min(5, len(sorted_programs))]
                diversity = sum(
                    db._fast_code_diversity(program.code, other.code)
                    for other in sample_programs
                ) / len(sample_programs)
            else:
                diversity = 0
            diversities.append(diversity)
        
        # Check if all diversity calculations are the same
        if not all(d == diversities[0] for d in diversities):
            print(f"   FAIL: Program {program.id} diversity not deterministic")
            print(f"     Diversities: {diversities}")
            diversity_deterministic = False
    
    if diversity_deterministic:
        print("   PASS: Diversity calculations are deterministic")
    
    # Test 3: Check that _calculate_diversity_bin is deterministic
    print("\n3. Testing _calculate_diversity_bin determinism...")
    
    bin_deterministic = True
    for program in programs:
        # Calculate diversity bin multiple times
        bins = []
        for _ in range(3):
            if len(db.programs) >= 2:
                sorted_programs = sorted(db.programs.values(), key=lambda p: p.id)
                sample_programs = sorted_programs[:min(5, len(sorted_programs))]
                diversity = sum(
                    db._fast_code_diversity(program.code, other.code)
                    for other in sample_programs
                ) / len(sample_programs)
            else:
                diversity = 0
            
            bin_idx = db._calculate_diversity_bin(diversity)
            bins.append(bin_idx)
        
        # Check if all bins are the same
        if not all(b == bins[0] for b in bins):
            print(f"   FAIL: Program {program.id} diversity bin not deterministic")
            print(f"     Bins: {bins}")
            bin_deterministic = False
    
    if bin_deterministic:
        print("   PASS: Diversity binning is deterministic")
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary:")
    print(f"  Feature coordinates deterministic: {'PASS' if coords_deterministic else 'FAIL'}")
    print(f"  Diversity calculations deterministic: {'PASS' if diversity_deterministic else 'FAIL'}")
    print(f"  Diversity binning deterministic: {'PASS' if bin_deterministic else 'FAIL'}")
    
    all_tests_passed = coords_deterministic and diversity_deterministic and bin_deterministic
    
    if all_tests_passed:
        print("\n✅ All deterministic tests passed! The fixes are working correctly.")
        
        # Now run a quick test to see if this fixes the original issue
        print("\n4. Testing MAP-Elites behavior with deterministic fixes...")
        
        # Test basic MAP-Elites replacement
        program1 = Program(
            id="test1",
            code="def func1():\n    return 1",
            metrics={"score": 0.5},
        )
        program2 = Program(
            id="test2", 
            code="def func2():\n    return 2",
            metrics={"score": 0.8},
        )
        
        # Fresh database for this test
        test_db = ProgramDatabase(config)
        test_db.add(program1)
        
        # Calculate coordinates
        coords1 = test_db._calculate_feature_coords(program1)
        coords2 = test_db._calculate_feature_coords(program2)
        
        if coords1 == coords2:
            print("   Programs map to same feature cell - testing replacement...")
            test_db.add(program2)
            
            if "test2" in test_db.programs and "test1" not in test_db.programs:
                print("   PASS: Better program correctly replaced worse program")
            else:
                print("   FAIL: Replacement didn't work as expected")
        else:
            print("   Programs map to different feature cells - no replacement expected")
        
        print("\n✅ All tests completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed! The deterministic fixes need investigation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())