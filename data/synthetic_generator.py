import os
import json
import random
from typing import List, Dict, Any

OUTPUT_DIR = "data/synthetic_benchmark"

def create_base_grid(rows=10, cols=10):
    return [[0 for _ in range(cols)] for _ in range(rows)]

def place_box(grid, min_r, min_c, max_r, max_c, color):
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            grid[r][c] = color

def generate_synthetic_benchmark(num_cases=50) -> List[Dict[str, Any]]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cases = []
    
    failure_types = ["translation", "rotation", "color", "missing", "spurious", "exact", "composite"]
    
    for i in range(num_cases):
        ftype = failure_types[i % len(failure_types)]
        case_name = f"synthetic_case_{i+1:03d}_{ftype}"
        
        inp = create_base_grid(8, 8)
        tru = create_base_grid(8, 8)
        prd = create_base_grid(8, 8)
        
        # Base input object (Red 2x2 box at top-left)
        place_box(inp, 1, 1, 2, 2, 2)
        place_box(tru, 1, 1, 2, 2, 2)
        
        expected_failures = []
        
        if ftype == "translation":
            # Shifted prediction 2 columns right
            place_box(prd, 1, 3, 2, 4, 2)
            expected_failures.append("position_or_translation_mismatch")
        elif ftype == "rotation":
            # 2x2 turned into 1x4 horizontal strip
            place_box(prd, 1, 1, 1, 4, 2)
            expected_failures.append("orientation_or_reflection_mismatch")
        elif ftype == "color":
            # Predicted blue (1) instead of red (2)
            place_box(prd, 1, 1, 2, 2, 1)
            expected_failures.append("color_mismatch")
        elif ftype == "missing":
            # Prediction leaves canvas blank
            pass
            expected_failures.append("missing_object")
        elif ftype == "spurious":
            # Prediction includes extra green box
            place_box(prd, 1, 1, 2, 2, 2)
            place_box(prd, 5, 5, 6, 6, 3)
            expected_failures.append("spurious_object")
        elif ftype == "composite":
            # Outer box red (2), inner box blue (1)
            place_box(inp, 1, 1, 5, 5, 2)
            place_box(inp, 2, 2, 4, 4, 1)
            place_box(tru, 1, 1, 5, 5, 2)
            place_box(tru, 2, 2, 4, 4, 1)
            # Pred shifts inner box
            place_box(prd, 1, 1, 5, 5, 2)
            place_box(prd, 3, 3, 5, 5, 1)
            expected_failures.append("position_or_translation_mismatch")
        else:  # exact match
            place_box(prd, 1, 1, 2, 2, 2)
            
        case_data = {
            "name": case_name,
            "input": inp,
            "true": tru,
            "pred": prd,
            "expected_failures": expected_failures
        }
        
        filepath = os.path.join(OUTPUT_DIR, f"{case_name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(case_data, f, indent=2)
            
        cases.append(case_data)
        
    print(f"Generated {len(cases)} synthetic benchmark test cases in {OUTPUT_DIR}")
    return cases

if __name__ == "__main__":
    generate_synthetic_benchmark()
