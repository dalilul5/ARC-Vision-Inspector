import json
import csv
import os
from typing import List, Dict, Any
from collections import Counter

def save_json(data: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def save_csv(rows: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return

    fieldnames = sorted(set(k for row in rows for k in row.keys()))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def summarize_global_failures(sample_reports: List[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter()
    for report in sample_reports:
        for failure in report.get("global_failures", []):
            counter[failure] += 1
    return dict(counter)

def calculate_severity_penalty(failure_type: str, failure_data: Dict[str, Any] = None) -> float:
    # Primary failures
    if failure_type == "missing_object": return 40.0
    if failure_type == "spurious_object": return 30.0
    if failure_type == "incorrect_color_mapping": return 20.0
    
    # Transformation failures
    if not failure_data:
        return 10.0
        
    if failure_type == "shape_mismatch": return 15.0
    if failure_type == "orientation_or_reflection_mismatch": return 10.0
    if failure_type == "color_mismatch": return 10.0
    
    if failure_type == "area_mismatch":
        expected = failure_data.get("expected_area", 1)
        if expected == 0: expected = 1
        ratio_diff = abs(failure_data.get("magnitude", 0) / expected)
        return min(10.0, ratio_diff * 20.0)
        
    if failure_type == "position_or_translation_mismatch":
        dist = failure_data.get("magnitude", 0)
        if dist <= 2: return 2.0
        if dist <= 5: return 5.0
        return 10.0
        
    return 0.0

def aggregate_hierarchical_failures(sample_reports: List[Dict[str, Any]], sample_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats = {
        "primary_failures": Counter(),
        "secondary_symptoms": Counter(),
        "transformation_failures": Counter(),
        "task_failure_density": {},
        "total_matched_pairs": 0,
        "total_tasks": len(sample_reports),
        "overall_health_score": 0
    }
    
    overall_health_sum = 0
    
    for meta, report in zip(sample_metadata, sample_reports):
        task_id = f"{meta['task_name']}_{meta['split']}_{meta['index']}"
        global_fails = report.get("global_failures", [])
        
        matched_pairs_count = len(report.get("matching", {}).get("matches", []))
        stats["total_matched_pairs"] += matched_pairs_count
        
        penalty_sum = 0.0
        has_primary = False
        
        # Apply primary penalties
        if "missing_object" in global_fails:
            stats["primary_failures"]["missing_object"] += 1
            has_primary = True
            penalty_sum += calculate_severity_penalty("missing_object")
        if "spurious_object" in global_fails:
            stats["primary_failures"]["spurious_object"] += 1
            has_primary = True
            penalty_sum += calculate_severity_penalty("spurious_object")
            
        # Pure rule failures vs secondary symptoms
        if "global_color_distribution_shift" in global_fails:
            if has_primary:
                stats["secondary_symptoms"]["global_color_distribution_shift"] += 1
            else:
                stats["primary_failures"]["incorrect_color_mapping"] += 1
                penalty_sum += calculate_severity_penalty("incorrect_color_mapping")
                
        # Transformation failures iteration
        pair_diags = report.get("pair_diagnostics", [])
        all_trans_fails_types = []
        for pd in pair_diags:
            for f in pd.get("pair_failures", []):
                f_type = f.get("type", "")
                if f_type != "no_pair_level_failure":
                    stats["transformation_failures"][f_type] += 1
                    all_trans_fails_types.append(f_type)
                    penalty_sum += calculate_severity_penalty(f_type, f)
                    
        # Health Score Calculation using Weighted Matrix
        health_score = max(0.0, 100.0 - penalty_sum)
        overall_health_sum += health_score
                    
        # Calculate density
        density = len(global_fails) + len(all_trans_fails_types)
        
        # Identify dominant issue
        dominant = []
        if has_primary:
            dominant.extend([f for f in ["missing_object", "spurious_object"] if f in global_fails])
        if len(dominant) == 0 and "global_color_distribution_shift" in global_fails:
            dominant.append("incorrect_color_mapping")
            
        if all_trans_fails_types:
            most_common_trans = Counter(all_trans_fails_types).most_common(1)[0][0]
            dominant.append(most_common_trans)
            
        stats["task_failure_density"][task_id] = {
            "density": density,
            "dominant_issues": " + ".join(dominant) if dominant else "None",
            "health_score": round(health_score, 1)
        }
        
    stats["overall_health_score"] = round(overall_health_sum / max(1, len(sample_reports)), 1)
        
    # Convert counters to regular dicts for JSON serialization
    return {
        "primary_failures": dict(stats["primary_failures"]),
        "secondary_symptoms": dict(stats["secondary_symptoms"]),
        "transformation_failures": dict(stats["transformation_failures"]),
        "task_failure_density": stats["task_failure_density"],
        "total_matched_pairs": stats["total_matched_pairs"],
        "total_tasks": stats["total_tasks"],
        "overall_health_score": stats["overall_health_score"]
    }

def print_executive_summary(total_tasks: int, hierarchy_stats: Dict[str, Any]):
    print("\n========================================")
    print("EXECUTIVE SUMMARY - BATCH EVALUATION V2.4")
    print("========================================")
    print(f"Total Tasks Evaluated: {hierarchy_stats['total_tasks']}")
    print(f"Total Matched Object Pairs: {hierarchy_stats['total_matched_pairs']}\n")
    
    print("=== TOP PRIMARY STRUCTURAL FAILURES (ROOT CAUSES) ===")
    primaries = sorted(hierarchy_stats["primary_failures"].items(), key=lambda x: x[1], reverse=True)
    for i, (fail, count) in enumerate(primaries):
        pct = (count / hierarchy_stats['total_tasks']) * 100 if hierarchy_stats['total_tasks'] > 0 else 0
        marker = "  <-- FOKUS UTAMA!" if i == 0 else ""
        print(f"  {i+1}. {fail.ljust(25)} : {count:3} tasks ({pct:.1f}%){marker}")
    if not primaries:
        print("  None detected.")

    print("\n=== NORMALIZED TRANSFORMATION FAILURE RATES ===")
    trans = sorted(hierarchy_stats["transformation_failures"].items(), key=lambda x: x[1], reverse=True)
    for i, (fail, count) in enumerate(trans):
        pct = (count / hierarchy_stats['total_matched_pairs']) * 100 if hierarchy_stats['total_matched_pairs'] > 0 else 0
        print(f"  {i+1}. {fail.ljust(25)} : {pct:5.1f}%  ({count}/{hierarchy_stats['total_matched_pairs']} pairs)")
    if not trans:
        print("  None detected.")

    print("\n=== OVERALL TASK HEALTH SCORE ===")
    avg_health = hierarchy_stats['overall_health_score']
    if avg_health >= 70:
        health_status = "[GREEN - Good]"
    elif avg_health >= 40:
        health_status = "[YELLOW - Needs Attention]"
    else:
        health_status = "[RED - Critical]"
    print(f"  Average Health Score: {avg_health} / 100  {health_status}")

    print("\n=== PRIORITY INVESTIGATION TARGETS ===")
    densities = sorted(hierarchy_stats["task_failure_density"].items(), key=lambda x: x[1]["health_score"])
    for i, (task_id, info) in enumerate(densities[:5]):
        h = info['health_score']
        color = "RED" if h < 40 else ("YELLOW" if h < 70 else "GREEN")
        issue_str = f" - Fix {info['dominant_issues']} first!" if h < 100 else ""
        print(f"  Task {task_id.ljust(22)} : Health {h:3} ({color.ljust(6)}){issue_str}")

    print("========================================\n")

def flatten_report_metadata(meta: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "task_name": meta["task_name"],
        "split": meta["split"],
        "index": meta["index"],
        "input_objects": report["object_counts"]["input"],
        "pred_objects": report["object_counts"]["pred"],
        "true_objects": report["object_counts"]["true"],
        "global_failures": "|".join(report.get("global_failures", [])),
    }
    return row
