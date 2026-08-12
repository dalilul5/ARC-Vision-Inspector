from typing import Dict, Any, List

def to_markdown_context(report: Dict[str, Any]) -> str:
    """Generates human-readable Markdown diagnostic card for LLM self-correction prompting."""
    lines = []
    lines.append("## 🔍 Diagnostic Evaluation Report")
    lines.append(f"**Global Failures**: `{', '.join(report.get('global_failures', [])) or 'None'}`")
    
    pair_diags = report.get("pair_diagnostics", [])
    if not pair_diags:
        lines.append("\n*No matched object pairs.*")
    else:
        lines.append("\n### Matched Object Pair Diagnostics")
        for pd in pair_diags:
            pred_id = pd.get("pred_obj_id")
            true_id = pd.get("true_obj_id")
            failures = pd.get("pair_failures", [])
            lines.append(f"\n- **Object Pair (Pred #{pred_id} ↔ Target #{true_id})**:")
            for f in failures:
                ftype = f.get("type")
                if ftype == "no_pair_level_failure":
                    lines.append("  - ✅ Exact Match (No failure)")
                elif ftype == "position_or_translation_mismatch":
                    lines.append(f"  - 📍 **Position Error**: Shifted by `(Δrow={f.get('delta_row')}, Δcol={f.get('delta_col')})` | Distance: `{f.get('magnitude')} px`")
                elif ftype == "orientation_or_reflection_mismatch":
                    lines.append(f"  - 🔄 **Orientation Error**: Shape transformed via `{f.get('shape_relation')}`. Action: `{f.get('suggested_correction')}`")
                elif ftype == "color_mismatch":
                    lines.append(f"  - 🎨 **Color Error**: Predicted color `{f.get('predicted_color')}`, expected `{f.get('expected_color')}`")
                elif ftype == "area_mismatch":
                    lines.append(f"  - 📏 **Area Error**: Predicted area `{f.get('predicted_area')}`, expected `{f.get('expected_area')}`")
                else:
                    lines.append(f"  - ⚠️ **{ftype}**: magnitude `{f.get('magnitude')}`")

    return "\n".join(lines)

def to_action_vector(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Converts diagnostic failures into discrete actionable recovery operations for RL or program synthesis engines."""
    actions = []
    
    matching = report.get("matching", {})
    for t_id in matching.get("unmatched_true", []):
        actions.append({
            "action": "add_missing_object",
            "true_obj_id": t_id
        })
    for p_id in matching.get("unmatched_pred", []):
        actions.append({
            "action": "remove_spurious_object",
            "pred_obj_id": p_id
        })

    for pd in report.get("pair_diagnostics", []):
        pred_id = pd.get("pred_obj_id")
        for f in pd.get("pair_failures", []):
            ftype = f.get("type")
            if ftype == "position_or_translation_mismatch":
                actions.append({
                    "action": "translate",
                    "pred_obj_id": pred_id,
                    "delta_row": f.get("delta_row"),
                    "delta_col": f.get("delta_col")
                })
            elif ftype == "orientation_or_reflection_mismatch":
                actions.append({
                    "action": "transform_shape",
                    "pred_obj_id": pred_id,
                    "suggested_correction": f.get("suggested_correction"),
                    "shape_relation": f.get("shape_relation")
                })
            elif ftype == "color_mismatch":
                actions.append({
                    "action": "recolor",
                    "pred_obj_id": pred_id,
                    "predicted_color": f.get("predicted_color"),
                    "expected_color": f.get("expected_color")
                })
            elif ftype == "area_mismatch":
                actions.append({
                    "action": "resize",
                    "pred_obj_id": pred_id,
                    "predicted_area": f.get("predicted_area"),
                    "expected_area": f.get("expected_area")
                })

    return actions

def to_json_api(report: Dict[str, Any]) -> Dict[str, Any]:
    """Generates clean API JSON payload containing diagnostic cards, action vectors, and summary stats."""
    return {
        "summary": {
            "object_counts": report.get("object_counts"),
            "global_failures": report.get("global_failures", []),
            "color_match": report.get("color_distribution", {}).get("match", False)
        },
        "action_vector": to_action_vector(report),
        "markdown_card": to_markdown_context(report),
        "raw_diagnostics": {
            "matching": report.get("matching"),
            "pair_diagnostics": report.get("pair_diagnostics")
        }
    }
