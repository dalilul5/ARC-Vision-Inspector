import os
import json
import gradio as gr

from core.diagnostics import DiagnosticEngine, diagnose_failure
from core.visualization import render_grid, overlay_objects
from core.formatters import to_markdown_context, to_action_vector, to_json_api
from core.grid_utils import validate_grid
from demo_data import EXAMPLE_INPUT, EXAMPLE_PRED, EXAMPLE_TRUE
import core.matching as matching_module

CONFIG_PATH = "weights.json"

def parse_grid_str(grid_str: str):
    try:
        grid_data = json.loads(grid_str)
    except Exception as e:
        raise ValueError(f"Invalid JSON format: {e}")
    return validate_grid(grid_data)


def run_single_inspection(input_str: str, pred_str: str, true_str: str):
    try:
        input_grid = parse_grid_str(input_str)
        pred_grid = parse_grid_str(pred_str)
        true_grid = parse_grid_str(true_str)

        engine = DiagnosticEngine()
        report = engine.run(input_grid, pred_grid, true_grid)

        input_img = overlay_objects(render_grid(input_grid), input_grid)
        pred_img = overlay_objects(render_grid(pred_grid), pred_grid)
        true_img = overlay_objects(render_grid(true_grid), true_grid)

        markdown_card = report.get("markdown_card", "")
        action_vector_str = json.dumps(report.get("action_vector", []), indent=2)
        report_json = json.dumps(report, indent=2)

        return input_img, pred_img, true_img, markdown_card, action_vector_str, report_json
    except Exception as err:
        return None, None, None, f"Error: {err}", "", ""

def run_model_comparison(input_str: str, pred_a_str: str, pred_b_str: str, true_str: str):
    try:
        input_grid = parse_grid_str(input_str)
        pred_a_grid = parse_grid_str(pred_a_str)
        pred_b_grid = parse_grid_str(pred_b_str)
        true_grid = parse_grid_str(true_str)

        report_a = diagnose_failure(input_grid, pred_a_grid, true_grid)
        report_b = diagnose_failure(input_grid, pred_b_grid, true_grid)

        img_a = overlay_objects(render_grid(pred_a_grid), pred_a_grid)
        img_b = overlay_objects(render_grid(pred_b_grid), pred_b_grid)

        fails_a = len(report_a.get("global_failures", [])) + sum(len(pd.get("pair_failures", [])) for pd in report_a.get("pair_diagnostics", []))
        fails_b = len(report_b.get("global_failures", [])) + sum(len(pd.get("pair_failures", [])) for pd in report_b.get("pair_diagnostics", []))

        if fails_a < fails_b:
            winner = f"🏆 Model A performed BETTER ({fails_a} failures vs {fails_b} in Model B)"
        elif fails_b < fails_a:
            winner = f"🏆 Model B performed BETTER ({fails_b} failures vs {fails_a} in Model A)"
        else:
            winner = f"🤝 TIE ({fails_a} failures in both models)"

        summary_md = f"""
### Model Comparison Delta Analysis
- **Winner**: {winner}
- **Model A Failures**: `{fails_a}` (Global: `{report_a.get('global_failures')}`)
- **Model B Failures**: `{fails_b}` (Global: `{report_b.get('global_failures')}`)
        """

        return img_a, img_b, summary_md
    except Exception as err:
        return None, None, f"Error comparing models: {err}"

def load_weight_preset():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return (
                data.get("weight_color", 3.0),
                data.get("weight_area", 2.0),
                data.get("weight_shape", 4.0),
                data.get("weight_centroid", 2.0),
                f"Loaded weights from {CONFIG_PATH}"
            )
    return 3.0, 2.0, 4.0, 2.0, "Using default weights"

def save_weight_preset(wc, wa, ws, wct):
    data = {
        "weight_color": wc,
        "weight_area": wa,
        "weight_shape": ws,
        "weight_centroid": wct
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return f"Saved weights to {CONFIG_PATH}"

def trigger_auto_tuning():
    from weight_optimizer import optimize_matching_weights
    res = optimize_matching_weights()
    return res.get("weight_color"), res.get("weight_area"), res.get("weight_shape"), res.get("weight_centroid"), f"Optimized via Grid Search! (Acc: {res.get('benchmark_accuracy'):.2%})"

demo = gr.Blocks(title="ARC-Vision-Inspector Diagnostic Workbench")

with demo:
    gr.Markdown("# 🔍 ARC-Vision-Inspector Diagnostic Workbench (Sprint 2)")
    gr.Markdown("Object-centric visual reasoning failure analysis, side-by-side comparison, and actionable recovery vectors.")

    with gr.Tabs():
        # TAB 1: SINGLE TASK INSPECTION
        with gr.Tab("Single Task Inspector"):
            with gr.Row():
                input_img_ui = gr.Image(label="Input Grid", type="pil")
                pred_img_ui = gr.Image(label="Model Prediction", type="pil")
                true_img_ui = gr.Image(label="Ground Truth Target", type="pil")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Grid JSON Inputs")
                    input_json_in = gr.Textbox(label="Input Grid JSON", value=json.dumps(EXAMPLE_INPUT), lines=3)
                    pred_json_in = gr.Textbox(label="Prediction Grid JSON", value=json.dumps(EXAMPLE_PRED), lines=3)
                    true_json_in = gr.Textbox(label="Ground Truth Grid JSON", value=json.dumps(EXAMPLE_TRUE), lines=3)
                    run_btn = gr.Button("🔍 Diagnose Failure", variant="primary")

                with gr.Column():
                    gr.Markdown("### 📄 Diagnostic Card (Markdown Context)")
                    card_md_out = gr.Markdown()
                    with gr.Accordion("Action Vector (Recovery Actions)", open=True):
                        action_vec_out = gr.Code(language="json", lines=6)

            with gr.Accordion("Raw Diagnostic Payload (JSON API)", open=False):
                json_out = gr.Code(language="json", lines=10)

            run_btn.click(
                fn=run_single_inspection,
                inputs=[input_json_in, pred_json_in, true_json_in],
                outputs=[input_img_ui, pred_img_ui, true_img_ui, card_md_out, action_vec_out, json_out]
            )

        # TAB 2: MODEL COMPARISON
        with gr.Tab("Side-by-Side Model Comparison"):
            gr.Markdown("Compare Model A vs Model B predictions side-by-side against Ground Truth Target.")
            with gr.Row():
                comp_img_a = gr.Image(label="Model A Prediction Overlay", type="pil")
                comp_img_b = gr.Image(label="Model B Prediction Overlay", type="pil")

            with gr.Row():
                with gr.Column():
                    comp_in_grid = gr.Textbox(label="Input Grid JSON", value=json.dumps(EXAMPLE_INPUT), lines=2)
                    comp_pred_a = gr.Textbox(label="Model A Prediction JSON", value=json.dumps(EXAMPLE_PRED), lines=2)
                    comp_pred_b = gr.Textbox(label="Model B Prediction JSON (e.g. Identity)", value=json.dumps(EXAMPLE_INPUT), lines=2)
                    comp_true_grid = gr.Textbox(label="Ground Truth Target JSON", value=json.dumps(EXAMPLE_TRUE), lines=2)
                    comp_btn = gr.Button("⚡ Compare Models", variant="primary")

                with gr.Column():
                    comp_results_md = gr.Markdown()

            comp_btn.click(
                fn=run_model_comparison,
                inputs=[comp_in_grid, comp_pred_a, comp_pred_b, comp_true_grid],
                outputs=[comp_img_a, comp_img_b, comp_results_md]
            )

        # TAB 3: WEIGHT OPTIMIZATION & PRESETS
        with gr.Tab("Weight Tuning & Config"):
            gr.Markdown("Adjust Hungarian matching score weights or auto-tune via Synthetic Benchmark Grid Search.")
            wc_slider = gr.Slider(0.0, 10.0, value=3.0, step=0.5, label="Color Match Weight")
            wa_slider = gr.Slider(0.0, 10.0, value=2.0, step=0.5, label="Area Match Weight")
            ws_slider = gr.Slider(0.0, 10.0, value=4.0, step=0.5, label="Shape Match Weight")
            wct_slider = gr.Slider(0.0, 10.0, value=2.0, step=0.5, label="Centroid Proximity Weight")
            
            w_status = gr.Textbox(label="Preset Status", value="Ready")

            with gr.Row():
                save_w_btn = gr.Button("💾 Save Preset")
                load_w_btn = gr.Button("📂 Load Preset")
                tune_w_btn = gr.Button("🚀 Auto-Tune via Benchmark Grid Search", variant="primary")

            save_w_btn.click(fn=save_weight_preset, inputs=[wc_slider, wa_slider, ws_slider, wct_slider], outputs=[w_status])
            load_w_btn.click(fn=load_weight_preset, outputs=[wc_slider, wa_slider, ws_slider, wct_slider, w_status])
            tune_w_btn.click(fn=trigger_auto_tuning, outputs=[wc_slider, wa_slider, ws_slider, wct_slider, w_status])

if __name__ == "__main__":
    demo.launch()
