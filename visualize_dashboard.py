import json
import os

JSON_PATH = "outputs/batch_run/batch_executive_summary.json"
HTML_PATH = "outputs/batch_run/dashboard.html"

def generate_html(data):
    # Prepare data for Chart.js
    trans_labels = list(data.get("transformation_failures", {}).keys())
    trans_counts = list(data.get("transformation_failures", {}).values())
    
    tasks = list(data.get("task_failure_density", {}).keys())
    health_scores = [info["health_score"] for info in data.get("task_failure_density", {}).values()]
    
    # Assign colors based on health
    bar_colors = []
    for score in health_scores:
        if score < 40:
            bar_colors.append("'rgba(255, 99, 132, 0.6)'")  # Red
        elif score < 70:
            bar_colors.append("'rgba(255, 206, 86, 0.6)'")   # Yellow
        else:
            bar_colors.append("'rgba(75, 192, 192, 0.6)'")   # Green

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ARC-Vision-Inspector V2.4 Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; margin: 0; padding: 20px; }}
        h1 {{ color: #333; text-align: center; }}
        .summary-cards {{ display: flex; justify-content: center; gap: 20px; margin-bottom: 30px; }}
        .card {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; min-width: 200px; }}
        .card h3 {{ margin: 0 0 10px 0; color: #666; font-size: 14px; text-transform: uppercase; }}
        .card .value {{ font-size: 32px; font-weight: bold; color: #2c3e50; }}
        .chart-container {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px; max-width: 800px; margin-left: auto; margin-right: auto; }}
    </style>
</head>
<body>

    <h1>ARC-Vision-Inspector V2.4 Dashboard</h1>

    <div class="summary-cards">
        <div class="card">
            <h3>Total Tasks Evaluated</h3>
            <div class="value">{data.get("total_tasks", 0)}</div>
        </div>
        <div class="card">
            <h3>Total Matched Pairs</h3>
            <div class="value">{data.get("total_matched_pairs", 0)}</div>
        </div>
        <div class="card">
            <h3>Average Health Score</h3>
            <div class="value">{data.get("overall_health_score", 0)}</div>
        </div>
    </div>

    <div class="chart-container">
        <canvas id="transChart"></canvas>
    </div>
    
    <div class="chart-container">
        <canvas id="healthChart"></canvas>
    </div>

    <script>
        // Transformation Failures Chart
        const ctxTrans = document.getElementById('transChart').getContext('2d');
        new Chart(ctxTrans, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(trans_labels)},
                datasets: [{{
                    label: 'Absolute Occurrences (All Pairs)',
                    data: {json.dumps(trans_counts)},
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ title: {{ display: true, text: 'Transformation Failures Breakdown' }} }},
                scales: {{ y: {{ beginAtZero: true }} }}
            }}
        }});

        // Task Health Score Chart
        const ctxHealth = document.getElementById('healthChart').getContext('2d');
        new Chart(ctxHealth, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(tasks)},
                datasets: [{{
                    label: 'Task Health Score (0-100)',
                    data: {json.dumps(health_scores)},
                    backgroundColor: [{', '.join(bar_colors)}],
                    borderColor: '#333',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ title: {{ display: true, text: 'Task Health Scores (Red < 40, Yellow < 70, Green >= 70)' }} }},
                scales: {{ y: {{ beginAtZero: true, max: 100 }} }}
            }}
        }});
    </script>
</body>
</html>"""
    
    os.makedirs(os.path.dirname(HTML_PATH), exist_ok=True)
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Dashboard generated successfully at: {HTML_PATH}")

def main():
    if not os.path.exists(JSON_PATH):
        print(f"Error: Could not find {JSON_PATH}. Run evaluate_arc.py first.")
        return
        
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    generate_html(data)

if __name__ == "__main__":
    main()