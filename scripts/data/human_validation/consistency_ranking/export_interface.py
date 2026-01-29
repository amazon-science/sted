#!/usr/bin/env python
"""
Export consistency ranking dataset to browser-based annotation interface.

Creates:
1. HTML interface for comparing sets of outputs
2. CSV file for offline annotation
"""

import json
import os
import csv
import argparse
from typing import List, Dict


def create_set_viewer_html(pair: Dict, pair_id: str) -> str:
    """Create HTML page for viewing a single ranking pair."""

    set_a = pair.get("set_a", {})
    set_b = pair.get("set_b", {})

    # Format outputs for display
    def format_outputs(outputs: List[Dict]) -> str:
        html_parts = []
        for i, output in enumerate(outputs):
            json_str = json.dumps(output, indent=2)
            html_parts.append(f"""
            <div class="output-item">
                <div class="output-label">Output {i+1}</div>
                <pre>{json_str}</pre>
            </div>
            """)
        return "\n".join(html_parts)

    outputs_a_html = format_outputs(set_a.get("outputs", []))
    outputs_b_html = format_outputs(set_b.get("outputs", []))

    metadata = pair.get("metadata", {})
    difficulty = metadata.get("difficulty", "unknown")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Ranking {pair_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ display: flex; gap: 20px; }}
        .set-box {{
            flex: 1;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .set-box h3 {{ margin-top: 0; color: #333; }}
        .output-item {{
            margin-bottom: 15px;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
        }}
        .output-label {{
            background: #f0f0f0;
            padding: 5px 10px;
            font-weight: bold;
            font-size: 12px;
            color: #666;
        }}
        pre {{
            background: #f8f8f8;
            padding: 10px;
            margin: 0;
            overflow-x: auto;
            font-size: 11px;
            max-height: 200px;
            overflow-y: auto;
        }}
        .header {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .task-guide {{
            background: #e8f4f8;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        .task-guide h4 {{ margin-top: 0; }}
        .metadata {{ color: #666; font-size: 14px; }}
        .set-a {{ border-left: 4px solid #4CAF50; }}
        .set-b {{ border-left: 4px solid #2196F3; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>Ranking Pair: {pair_id}</h2>
        <div class="metadata">
            <strong>Difficulty:</strong> {difficulty} |
            <strong>Task:</strong> Which set is more consistent?
        </div>
    </div>

    <div class="container">
        <div class="set-box set-a">
            <h3>Set A ({set_a.get('n_outputs', 0)} outputs)</h3>
            {outputs_a_html}
        </div>
        <div class="set-box set-b">
            <h3>Set B ({set_b.get('n_outputs', 0)} outputs)</h3>
            {outputs_b_html}
        </div>
    </div>

    <div class="task-guide">
        <h4>Annotation Task</h4>
        <p><strong>Question:</strong> Which set of outputs is more consistent?</p>
        <ul>
            <li><strong>A</strong> - Set A is more consistent (outputs are more similar to each other)</li>
            <li><strong>B</strong> - Set B is more consistent</li>
            <li><strong>Equal</strong> - Both sets are equally consistent</li>
        </ul>
        <p><em>Consistency means: Would these outputs work interchangeably? Do they convey the same information?</em></p>
    </div>
</body>
</html>"""
    return html


def create_ranking_index_html(pairs: List[Dict], output_dir: str) -> str:
    """Create index HTML page for ranking annotation."""

    rows = ""
    for i, pair in enumerate(pairs):
        pair_id = pair.get("id", f"rank_{i:04d}")
        metadata = pair.get("metadata", {})
        difficulty = metadata.get("difficulty", "unknown")

        rows += f"""
        <tr>
            <td>{i+1}</td>
            <td><a href="pairs/{pair_id}.html" target="_blank">{pair_id}</a></td>
            <td>{difficulty}</td>
            <td>
                <select id="choice_{pair_id}" onchange="saveChoice('{pair_id}', this.value)">
                    <option value="">-- Select --</option>
                    <option value="A">A is more consistent</option>
                    <option value="B">B is more consistent</option>
                    <option value="equal">Equally consistent</option>
                </select>
            </td>
            <td>
                <select id="confidence_{pair_id}" onchange="saveConfidence('{pair_id}', this.value)">
                    <option value="">-- Confidence --</option>
                    <option value="1">1 - Very uncertain</option>
                    <option value="2">2 - Somewhat uncertain</option>
                    <option value="3">3 - Neutral</option>
                    <option value="4">4 - Somewhat confident</option>
                    <option value="5">5 - Very confident</option>
                </select>
            </td>
            <td><input type="text" id="notes_{pair_id}" placeholder="Optional" onchange="saveNotes('{pair_id}', this.value)"></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Consistency Ranking Annotation</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background: #9C27B0; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        tr:hover {{ background: #ddd; }}
        select {{ padding: 5px; min-width: 150px; }}
        input[type="text"] {{ width: 150px; padding: 5px; }}
        .header {{ margin-bottom: 20px; }}
        .export-btn {{
            background: #2196F3;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-right: 10px;
        }}
        .export-btn:hover {{ background: #1976D2; }}
        .save-btn {{
            background: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-right: 10px;
        }}
        .save-btn:hover {{ background: #388E3C; }}
        .load-btn {{
            background: #FF9800;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-right: 10px;
        }}
        .load-btn:hover {{ background: #F57C00; }}
        .file-input {{
            display: none;
        }}
        .progress {{ margin: 10px 0; font-size: 14px; }}
        .status-msg {{
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            display: none;
        }}
        .status-success {{ background: #c8e6c9; color: #2e7d32; }}
        .status-error {{ background: #ffcdd2; color: #c62828; }}
        .instructions {{
            background: #fff3e0;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Consistency Ranking Annotation</h1>

        <div class="instructions">
            <h3>Task Instructions</h3>
            <p>For each pair, you'll see two sets of outputs (Set A and Set B). Each set contains multiple outputs generated for the same input.</p>
            <p><strong>Your task:</strong> Decide which set shows more <em>consistent</em> outputs.</p>
            <ul>
                <li><strong>Consistent:</strong> Outputs that are similar to each other and could be used interchangeably</li>
                <li><strong>Inconsistent:</strong> Outputs that differ significantly in structure or content</li>
            </ul>
        </div>

        <p class="progress">Progress: <span id="progress">0</span> / {len(pairs)} completed</p>
        <div style="margin-bottom: 15px;">
            <button class="save-btn" onclick="saveProgressToFile()">Save Progress (JSON)</button>
            <button class="load-btn" onclick="document.getElementById('loadFileInput').click()">Load Progress (JSON)</button>
            <input type="file" id="loadFileInput" class="file-input" accept=".json" onchange="loadProgressFromFile(event)">
            <button class="export-btn" onclick="exportAnnotations()">Export Annotations (CSV)</button>
        </div>
        <div id="statusMessage" class="status-msg"></div>
    </div>

    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Pair ID</th>
                <th>Difficulty</th>
                <th>Which is more consistent?</th>
                <th>Confidence</th>
                <th>Notes</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>

    <script>
        window.onload = function() {{
            const saved = localStorage.getItem('ranking_annotations');
            if (saved) {{
                const annotations = JSON.parse(saved);
                for (const [pairId, data] of Object.entries(annotations)) {{
                    const choiceSelect = document.getElementById('choice_' + pairId);
                    const confSelect = document.getElementById('confidence_' + pairId);
                    const notesInput = document.getElementById('notes_' + pairId);
                    if (choiceSelect && data.choice) choiceSelect.value = data.choice;
                    if (confSelect && data.confidence) confSelect.value = data.confidence;
                    if (notesInput && data.notes) notesInput.value = data.notes;
                }}
            }}
            updateProgress();
        }};

        function getAnnotations() {{
            const saved = localStorage.getItem('ranking_annotations');
            return saved ? JSON.parse(saved) : {{}};
        }}

        function saveChoice(pairId, choice) {{
            const annotations = getAnnotations();
            if (!annotations[pairId]) annotations[pairId] = {{}};
            annotations[pairId].choice = choice;
            localStorage.setItem('ranking_annotations', JSON.stringify(annotations));
            updateProgress();
        }}

        function saveConfidence(pairId, confidence) {{
            const annotations = getAnnotations();
            if (!annotations[pairId]) annotations[pairId] = {{}};
            annotations[pairId].confidence = confidence;
            localStorage.setItem('ranking_annotations', JSON.stringify(annotations));
        }}

        function saveNotes(pairId, notes) {{
            const annotations = getAnnotations();
            if (!annotations[pairId]) annotations[pairId] = {{}};
            annotations[pairId].notes = notes;
            localStorage.setItem('ranking_annotations', JSON.stringify(annotations));
        }}

        function updateProgress() {{
            const annotations = getAnnotations();
            const completed = Object.values(annotations).filter(a => a.choice).length;
            document.getElementById('progress').textContent = completed;
        }}

        function exportAnnotations() {{
            const annotations = getAnnotations();
            let csv = 'pair_id,choice,confidence,notes\\n';
            for (const [pairId, data] of Object.entries(annotations)) {{
                const choice = data.choice || '';
                const confidence = data.confidence || '';
                const notes = (data.notes || '').replace(/"/g, '""');
                csv += `${{pairId}},${{choice}},${{confidence}},"${{notes}}"\\n`;
            }}

            const blob = new Blob([csv], {{ type: 'text/csv' }});
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'ranking_annotations_' + new Date().toISOString().split('T')[0] + '.csv';
            a.click();
        }}

        function showStatus(message, isSuccess) {{
            const statusEl = document.getElementById('statusMessage');
            statusEl.textContent = message;
            statusEl.className = 'status-msg ' + (isSuccess ? 'status-success' : 'status-error');
            statusEl.style.display = 'block';
            setTimeout(() => {{ statusEl.style.display = 'none'; }}, 5000);
        }}

        function saveProgressToFile() {{
            const annotations = getAnnotations();
            const data = {{
                version: '1.0',
                type: 'ranking_annotations',
                savedAt: new Date().toISOString(),
                annotations: annotations
            }};

            const blob = new Blob([JSON.stringify(data, null, 2)], {{ type: 'application/json' }});
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'ranking_progress_' + new Date().toISOString().split('T')[0] + '.json';
            a.click();

            const count = Object.values(annotations).filter(a => a.choice).length;
            showStatus('Progress saved! (' + count + ' annotations)', true);
        }}

        function loadProgressFromFile(event) {{
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(e) {{
                try {{
                    const data = JSON.parse(e.target.result);

                    // Validate file format
                    if (!data.annotations || data.type !== 'ranking_annotations') {{
                        showStatus('Invalid file format. Please select a valid progress file.', false);
                        return;
                    }}

                    // Merge with existing annotations (loaded data takes precedence)
                    const existing = getAnnotations();
                    const merged = {{ ...existing, ...data.annotations }};
                    localStorage.setItem('ranking_annotations', JSON.stringify(merged));

                    // Update UI
                    for (const [pairId, pairData] of Object.entries(merged)) {{
                        const choiceSelect = document.getElementById('choice_' + pairId);
                        const confSelect = document.getElementById('confidence_' + pairId);
                        const notesInput = document.getElementById('notes_' + pairId);
                        if (choiceSelect && pairData.choice) choiceSelect.value = pairData.choice;
                        if (confSelect && pairData.confidence) confSelect.value = pairData.confidence;
                        if (notesInput && pairData.notes) notesInput.value = pairData.notes;
                    }}

                    updateProgress();
                    const count = Object.values(merged).filter(a => a.choice).length;
                    showStatus('Progress loaded! (' + count + ' annotations restored)', true);
                }} catch (err) {{
                    showStatus('Error reading file: ' + err.message, false);
                }}
            }};
            reader.readAsText(file);

            // Reset file input so same file can be loaded again
            event.target.value = '';
        }}
    </script>
</body>
</html>"""
    return html


def export_ranking_interface(
    input_file: str,
    output_dir: str,
):
    """Export consistency ranking dataset for annotation."""

    # Load dataset
    with open(input_file, "r") as f:
        data = json.load(f)

    pairs = data.get("pairs", [])
    print(f"Loaded {len(pairs)} ranking pairs from {input_file}")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    pairs_dir = os.path.join(output_dir, "pairs")
    os.makedirs(pairs_dir, exist_ok=True)

    # Create individual HTML pages
    print("Creating individual pair pages...")
    for pair in pairs:
        pair_id = pair.get("id", "unknown")
        html = create_set_viewer_html(pair, pair_id)

        html_path = os.path.join(pairs_dir, f"{pair_id}.html")
        with open(html_path, "w") as f:
            f.write(html)

    # Create index page
    print("Creating index page...")
    index_html = create_ranking_index_html(pairs, output_dir)
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(index_html)

    # Create CSV
    print("Creating CSV file...")
    csv_path = os.path.join(output_dir, "ranking_annotation_sheet.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_id", "difficulty", "n_outputs_a", "n_outputs_b",
            "choice", "confidence", "notes"
        ])

        for pair in pairs:
            writer.writerow([
                pair.get("id", ""),
                pair.get("metadata", {}).get("difficulty", ""),
                pair.get("set_a", {}).get("n_outputs", ""),
                pair.get("set_b", {}).get("n_outputs", ""),
                "",  # choice
                "",  # confidence
                "",  # notes
            ])

    print(f"\nExport complete!")
    print(f"  - Open {output_dir}/index.html in browser for annotation")
    print(f"  - Or use {csv_path} for spreadsheet annotation")


def main():
    parser = argparse.ArgumentParser(
        description="Export consistency ranking dataset for annotation"
    )
    parser.add_argument(
        "--input",
        default="consistency_ranking_dataset.json",
        help="Input dataset JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="ranking_annotation_interface",
        help="Output directory",
    )

    args = parser.parse_args()
    export_ranking_interface(args.input, args.output_dir)


if __name__ == "__main__":
    main()
