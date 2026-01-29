#!/usr/bin/env python
"""
Export human validation dataset to spreadsheet format for easy annotation.

Creates:
1. CSV file with pair IDs and links to JSON viewer
2. JSON files for each pair (viewable in browser)
3. HTML index page for easy navigation
"""

import json
import os
import csv
import argparse
from typing import List, Dict


def create_json_viewer_html(pair: Dict, pair_id: str) -> str:
    """Create HTML page for viewing a single pair."""

    json_a_str = json.dumps(pair["json_a"], indent=2)
    json_b_str = json.dumps(pair["json_b"], indent=2)

    metadata = pair.get("metadata", {})
    sted_score = metadata.get("sted_score", "N/A")
    source = metadata.get("source", "unknown")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Pair {pair_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ display: flex; gap: 20px; }}
        .json-box {{
            flex: 1;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .json-box h3 {{ margin-top: 0; color: #333; }}
        pre {{
            background: #f8f8f8;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 12px;
            max-height: 500px;
            overflow-y: auto;
        }}
        .header {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .rating-guide {{
            background: #e8f4f8;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        .rating-guide h4 {{ margin-top: 0; }}
        .rating-guide ul {{ margin: 0; padding-left: 20px; }}
        .metadata {{ color: #666; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>Pair: {pair_id}</h2>
        <div class="metadata">
            <strong>Source:</strong> {source} |
            <strong>STED Score:</strong> {sted_score:.3f if isinstance(sted_score, float) else sted_score}
        </div>
    </div>

    <div class="container">
        <div class="json-box">
            <h3>JSON A</h3>
            <pre>{json_a_str}</pre>
        </div>
        <div class="json-box">
            <h3>JSON B</h3>
            <pre>{json_b_str}</pre>
        </div>
    </div>

    <div class="rating-guide">
        <h4>Rating Scale</h4>
        <ul>
            <li><strong>5 - Identical/Equivalent:</strong> Fully interchangeable, same information</li>
            <li><strong>4 - Mostly Similar:</strong> Minor differences (field names, formatting)</li>
            <li><strong>3 - Somewhat Similar:</strong> Some overlap, notable differences</li>
            <li><strong>2 - Mostly Different:</strong> Significant structural/content gaps</li>
            <li><strong>1 - Completely Different:</strong> Incompatible outputs</li>
        </ul>
    </div>
</body>
</html>"""
    return html


def create_index_html(pairs: List[Dict], output_dir: str) -> str:
    """Create index HTML page listing all pairs."""

    rows = ""
    for i, pair in enumerate(pairs):
        pair_id = pair.get("id", f"pair_{i:04d}")
        metadata = pair.get("metadata", {})
        sted_score = metadata.get("sted_score", "N/A")
        source = metadata.get("source", "unknown")

        score_str = f"{sted_score:.3f}" if isinstance(sted_score, float) else str(sted_score)

        rows += f"""
        <tr>
            <td>{i+1}</td>
            <td><a href="pairs/{pair_id}.html" target="_blank">{pair_id}</a></td>
            <td>{source}</td>
            <td>{score_str}</td>
            <td><input type="number" min="1" max="5" id="rating_{pair_id}" onchange="saveRating('{pair_id}', this.value)"></td>
            <td><input type="text" id="notes_{pair_id}" placeholder="Optional notes" onchange="saveNotes('{pair_id}', this.value)"></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Human Validation - Annotation Interface</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        tr:hover {{ background: #ddd; }}
        input[type="number"] {{ width: 50px; padding: 5px; }}
        input[type="text"] {{ width: 200px; padding: 5px; }}
        .header {{ margin-bottom: 20px; }}
        .export-btn {{
            background: #2196F3;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        .export-btn:hover {{ background: #1976D2; }}
        .progress {{ margin: 10px 0; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Human Validation Annotation</h1>
        <p>Click pair ID to view JSON comparison. Enter rating (1-5) for each pair.</p>
        <p class="progress">Progress: <span id="progress">0</span> / {len(pairs)} completed</p>
        <button class="export-btn" onclick="exportAnnotations()">Export Annotations (CSV)</button>
    </div>

    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Pair ID</th>
                <th>Source</th>
                <th>STED Score</th>
                <th>Your Rating (1-5)</th>
                <th>Notes</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>

    <script>
        // Load saved annotations from localStorage
        window.onload = function() {{
            const saved = localStorage.getItem('annotations');
            if (saved) {{
                const annotations = JSON.parse(saved);
                for (const [pairId, data] of Object.entries(annotations)) {{
                    const ratingInput = document.getElementById('rating_' + pairId);
                    const notesInput = document.getElementById('notes_' + pairId);
                    if (ratingInput && data.rating) ratingInput.value = data.rating;
                    if (notesInput && data.notes) notesInput.value = data.notes;
                }}
            }}
            updateProgress();
        }};

        function getAnnotations() {{
            const saved = localStorage.getItem('annotations');
            return saved ? JSON.parse(saved) : {{}};
        }}

        function saveRating(pairId, rating) {{
            const annotations = getAnnotations();
            if (!annotations[pairId]) annotations[pairId] = {{}};
            annotations[pairId].rating = rating;
            localStorage.setItem('annotations', JSON.stringify(annotations));
            updateProgress();
        }}

        function saveNotes(pairId, notes) {{
            const annotations = getAnnotations();
            if (!annotations[pairId]) annotations[pairId] = {{}};
            annotations[pairId].notes = notes;
            localStorage.setItem('annotations', JSON.stringify(annotations));
        }}

        function updateProgress() {{
            const annotations = getAnnotations();
            const completed = Object.values(annotations).filter(a => a.rating).length;
            document.getElementById('progress').textContent = completed;
        }}

        function exportAnnotations() {{
            const annotations = getAnnotations();
            let csv = 'pair_id,rating,notes\\n';
            for (const [pairId, data] of Object.entries(annotations)) {{
                const rating = data.rating || '';
                const notes = (data.notes || '').replace(/"/g, '""');
                csv += `${{pairId}},${{rating}},"${{notes}}"\\n`;
            }}

            const blob = new Blob([csv], {{ type: 'text/csv' }});
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'annotations_' + new Date().toISOString().split('T')[0] + '.csv';
            a.click();
        }}
    </script>
</body>
</html>"""
    return html


def export_for_annotation(
    input_file: str,
    output_dir: str,
    create_csv: bool = True,
):
    """Export dataset for spreadsheet-based annotation."""

    # Load dataset
    with open(input_file, "r") as f:
        data = json.load(f)

    pairs = data.get("pairs", [])
    print(f"Loaded {len(pairs)} pairs from {input_file}")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    pairs_dir = os.path.join(output_dir, "pairs")
    os.makedirs(pairs_dir, exist_ok=True)

    # Create individual HTML pages for each pair
    print("Creating individual pair pages...")
    for i, pair in enumerate(pairs):
        pair_id = pair.get("id", f"pair_{i:04d}")
        html = create_json_viewer_html(pair, pair_id)

        html_path = os.path.join(pairs_dir, f"{pair_id}.html")
        with open(html_path, "w") as f:
            f.write(html)

    # Create index page
    print("Creating index page...")
    index_html = create_index_html(pairs, output_dir)
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(index_html)

    # Create CSV for offline annotation
    if create_csv:
        print("Creating CSV file...")
        csv_path = os.path.join(output_dir, "annotation_sheet.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "pair_id", "source", "sted_score", "deepdiff_score",
                "ted_score", "bertscore", "rating", "notes"
            ])

            for pair in pairs:
                pair_id = pair.get("id", "unknown")
                metadata = pair.get("metadata", {})
                scores = metadata.get("all_scores", {})

                writer.writerow([
                    pair_id,
                    metadata.get("source", ""),
                    f"{metadata.get('sted_score', ''):.3f}" if metadata.get('sted_score') else "",
                    f"{scores.get('deepdiff', ''):.3f}" if scores.get('deepdiff') else "",
                    f"{scores.get('ted', ''):.3f}" if scores.get('ted') else "",
                    f"{scores.get('bertscore', ''):.3f}" if scores.get('bertscore') else "",
                    "",  # rating - to be filled
                    "",  # notes - to be filled
                ])

    print(f"\nExport complete!")
    print(f"  - Open {output_dir}/index.html in browser for annotation")
    print(f"  - Or use {output_dir}/annotation_sheet.csv for spreadsheet annotation")
    print(f"  - Individual pairs viewable in {pairs_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Export human validation dataset for annotation"
    )
    parser.add_argument(
        "--input",
        default="human_validation_dataset.json",
        help="Input dataset JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="annotation_interface",
        help="Output directory for annotation files",
    )

    args = parser.parse_args()
    export_for_annotation(args.input, args.output_dir)


if __name__ == "__main__":
    main()
