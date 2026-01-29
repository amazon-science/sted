#!/usr/bin/env python
"""
Export best-match selection dataset to HTML annotation interface.

Creates:
1. HTML index page with all samples
2. Individual HTML pages for each sample
3. CSV export functionality
"""

import json
import os
import csv
import argparse
from typing import List, Dict


def create_sample_html(item: Dict, item_idx: int, total_items: int) -> str:
    """Create HTML page for a single best-match selection sample."""

    sample_id = item.get("sample_id", item.get("id", f"sample_{item_idx}"))
    gt = item.get("ground_truth", {})
    candidates = item.get("candidates", [])

    gt_str = json.dumps(gt, indent=2)

    # Build candidate options HTML
    candidates_html = ""
    for cand in candidates:
        label = cand.get("label", "?")
        response = cand.get("response", {})
        model = cand.get("model", "unknown")
        response_str = json.dumps(response, indent=2)

        candidates_html += f"""
        <div class="candidate" onclick="selectCandidate('{label}')">
            <div class="candidate-header">
                <input type="radio" name="choice" id="choice_{label}" value="{label}" onchange="saveChoice('{label}')">
                <label for="choice_{label}" class="candidate-label">[{label}]</label>
                <span class="model-name">({model})</span>
            </div>
            <pre class="response-json">{response_str}</pre>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Sample {item_idx + 1} / {total_items}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            line-height: 1.5;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}

        .header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header h2 {{ margin: 0 0 10px 0; color: #333; }}
        .progress {{ color: #666; }}
        .nav-buttons {{ margin-top: 15px; }}
        .nav-buttons a {{
            display: inline-block;
            padding: 8px 16px;
            margin-right: 10px;
            background: #007bff;
            color: white;
            text-decoration: none;
            border-radius: 4px;
        }}
        .nav-buttons a:hover {{ background: #0056b3; }}
        .nav-buttons a.disabled {{
            background: #ccc;
            pointer-events: none;
        }}

        .task-description {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #2196f3;
        }}
        .task-description h3 {{ margin: 0 0 10px 0; color: #1565c0; }}

        .ground-truth {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #4caf50;
        }}
        .ground-truth h3 {{ margin: 0 0 15px 0; color: #2e7d32; }}

        .candidates-section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .candidates-section h3 {{ margin: 0 0 15px 0; color: #333; }}

        .candidate {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            cursor: pointer;
            border: 2px solid transparent;
            transition: all 0.2s;
        }}
        .candidate:hover {{ background: #f0f0f0; border-color: #ddd; }}
        .candidate.selected {{
            background: #e8f5e9;
            border-color: #4caf50;
        }}
        .candidate-header {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }}
        .candidate-label {{
            font-size: 18px;
            font-weight: bold;
            margin-left: 10px;
            color: #333;
        }}
        .model-name {{
            margin-left: 10px;
            color: #666;
            font-size: 14px;
        }}

        pre {{
            background: #263238;
            color: #aed581;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 13px;
            margin: 0;
        }}
        .ground-truth pre {{
            background: #1b5e20;
            color: #c5e1a5;
        }}
        .response-json {{ background: #37474f; }}

        .confidence-section {{
            margin-top: 20px;
            padding: 15px;
            background: #fff3e0;
            border-radius: 8px;
        }}
        .confidence-section h4 {{ margin: 0 0 10px 0; }}
        .confidence-buttons {{
            display: flex;
            gap: 10px;
        }}
        .confidence-btn {{
            padding: 8px 16px;
            border: 1px solid #ccc;
            background: white;
            border-radius: 4px;
            cursor: pointer;
        }}
        .confidence-btn:hover {{ background: #f5f5f5; }}
        .confidence-btn.selected {{
            background: #ff9800;
            color: white;
            border-color: #ff9800;
        }}

        .notes-section {{
            margin-top: 15px;
        }}
        .notes-section textarea {{
            width: 100%;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            min-height: 60px;
            font-family: inherit;
        }}

        .save-status {{
            margin-top: 15px;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
        }}
        .save-status.saved {{ background: #e8f5e9; color: #2e7d32; }}
        .save-status.unsaved {{ background: #ffebee; color: #c62828; }}

        .confirm-section {{
            margin-top: 20px;
            text-align: center;
        }}
        .confirm-btn {{
            padding: 15px 40px;
            font-size: 18px;
            font-weight: bold;
            background: #4caf50;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .confirm-btn:hover {{ background: #388e3c; transform: scale(1.02); }}
        .confirm-btn:disabled {{
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }}
        .confirm-btn.confirmed {{
            background: #2196f3;
        }}
        .skip-btn {{
            padding: 15px 40px;
            font-size: 18px;
            font-weight: bold;
            background: #9e9e9e;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            margin-left: 20px;
        }}
        .skip-btn:hover {{ background: #757575; }}
        .skip-btn.skipped {{
            background: #ff9800;
        }}
        .skip-reason {{
            margin-top: 15px;
            display: none;
        }}
        .skip-reason.visible {{
            display: block;
        }}
        .skip-reason select {{
            padding: 8px 12px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 14px;
            width: 100%;
            max-width: 400px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>Sample {item_idx + 1} of {total_items}</h2>
            <div class="progress">ID: {sample_id}</div>
            <div class="nav-buttons">
                {"<a href='rank_" + f"{item_idx-1:04d}" + ".html'>Previous</a>" if item_idx > 0 else "<a class='disabled'>Previous</a>"}
                <a href="../index.html">Back to List</a>
                {"<a href='rank_" + f"{item_idx+1:04d}" + ".html' onclick='confirmBeforeNav(event)'>Next</a>" if item_idx < total_items - 1 else "<a class='disabled'>Next</a>"}
            </div>
        </div>

        <div class="task-description">
            <h3>Task: Select the Most Similar Response</h3>
            <p>Compare the responses below to the Ground Truth and select the one that is <strong>MOST SIMILAR</strong>.</p>
            <p>Consider: same tool names, same parameters, same structure, functional equivalence.</p>
        </div>

        <div class="ground-truth">
            <h3>Ground Truth (Reference)</h3>
            <pre>{gt_str}</pre>
        </div>

        <div class="candidates-section">
            <h3>Candidate Responses - Select the Most Similar:</h3>
            {candidates_html}

            <div class="confidence-section">
                <h4>How confident are you in your choice?</h4>
                <div class="confidence-buttons">
                    <button class="confidence-btn" onclick="saveConfidence(1)">1 - Very Unsure</button>
                    <button class="confidence-btn" onclick="saveConfidence(2)">2 - Unsure</button>
                    <button class="confidence-btn" onclick="saveConfidence(3)">3 - Neutral</button>
                    <button class="confidence-btn" onclick="saveConfidence(4)">4 - Confident</button>
                    <button class="confidence-btn" onclick="saveConfidence(5)">5 - Very Confident</button>
                </div>
            </div>

            <div class="notes-section">
                <h4>Notes (optional)</h4>
                <textarea id="notes" placeholder="Any additional comments..." onchange="saveNotes()"></textarea>
            </div>

            <div id="save-status" class="save-status unsaved">Not saved yet</div>

            <div class="skip-reason" id="skip-reason-section">
                <h4>Reason for skipping:</h4>
                <select id="skip-reason" onchange="saveSkipReason()">
                    <option value="">-- Select a reason --</option>
                    <option value="all_equally_wrong">All options are equally wrong/incorrect</option>
                    <option value="all_equally_similar">All options are equally similar (can't decide)</option>
                    <option value="ground_truth_unclear">Ground truth is unclear or ambiguous</option>
                    <option value="too_complex">Too complex to evaluate</option>
                    <option value="other">Other (add note below)</option>
                </select>
            </div>

            <div class="confirm-section">
                <button class="confirm-btn" id="confirm-btn" onclick="confirmAndNext()">Confirm & Next</button>
                <button class="skip-btn" id="skip-btn" onclick="toggleSkip()">Skip</button>
            </div>
        </div>
    </div>

    <script>
        const sampleId = "{sample_id}";
        const itemId = "rank_{item_idx:04d}";

        function getAnnotations() {{
            const saved = localStorage.getItem('best_match_annotations');
            return saved ? JSON.parse(saved) : {{}};
        }}

        function saveAnnotations(annotations) {{
            localStorage.setItem('best_match_annotations', JSON.stringify(annotations));
        }}

        function selectCandidate(label) {{
            document.getElementById('choice_' + label).checked = true;
            saveChoice(label);
        }}

        function saveChoice(label) {{
            const annotations = getAnnotations();
            if (!annotations[itemId]) annotations[itemId] = {{}};
            annotations[itemId].choice = label;
            annotations[itemId].timestamp = new Date().toISOString();
            saveAnnotations(annotations);
            updateUI();
        }}

        function saveConfidence(level) {{
            const annotations = getAnnotations();
            if (!annotations[itemId]) annotations[itemId] = {{}};
            annotations[itemId].confidence = level;
            saveAnnotations(annotations);
            updateUI();
        }}

        function saveNotes() {{
            const notes = document.getElementById('notes').value;
            const annotations = getAnnotations();
            if (!annotations[itemId]) annotations[itemId] = {{}};
            annotations[itemId].notes = notes;
            saveAnnotations(annotations);
            updateUI();
        }}

        function toggleSkip() {{
            const annotations = getAnnotations();
            if (!annotations[itemId]) annotations[itemId] = {{}};

            if (annotations[itemId].skipped) {{
                // Un-skip
                annotations[itemId].skipped = false;
                annotations[itemId].skipReason = null;
            }} else {{
                // Skip - clear any choice
                annotations[itemId].skipped = true;
                annotations[itemId].choice = null;
                annotations[itemId].confirmed = false;
            }}
            annotations[itemId].timestamp = new Date().toISOString();
            saveAnnotations(annotations);
            updateUI();
        }}

        function saveSkipReason() {{
            const reason = document.getElementById('skip-reason').value;
            const annotations = getAnnotations();
            if (!annotations[itemId]) annotations[itemId] = {{}};
            annotations[itemId].skipReason = reason;
            saveAnnotations(annotations);
            updateUI();
        }}

        function skipAndNext() {{
            const annotations = getAnnotations();
            const ann = annotations[itemId] || {{}};

            if (!ann.skipReason) {{
                alert('Please select a reason for skipping.');
                return;
            }}

            // Mark as confirmed skip
            ann.skipped = true;
            ann.confirmed = true;
            ann.confirmedAt = new Date().toISOString();
            annotations[itemId] = ann;
            saveAnnotations(annotations);

            // Navigate to next
            setTimeout(() => {{
                {"window.location.href = 'rank_" + f"{item_idx+1:04d}" + ".html';" if item_idx < total_items - 1 else "window.location.href = '../index.html';"}
            }}, 300);
        }}

        function updateUI() {{
            const annotations = getAnnotations();
            const ann = annotations[itemId] || {{}};

            // Update skip state
            const skipBtn = document.getElementById('skip-btn');
            const skipReasonSection = document.getElementById('skip-reason-section');
            const confirmBtn = document.getElementById('confirm-btn');

            if (ann.skipped) {{
                skipBtn.textContent = 'Undo Skip';
                skipBtn.classList.add('skipped');
                skipReasonSection.classList.add('visible');
                confirmBtn.textContent = 'Confirm Skip & Next';
                confirmBtn.onclick = skipAndNext;

                // Disable candidate selection when skipped
                document.querySelectorAll('.candidate').forEach(el => {{
                    el.style.opacity = '0.5';
                    el.style.pointerEvents = 'none';
                }});
            }} else {{
                skipBtn.textContent = 'Skip';
                skipBtn.classList.remove('skipped');
                skipReasonSection.classList.remove('visible');
                confirmBtn.textContent = 'Confirm & Next';
                confirmBtn.onclick = confirmAndNext;

                // Enable candidate selection
                document.querySelectorAll('.candidate').forEach(el => {{
                    el.style.opacity = '1';
                    el.style.pointerEvents = 'auto';
                }});
            }}

            // Update skip reason dropdown
            if (ann.skipReason) {{
                document.getElementById('skip-reason').value = ann.skipReason;
            }}

            // Update candidate selection
            document.querySelectorAll('.candidate').forEach(el => el.classList.remove('selected'));
            if (ann.choice) {{
                const radio = document.getElementById('choice_' + ann.choice);
                if (radio) {{
                    radio.checked = true;
                    radio.closest('.candidate').classList.add('selected');
                }}
            }}

            // Update confidence buttons
            document.querySelectorAll('.confidence-btn').forEach((btn, idx) => {{
                btn.classList.toggle('selected', ann.confidence === idx + 1);
            }});

            // Update notes
            if (ann.notes) {{
                document.getElementById('notes').value = ann.notes;
            }}

            // Update save status
            const status = document.getElementById('save-status');
            if (ann.skipped) {{
                status.textContent = 'Marked as SKIP' + (ann.skipReason ? ' - ' + ann.skipReason : '');
                status.className = 'save-status saved';
            }} else if (ann.choice) {{
                status.textContent = 'Saved: Choice ' + ann.choice + (ann.confidence ? ', Confidence ' + ann.confidence : '');
                status.className = 'save-status saved';
            }} else {{
                status.textContent = 'Not saved yet - please select a response or skip';
                status.className = 'save-status unsaved';
            }}
        }}

        function confirmBeforeNav(event) {{
            const annotations = getAnnotations();
            const ann = annotations[itemId] || {{}};

            // If a choice is made, confirm it before navigating
            if (ann.choice && !ann.confirmed) {{
                ann.confirmed = true;
                ann.confirmedAt = new Date().toISOString();
                annotations[itemId] = ann;
                saveAnnotations(annotations);
            }}
            // Allow navigation to proceed
        }}

        function confirmAndNext() {{
            const annotations = getAnnotations();
            const ann = annotations[itemId] || {{}};

            if (!ann.choice) {{
                alert('Please select a response before confirming.');
                return;
            }}

            // Mark as confirmed
            ann.confirmed = true;
            ann.confirmedAt = new Date().toISOString();
            annotations[itemId] = ann;
            saveAnnotations(annotations);

            // Update button state
            const btn = document.getElementById('confirm-btn');
            btn.textContent = 'Confirmed!';
            btn.classList.add('confirmed');

            // Navigate to next after brief delay
            setTimeout(() => {{
                {"window.location.href = 'rank_" + f"{item_idx+1:04d}" + ".html';" if item_idx < total_items - 1 else "window.location.href = '../index.html';"}
            }}, 500);
        }}

        // Load saved state on page load
        window.onload = function() {{
            updateUI();
            // Update confirm button if already confirmed
            const annotations = getAnnotations();
            const ann = annotations[itemId] || {{}};
            if (ann.confirmed) {{
                const btn = document.getElementById('confirm-btn');
                btn.textContent = 'Confirmed';
                btn.classList.add('confirmed');
            }}
        }};
    </script>
</body>
</html>"""
    return html


def create_index_html(items: List[Dict], output_dir: str, method_mapping: Dict = None, dataset_info: Dict = None) -> str:
    """Create index HTML page listing all samples with dataset filtering."""

    # Convert method mapping to JSON for injection
    method_mapping_json = json.dumps(method_mapping or {})

    # Build dataset info for JavaScript
    dataset_info = dataset_info or {}
    dataset_info_json = json.dumps(dataset_info)

    # Group items by dataset
    datasets = {}
    for i, item in enumerate(items):
        dataset = item.get("metadata", {}).get("dataset", "unknown")
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append((i, item))

    datasets_list = list(datasets.keys())
    datasets_list_json = json.dumps(datasets_list)

    # Build dataset tabs HTML (avoiding backslash in f-string)
    dataset_tabs_html = ""
    for ds in datasets_list:
        sq = "'"  # single quote
        dataset_tabs_html += f'<div class="dataset-tab" onclick="filterDataset({sq}{ds}{sq})" id="tab-{ds}">{ds.upper()} <span class="count">({len(datasets.get(ds, []))})</span></div>'

    # Build per-dataset progress bars HTML
    progress_items_html = ""
    for ds in datasets_list:
        progress_items_html += f'''<div class="progress-item">
                    <h4>{ds.upper()}</h4>
                    <div class="progress-bar">
                        <div class="progress-fill {ds}" id="progress-fill-{ds}"></div>
                    </div>
                    <div class="progress-text" id="progress-text-{ds}">0 / {len(datasets.get(ds, []))} completed</div>
                </div>'''

    rows = ""
    for i, item in enumerate(items):
        item_id = item.get("id", f"rank_{i:04d}")
        sample_id = item.get("sample_id", "")
        n_candidates = len(item.get("candidates", []))
        dataset = item.get("metadata", {}).get("dataset", "unknown")

        rows += f"""
        <tr id="row_{item_id}" data-dataset="{dataset}">
            <td>{i + 1}</td>
            <td><a href="samples/{item_id}.html" target="_blank">{item_id}</a></td>
            <td>{sample_id[:20]}...</td>
            <td>{dataset}</td>
            <td>{n_candidates}</td>
            <td class="choice-cell" id="choice_{item_id}">-</td>
            <td class="confidence-cell" id="conf_{item_id}">-</td>
            <td class="status-cell" id="status_{item_id}">Pending</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Best-Match Selection - Human Validation</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}

        .header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header h1 {{ margin: 0 0 10px 0; color: #333; }}
        .header p {{ color: #666; margin: 5px 0; }}

        .dataset-tabs {{
            display: flex;
            gap: 10px;
            margin: 15px 0;
            flex-wrap: wrap;
        }}
        .dataset-tab {{
            padding: 10px 20px;
            border: 2px solid #ddd;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
            background: white;
        }}
        .dataset-tab:hover {{ border-color: #2196f3; }}
        .dataset-tab.active {{
            background: #2196f3;
            color: white;
            border-color: #2196f3;
        }}
        .dataset-tab .count {{
            font-size: 12px;
            opacity: 0.8;
            margin-left: 5px;
        }}

        .progress-container {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .progress-item {{
            flex: 1;
            min-width: 200px;
            background: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
        }}
        .progress-item h4 {{ margin: 0 0 10px 0; }}
        .progress-bar {{
            background: #e0e0e0;
            border-radius: 10px;
            height: 20px;
            margin: 10px 0;
            overflow: hidden;
        }}
        .progress-fill {{
            background: linear-gradient(90deg, #4caf50, #8bc34a);
            height: 100%;
            width: 0%;
            transition: width 0.3s;
        }}
        .progress-fill.toucan {{ background: linear-gradient(90deg, #ff9800, #ffb74d); }}
        .progress-fill.sharegpt {{ background: linear-gradient(90deg, #9c27b0, #ba68c8); }}
        .progress-text {{ font-weight: bold; font-size: 14px; }}

        .actions {{
            margin: 20px 0;
        }}
        .btn {{
            display: inline-block;
            padding: 10px 20px;
            margin-right: 10px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            text-decoration: none;
        }}
        .btn-primary {{ background: #2196f3; color: white; }}
        .btn-primary:hover {{ background: #1976d2; }}
        .btn-success {{ background: #4caf50; color: white; }}
        .btn-success:hover {{ background: #388e3c; }}
        .btn-danger {{ background: #f44336; color: white; }}
        .btn-danger:hover {{ background: #d32f2f; }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background: #37474f;
            color: white;
            font-weight: 500;
        }}
        tr:hover {{ background: #f5f5f5; }}

        .status-pending {{ color: #ff9800; }}
        .status-selected {{ color: #2196f3; }}
        .status-confirmed {{ color: #4caf50; font-weight: bold; }}
        .status-skipped {{ color: #9e9e9e; font-style: italic; }}

        .btn-warning {{ background: #ff9800; color: white; }}
        .btn-warning:hover {{ background: #f57c00; }}
        .btn-save {{ background: #00897b; color: white; }}
        .btn-save:hover {{ background: #00695c; }}
        .btn-load {{ background: #7b1fa2; color: white; }}
        .btn-load:hover {{ background: #6a1b9a; }}
        .file-input {{ display: none; }}
        .status-msg {{
            display: inline-block;
            padding: 8px 15px;
            border-radius: 4px;
            margin-left: 10px;
            font-size: 14px;
            opacity: 0;
            transition: opacity 0.3s;
        }}
        .status-msg.visible {{ opacity: 1; }}
        .status-msg.success {{ background: #c8e6c9; color: #2e7d32; }}
        .status-msg.error {{ background: #ffcdd2; color: #c62828; }}

        .results-panel {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #ff9800;
        }}
        .results-panel h3 {{ margin: 0 0 15px 0; color: #e65100; }}
        .results-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        .results-table th, .results-table td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        .results-table th {{ background: #fff3e0; }}
        .win-bar {{
            background: #e0e0e0;
            border-radius: 4px;
            height: 20px;
            overflow: hidden;
        }}
        .win-bar-fill {{
            background: linear-gradient(90deg, #4caf50, #8bc34a);
            height: 100%;
        }}

        .results-tabs {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .results-section {{
            flex: 1;
            min-width: 300px;
            padding: 15px;
            background: #fafafa;
            border-radius: 8px;
            margin-bottom: 15px;
        }}
        .results-section h4 {{
            margin: 0 0 10px 0;
            color: #37474f;
            border-bottom: 2px solid #ff9800;
            padding-bottom: 5px;
        }}

        .instructions {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #2196f3;
        }}
        .instructions h3 {{ margin: 0 0 10px 0; color: #1565c0; }}
        .instructions ol {{ margin: 0; padding-left: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Best-Match Selection - Human Validation Study</h1>
            <p>Select the response most similar to the ground truth for each sample.</p>

            <div class="dataset-tabs">
                <div class="dataset-tab active" onclick="filterDataset('all')" id="tab-all">
                    All <span class="count">({len(items)})</span>
                </div>
                {dataset_tabs_html}
            </div>

            <div class="progress-container">
                <div class="progress-item">
                    <h4>Overall Progress</h4>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress-fill-all"></div>
                    </div>
                    <div class="progress-text" id="progress-text-all">0 / {len(items)} completed</div>
                </div>
                {progress_items_html}
            </div>
        </div>

        <div class="instructions">
            <h3>Instructions</h3>
            <ol>
                <li>Select a dataset tab to focus on Toucan or ShareGPT samples</li>
                <li>Click on a sample ID to open the annotation page</li>
                <li>Compare each candidate response to the Ground Truth</li>
                <li>Select the response that is MOST SIMILAR</li>
                <li>Optionally rate your confidence (1-5)</li>
                <li>Your progress is saved automatically</li>
            </ol>
        </div>

        <div class="actions">
            <button class="btn btn-save" onclick="saveProgressToFile()">Save Progress (JSON)</button>
            <button class="btn btn-load" onclick="document.getElementById('loadFileInput').click()">Load Progress (JSON/CSV)</button>
            <input type="file" id="loadFileInput" class="file-input" accept=".json,.csv" onchange="loadProgressFromFile(event)">
            <button class="btn btn-success" onclick="exportAnnotations()">Export Annotations (CSV)</button>
            <button class="btn btn-primary" onclick="exportJSON()">Export as JSON</button>
            <button class="btn btn-warning" onclick="showResults()">Show Results</button>
            <button class="btn btn-danger" onclick="clearAnnotations()">Clear All</button>
            <span id="statusMessage" class="status-msg"></span>
        </div>

        <div id="results-panel" class="results-panel" style="display: none;">
            <h3>Human Validation Results</h3>
            <div id="results-content"></div>
        </div>

        <table id="samples-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Sample</th>
                    <th>ID</th>
                    <th>Dataset</th>
                    <th>Options</th>
                    <th>Your Choice</th>
                    <th>Confidence</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>

    <script>
        // Method mapping data (which method picked which label for each item)
        const methodMapping = {method_mapping_json};
        const datasetsList = {datasets_list_json};
        const datasetInfo = {dataset_info_json};
        let currentFilter = 'all';

        function getAnnotations() {{
            const saved = localStorage.getItem('best_match_annotations');
            return saved ? JSON.parse(saved) : {{}};
        }}

        function filterDataset(dataset) {{
            currentFilter = dataset;

            // Update tab styles
            document.querySelectorAll('.dataset-tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.getElementById('tab-' + dataset).classList.add('active');

            // Filter table rows
            document.querySelectorAll('tbody tr').forEach(row => {{
                const rowDataset = row.getAttribute('data-dataset');
                if (dataset === 'all' || rowDataset === dataset) {{
                    row.style.display = '';
                }} else {{
                    row.style.display = 'none';
                }}
            }});
        }}

        function updateProgress() {{
            const annotations = getAnnotations();

            // Track per-dataset stats
            const stats = {{ all: {{ completed: 0, skipped: 0, total: 0 }} }};
            datasetsList.forEach(ds => {{
                stats[ds] = {{ completed: 0, skipped: 0, total: 0 }};
            }});

            document.querySelectorAll('tbody tr').forEach(row => {{
                const itemId = row.id.replace('row_', '');
                const dataset = row.getAttribute('data-dataset');
                const ann = annotations[itemId];

                stats.all.total++;
                if (stats[dataset]) stats[dataset].total++;

                const choiceCell = document.getElementById('choice_' + itemId);
                const confCell = document.getElementById('conf_' + itemId);
                const statusCell = document.getElementById('status_' + itemId);

                if (ann && ann.skipped && ann.confirmed) {{
                    choiceCell.textContent = 'SKIP';
                    confCell.textContent = ann.skipReason || '-';
                    statusCell.textContent = 'Skipped';
                    statusCell.className = 'status-cell status-skipped';
                    stats.all.completed++;
                    stats.all.skipped++;
                    if (stats[dataset]) {{
                        stats[dataset].completed++;
                        stats[dataset].skipped++;
                    }}
                }} else if (ann && ann.choice) {{
                    choiceCell.textContent = ann.choice;
                    confCell.textContent = ann.confidence || '-';
                    if (ann.confirmed) {{
                        statusCell.textContent = 'Confirmed';
                        statusCell.className = 'status-cell status-confirmed';
                        stats.all.completed++;
                        if (stats[dataset]) stats[dataset].completed++;
                    }} else {{
                        statusCell.textContent = 'Selected';
                        statusCell.className = 'status-cell status-selected';
                    }}
                }} else {{
                    choiceCell.textContent = '-';
                    confCell.textContent = '-';
                    statusCell.textContent = 'Pending';
                    statusCell.className = 'status-cell status-pending';
                }}
            }});

            // Update progress bars
            const updateProgressBar = (key) => {{
                const s = stats[key];
                const skipText = s.skipped > 0 ? ` (${{s.skipped}} skipped)` : '';
                const textEl = document.getElementById('progress-text-' + key);
                const fillEl = document.getElementById('progress-fill-' + key);
                if (textEl) textEl.textContent = `${{s.completed}}${{skipText}} / ${{s.total}} completed`;
                if (fillEl) fillEl.style.width = s.total > 0 ? (s.completed / s.total * 100) + '%' : '0%';
            }};

            updateProgressBar('all');
            datasetsList.forEach(ds => updateProgressBar(ds));
        }}

        function exportAnnotations() {{
            const annotations = getAnnotations();
            let csv = 'item_id,choice,confidence,skipped,skip_reason,notes,timestamp\\n';

            for (const [itemId, data] of Object.entries(annotations)) {{
                const choice = data.skipped ? 'SKIP' : (data.choice || '');
                const confidence = data.confidence || '';
                const skipped = data.skipped ? 'true' : 'false';
                const skipReason = data.skipReason || '';
                const notes = (data.notes || '').replace(/"/g, '""').replace(/\\n/g, ' ');
                const timestamp = data.timestamp || '';
                csv += `${{itemId}},${{choice}},${{confidence}},${{skipped}},${{skipReason}},"${{notes}}",${{timestamp}}\\n`;
            }}

            downloadFile(csv, 'best_match_annotations.csv', 'text/csv');
        }}

        function exportJSON() {{
            const annotations = getAnnotations();
            const json = JSON.stringify(annotations, null, 2);
            downloadFile(json, 'best_match_annotations.json', 'application/json');
        }}

        function downloadFile(content, filename, type) {{
            const blob = new Blob([content], {{ type: type }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
        }}

        function clearAnnotations() {{
            if (confirm('Are you sure you want to clear all annotations? This cannot be undone.')) {{
                localStorage.removeItem('best_match_annotations');
                updateProgress();
            }}
        }}

        function showStatus(message, isSuccess) {{
            const statusEl = document.getElementById('statusMessage');
            statusEl.textContent = message;
            statusEl.className = 'status-msg visible ' + (isSuccess ? 'success' : 'error');
            setTimeout(() => {{ statusEl.className = 'status-msg'; }}, 5000);
        }}

        function saveProgressToFile() {{
            const annotations = getAnnotations();
            const data = {{
                version: '1.0',
                type: 'best_match_annotations',
                savedAt: new Date().toISOString(),
                annotations: annotations
            }};

            const blob = new Blob([JSON.stringify(data, null, 2)], {{ type: 'application/json' }});
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'best_match_progress_' + new Date().toISOString().split('T')[0] + '.json';
            a.click();
            window.URL.revokeObjectURL(url);

            const count = Object.values(annotations).filter(a => a.choice || a.skipped).length;
            showStatus('Progress saved! (' + count + ' annotations)', true);
        }}

        function parseCSV(text) {{
            // Simple CSV parser that handles quoted fields
            const lines = text.trim().split('\\n');
            if (lines.length < 2) return {{}};

            const headers = lines[0].split(',').map(h => h.trim());
            const annotations = {{}};

            for (let i = 1; i < lines.length; i++) {{
                const line = lines[i];
                if (!line.trim()) continue;

                // Parse CSV line handling quoted fields
                const values = [];
                let current = '';
                let inQuotes = false;
                for (let j = 0; j < line.length; j++) {{
                    const char = line[j];
                    if (char === '"') {{
                        inQuotes = !inQuotes;
                    }} else if (char === ',' && !inQuotes) {{
                        values.push(current.trim());
                        current = '';
                    }} else {{
                        current += char;
                    }}
                }}
                values.push(current.trim());

                // Map to object using headers
                const row = {{}};
                headers.forEach((h, idx) => {{
                    row[h] = values[idx] || '';
                }});

                // Convert to annotation format
                const itemId = row['item_id'] || row['itemId'];
                if (!itemId) continue;

                const ann = {{}};
                if (row['choice'] && row['choice'] !== 'SKIP' && row['choice'] !== '-') {{
                    ann.choice = row['choice'];
                    ann.confirmed = true;
                }}
                if (row['confidence'] && row['confidence'] !== '-') {{
                    ann.confidence = parseInt(row['confidence']) || null;
                }}
                if (row['skipped'] === 'true' || row['choice'] === 'SKIP') {{
                    ann.skipped = true;
                    ann.confirmed = true;
                    ann.skipReason = row['skip_reason'] || row['skipReason'] || null;
                }}
                if (row['notes']) {{
                    ann.notes = row['notes'].replace(/""/g, '"');
                }}
                if (row['timestamp']) {{
                    ann.timestamp = row['timestamp'];
                }}

                if (Object.keys(ann).length > 0) {{
                    annotations[itemId] = ann;
                }}
            }}
            return annotations;
        }}

        function loadProgressFromFile(event) {{
            const file = event.target.files[0];
            if (!file) return;

            const isCSV = file.name.toLowerCase().endsWith('.csv');

            const reader = new FileReader();
            reader.onload = function(e) {{
                try {{
                    let loadedAnnotations = {{}};

                    if (isCSV) {{
                        // Parse CSV file
                        loadedAnnotations = parseCSV(e.target.result);
                        if (Object.keys(loadedAnnotations).length === 0) {{
                            showStatus('No valid annotations found in CSV file.', false);
                            return;
                        }}
                    }} else {{
                        // Parse JSON file
                        const data = JSON.parse(e.target.result);

                        // Validate file format
                        if (!data.annotations || data.type !== 'best_match_annotations') {{
                            showStatus('Invalid JSON format. Please select a valid progress file.', false);
                            return;
                        }}
                        loadedAnnotations = data.annotations;
                    }}

                    // Merge with existing annotations (loaded data takes precedence)
                    const existing = getAnnotations();
                    const merged = {{ ...existing, ...loadedAnnotations }};
                    localStorage.setItem('best_match_annotations', JSON.stringify(merged));

                    updateProgress();
                    const count = Object.values(merged).filter(a => a.choice || a.skipped).length;
                    const fileType = isCSV ? 'CSV' : 'JSON';
                    showStatus('Progress loaded from ' + fileType + '! (' + count + ' annotations restored)', true);
                }} catch (err) {{
                    showStatus('Error reading file: ' + err.message, false);
                }}
            }};
            reader.readAsText(file);

            // Reset file input so same file can be loaded again
            event.target.value = '';
        }}

        function showResults() {{
            const annotations = getAnnotations();
            const methods = ['sted', 'deepdiff', 'ted', 'bertscore'];

            // Build item to dataset mapping
            const itemToDataset = {{}};
            document.querySelectorAll('tbody tr').forEach(row => {{
                const itemId = row.id.replace('row_', '');
                itemToDataset[itemId] = row.getAttribute('data-dataset');
            }});

            // Calculate per-dataset stats
            const calcStats = (filterDataset) => {{
                const wins = {{}};
                const totals = {{}};
                methods.forEach(m => {{ wins[m] = 0; totals[m] = 0; }});
                let confirmedCount = 0;
                let skippedCount = 0;

                for (const [itemId, data] of Object.entries(annotations)) {{
                    // Filter by dataset if specified
                    if (filterDataset && filterDataset !== 'all') {{
                        if (itemToDataset[itemId] !== filterDataset) continue;
                    }}

                    if (!data.confirmed) continue;
                    if (data.skipped) {{
                        skippedCount++;
                        continue;
                    }}
                    if (!data.choice) continue;

                    confirmedCount++;
                    const humanChoice = data.choice;
                    const methodPicks = methodMapping[itemId] || {{}};

                    methods.forEach(method => {{
                        if (methodPicks[method]) {{
                            totals[method]++;
                            if (methodPicks[method] === humanChoice) {{
                                wins[method]++;
                            }}
                        }}
                    }});
                }}

                return {{ wins, totals, confirmedCount, skippedCount }};
            }};

            // Calculate for all datasets
            const allStats = calcStats('all');
            const perDatasetStats = {{}};
            datasetsList.forEach(ds => {{
                perDatasetStats[ds] = calcStats(ds);
            }});

            if (allStats.confirmedCount === 0) {{
                alert('No confirmed annotations yet. Please complete some samples first.');
                return;
            }}

            // Build results table HTML
            const buildResultsTable = (stats, title) => {{
                const totalAnnotated = stats.confirmedCount + stats.skippedCount;
                const skippedRate = totalAnnotated > 0 ? ((stats.skippedCount / totalAnnotated) * 100).toFixed(1) : 0;

                const results = methods.map(m => ({{
                    method: m,
                    wins: stats.wins[m],
                    total: stats.totals[m],
                    rate: stats.totals[m] > 0 ? (stats.wins[m] / stats.totals[m] * 100).toFixed(1) : 0
                }})).sort((a, b) => b.rate - a.rate);

                let html = `<h4>${{title}}</h4>`;
                html += `<p><strong>${{stats.confirmedCount}} annotations</strong>`;
                if (stats.skippedCount > 0) html += ` <em>(${{stats.skippedCount}} skipped - ${{skippedRate}}%)</em>`;
                html += `</p>`;

                if (stats.confirmedCount === 0) {{
                    html += `<p><em>No annotations yet</em></p>`;
                    return html;
                }}

                html += `<table class="results-table">
                    <thead><tr><th>Rank</th><th>Method</th><th>Wins</th><th>Total</th><th>Win Rate</th><th></th></tr></thead>
                    <tbody>`;

                results.forEach((r, idx) => {{
                    html += `<tr>
                        <td>${{idx + 1}}</td>
                        <td><strong>${{r.method.toUpperCase()}}</strong></td>
                        <td>${{r.wins}}</td>
                        <td>${{r.total}}</td>
                        <td>${{r.rate}}%</td>
                        <td><div class="win-bar"><div class="win-bar-fill" style="width: ${{r.rate}}%"></div></div></td>
                    </tr>`;
                }});
                html += '</tbody></table>';
                return html;
            }};

            // Build combined HTML with tabs for each dataset
            let html = '<div class="results-tabs">';

            // Overall results first
            html += '<div class="results-section">';
            html += buildResultsTable(allStats, 'Overall Results');
            html += '</div>';

            // Per-dataset results
            datasetsList.forEach(ds => {{
                html += '<div class="results-section">';
                html += buildResultsTable(perDatasetStats[ds], ds.toUpperCase() + ' Results');
                html += '</div>';
            }});

            html += '</div>';

            document.getElementById('results-content').innerHTML = html;
            document.getElementById('results-panel').style.display = 'block';
        }}

        // Update on load and when storage changes
        window.onload = updateProgress;
        window.addEventListener('storage', updateProgress);
        setInterval(updateProgress, 2000);  // Poll for updates
    </script>
</body>
</html>"""
    return html


def export_best_match_interface(
    input_file: str,
    output_dir: str,
    create_csv: bool = True,
):
    """Export best-match selection dataset for annotation."""

    # Load dataset
    print(f"Loading dataset from {input_file}...")
    with open(input_file, "r") as f:
        data = json.load(f)

    all_items = data.get("items", data.get("samples", []))
    print(f"Loaded {len(all_items)} samples")

    # Filter to only include items with >= 2 candidates (method disagreements)
    items = [item for item in all_items if len(item.get("candidates", [])) >= 2]
    print(f"Filtered to {len(items)} samples with >= 2 candidates (method disagreements)")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # Reassign consecutive IDs after filtering
    for i, item in enumerate(items):
        item["id"] = f"rank_{i:04d}"

    # Create individual sample pages
    print("Creating sample pages...")
    for i, item in enumerate(items):
        item_id = item["id"]
        html = create_sample_html(item, i, len(items))

        html_path = os.path.join(samples_dir, f"{item_id}.html")
        with open(html_path, "w") as f:
            f.write(html)

    # Build method mapping for analysis (before creating index)
    method_mapping = {}
    for item in items:
        item_id = item.get("id", "")
        method_picks = item.get("metadata", {}).get("method_picks", {})
        method_mapping[item_id] = method_picks

    # Create index page
    print("Creating index page...")
    index_html = create_index_html(items, output_dir, method_mapping)
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(index_html)

    # Create CSV template
    if create_csv:
        print("Creating CSV template...")
        csv_path = os.path.join(output_dir, "annotation_template.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["item_id", "sample_id", "n_candidates", "choice", "confidence", "notes"])

            for item in items:
                writer.writerow([
                    item.get("id", ""),
                    item.get("sample_id", ""),
                    len(item.get("candidates", [])),
                    "",  # choice
                    "",  # confidence
                    "",  # notes
                ])

    # Create method mapping file (for analysis - keep hidden from annotators)
    method_mapping_path = os.path.join(output_dir, "_method_mapping.json")
    method_mapping = {}
    for item in items:
        item_id = item.get("id", "")
        method_picks = item.get("metadata", {}).get("method_picks", {})
        method_mapping[item_id] = method_picks

    with open(method_mapping_path, "w") as f:
        json.dump(method_mapping, f, indent=2)

    print(f"\nExport complete!")
    print(f"  - Open {output_dir}/index.html in browser for annotation")
    print(f"  - Individual samples in {samples_dir}/")
    print(f"  - CSV template: {csv_path}")
    print(f"  - Method mapping (for analysis): {method_mapping_path}")


def export_combined_interface(
    input_files: List[str],
    output_dir: str,
    create_csv: bool = True,
):
    """Export combined best-match selection dataset for annotation from multiple input files."""

    all_items = []

    # Load and combine all datasets
    for input_file in input_files:
        print(f"Loading dataset from {input_file}...")
        with open(input_file, "r") as f:
            data = json.load(f)

        items = data.get("items", data.get("samples", []))
        # Filter to only include items with >= 2 candidates (method disagreements)
        items = [item for item in items if len(item.get("candidates", [])) >= 2]

        # Filter out items with errors in ground_truth (e.g., "Invalid JSON")
        def has_ground_truth_error(item):
            gt = item.get("ground_truth")
            if gt is None:
                return False
            if isinstance(gt, dict):
                return gt.get("error") is not None
            # gt could be a list - valid ground truth
            return False

        items_before_error_filter = len(items)
        items = [item for item in items if not has_ground_truth_error(item)]
        if items_before_error_filter != len(items):
            print(f"  Filtered out {items_before_error_filter - len(items)} items with ground_truth errors")

        # Infer dataset name from filename if not set in metadata
        inferred_dataset = None
        filename_lower = os.path.basename(input_file).lower()
        if "toucan" in filename_lower:
            inferred_dataset = "toucan"
        elif "sharegpt" in filename_lower:
            inferred_dataset = "sharegpt"
        else:
            # Try to get from top-level metadata
            inferred_dataset = data.get("metadata", {}).get("dataset", "unknown")

        # Set dataset in each item's metadata if not already set
        for item in items:
            if "metadata" not in item:
                item["metadata"] = {}
            if not item["metadata"].get("dataset"):
                item["metadata"]["dataset"] = inferred_dataset

        print(f"  Loaded {len(items)} samples with >= 2 candidates (dataset: {inferred_dataset})")
        all_items.extend(items)

    print(f"\nTotal: {len(all_items)} samples from {len(input_files)} files")

    if not all_items:
        print("No items to export!")
        return

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # Reassign consecutive IDs after combining
    for i, item in enumerate(all_items):
        item["id"] = f"rank_{i:04d}"

    # Create individual sample pages
    print("Creating sample pages...")
    for i, item in enumerate(all_items):
        item_id = item["id"]
        html = create_sample_html(item, i, len(all_items))

        html_path = os.path.join(samples_dir, f"{item_id}.html")
        with open(html_path, "w") as f:
            f.write(html)

    # Build method mapping for analysis (before creating index)
    method_mapping = {}
    for item in all_items:
        item_id = item.get("id", "")
        method_picks = item.get("metadata", {}).get("method_picks", {})
        method_mapping[item_id] = method_picks

    # Create index page
    print("Creating index page...")
    index_html = create_index_html(all_items, output_dir, method_mapping)
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(index_html)

    # Create CSV template
    csv_path = None
    if create_csv:
        print("Creating CSV template...")
        csv_path = os.path.join(output_dir, "annotation_template.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["item_id", "sample_id", "dataset", "n_candidates", "choice", "confidence", "notes"])

            for item in all_items:
                dataset = item.get("metadata", {}).get("dataset", "unknown")
                writer.writerow([
                    item.get("id", ""),
                    item.get("sample_id", ""),
                    dataset,
                    len(item.get("candidates", [])),
                    "",  # choice
                    "",  # confidence
                    "",  # notes
                ])

    # Create method mapping file (for analysis - keep hidden from annotators)
    method_mapping_path = os.path.join(output_dir, "_method_mapping.json")
    with open(method_mapping_path, "w") as f:
        json.dump(method_mapping, f, indent=2)

    # Print summary
    datasets = {}
    for item in all_items:
        dataset = item.get("metadata", {}).get("dataset", "unknown")
        if dataset not in datasets:
            datasets[dataset] = 0
        datasets[dataset] += 1

    print(f"\nExport complete!")
    print(f"  - Total samples: {len(all_items)}")
    for ds, count in datasets.items():
        print(f"    - {ds.upper()}: {count}")
    print(f"  - Open {output_dir}/index.html in browser for annotation")
    print(f"  - Individual samples in {samples_dir}/")
    if csv_path:
        print(f"  - CSV template: {csv_path}")
    print(f"  - Method mapping (for analysis): {method_mapping_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export best-match selection dataset for annotation"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=["ranking_validation_dataset.json"],
        help="Input dataset JSON file(s) - can specify multiple files",
    )
    parser.add_argument(
        "--output-dir",
        default="best_match_annotation",
        help="Output directory for annotation files",
    )

    args = parser.parse_args()

    if len(args.input) == 1:
        # Single file - use original function
        export_best_match_interface(args.input[0], args.output_dir)
    else:
        # Multiple files - use combined function
        export_combined_interface(args.input, args.output_dir)


if __name__ == "__main__":
    main()
