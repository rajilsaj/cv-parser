from datetime import datetime
from pathlib import Path
import random
import re
import json
from uuid import uuid4

import pandas as pd
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
ALLOWED_EXTENSIONS = {"csv"}

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

# --- Evaluation Metrics Functions ---

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "item"):  # Handles numpy types
            return obj.item()
        return super().default(obj)

def count_turns(text):
    return len(re.findall(r"(User:|Bot:)", str(text)))

def diversity_score(text):
    words = str(text).split()
    return len(set(words)) / max(len(words), 1)

def repetition_score(text):
    words = str(text).split()
    if len(words) < 5:
        return 1
    repeated = sum([words.count(w) > 3 for w in set(words)])
    return 1 - (repeated / len(set(words)))

def structure_score(text):
    return int("User:" in str(text) and "Bot:" in str(text))

def evaluate_text(text):
    score = 0
    turns = count_turns(text)
    if 6 <= turns <= 20:
        score += 1
    if diversity_score(text) > 0.5:
        score += 1
    if repetition_score(text) > 0.7:
        score += 1
    score += structure_score(text)
    return score

def get_file_stats(csv_path: Path) -> dict:
    try:
        # Robust reading: skip bad lines, handle encoding, auto-detect separator
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip', sep=None, engine='python', encoding_errors='replace')
        except Exception:
            # Fallback to comma if auto-detect fails
            df = pd.read_csv(csv_path, on_bad_lines='skip', sep=',', encoding_errors='replace')

        if "conversation_text" not in df.columns:
            return {}
        
        df["quality_score"] = df["conversation_text"].apply(evaluate_text)
        avg_score = df["quality_score"].mean()
        avg_turns = df["conversation_text"].apply(count_turns).mean()
        avg_div = df["conversation_text"].apply(diversity_score).mean()
        
        params = {
            "temperature": 0.4,
            "NUM_VARIANTS": 10,
            "sentiments": ["angry", "confused", "neutral"],
            "sentiment_probs": [0.15, 0.3, 0.55],
            "user_speech": ["casual", "professional", "slang", "typos"],
            "user_speech_probs": [0.5, 0.2, 0.1, 0.2]
        }
        if avg_div < 0.45:
            params["temperature"] += 0.2
        if avg_score < 2:
            params["temperature"] -= 0.1
        if avg_turns < 6:
            params["NUM_VARIANTS"] += 5
        if avg_turns > 20:
            params["temperature"] -= 0.1
            
        return {
            "avg_score": round(float(avg_score), 2),
            "avg_turns": round(float(avg_turns), 2),
            "avg_div": round(float(avg_div), 2),
            "params": params
        }
    except Exception as e:
        print(f"Error evaluating {csv_path}: {e}")
        return {}

# --- End of Metrics ---

def is_allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def rotate_folders(parent_dir: Path, type_label: str):
    """Renames any 'new_type_...' folders to 'old_type_...'"""
    for item in parent_dir.iterdir():
        if item.is_dir() and item.name.startswith(f"new_{type_label}_"):
            old_name = item.name.replace(f"new_{type_label}_", f"old_{type_label}_")
            item.rename(parent_dir / old_name)

def merge_csv_files(file_paths: list[Path], uploaded_info: list[dict]) -> tuple[Path, Path]:
    merged_data = []
    for csv_path in file_paths:
        try:
            # Try auto-detect first, fallback to comma
            try:
                df = pd.read_csv(csv_path, index_col=None, header=0, on_bad_lines='skip', sep=None, engine='python', encoding_errors='replace')
            except Exception:
                df = pd.read_csv(csv_path, index_col=None, header=0, on_bad_lines='skip', sep=',', encoding_errors='replace')
            merged_data.append(df)
        except Exception as e:
            print(f"Error reading {csv_path} during merge: {e}")

    if not merged_data:
        raise ValueError("No valid CSV files could be read.")

    result = pd.concat(merged_data, axis=0, ignore_index=True)
    
    # Rotate results folders
    rotate_folders(RESULTS_DIR, "result")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"new_result_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Merged CSV
    output_name = f"merged_{uuid4().hex}.csv"
    output_path = output_dir / output_name
    result.to_csv(output_path, index=False)

    # Calculate and Save Stats for constituent files
    stats_data = {}
    for info, csv_path in zip(uploaded_info, file_paths):
        stats_data[info['name']] = get_file_stats(csv_path)
    
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats_data, f, indent=4, cls=CustomJSONEncoder)
    
    # Generate Text Report
    report_content = [
        f"CSV MERGE REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 50,
        "\nUPLOADED FILES:",
    ]
    
    for info in uploaded_info:
        report_content.append(f"- {info['name']}")
        report_content.append(f"  Rows: {info['rows']}")
        report_content.append(f"  Headers: {', '.join(info['headers'])}")
        report_content.append("-" * 30)
        
    report_content.extend([
        "\nMERGED RESULT:",
        f"- Total Rows: {len(result)}",
        f"- Total Columns: {len(result.columns)}",
        f"- Headers: {', '.join(result.columns)}",
        "\n" + "=" * 50,
        "Generated by CSV Workspace"
    ])
    
    report_path = output_dir / "report.txt"
    report_path.write_text("\n".join(report_content))
    
    return output_path, report_path

def get_all_results() -> list[dict]:
    results = []
    if not RESULTS_DIR.exists():
        return results

    # Folders are like new_result_20260425_001748 or old_result_20260305_194651
    for folder in sorted(RESULTS_DIR.iterdir(), reverse=True):
        if not folder.is_dir() or not (folder.name.startswith("new_result_") or folder.name.startswith("old_result_")):
            continue
        
        parts = folder.name.split("_")
        if len(parts) < 4:
            continue
            
        date_str = parts[2]
        time_str = parts[3]
        try:
            dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            formatted_date = folder.name

        csv_file = next(folder.glob("merged_*.csv"), None)
        report_file = folder / "report.txt"
        stats_file = folder / "stats.json"
        
        uploaded_files = []
        total_conversations = 0
        
        # Load stats if available
        saved_stats = {}
        if stats_file.exists():
            try:
                with open(stats_file, "r") as f:
                    saved_stats = json.load(f)
            except Exception:
                saved_stats = {}

        if report_file.exists():
            content = report_file.read_text()
            if "UPLOADED FILES:" in content:
                files_part = content.split("UPLOADED FILES:")[1].split("MERGED RESULT:")[0]
                for line in files_part.splitlines():
                    line = line.strip()
                    if line.startswith("- "):
                        filename = line[2:].strip()
                        
                        # Get stats for this file
                        file_stats = saved_stats.get(filename)
                        if not file_stats:
                            # Try to find the file in uploads to compute stats on the fly
                            # This is slow but handles existing records without stats.json
                            found_path = None
                            for upload_folder in UPLOAD_DIR.iterdir():
                                if upload_folder.is_dir():
                                    potential_path = upload_folder / filename
                                    if potential_path.exists():
                                        found_path = potential_path
                                        break
                            if found_path:
                                file_stats = get_file_stats(found_path)
                                # Cache it for this run, but we should probably save it to stats.json
                                saved_stats[filename] = file_stats
                        
                        uploaded_files.append({
                            "name": filename,
                            "stats": file_stats
                        })
            
            if "Total Rows:" in content:
                try:
                    total_conversations = int(content.split("Total Rows:")[1].splitlines()[0].strip())
                except (ValueError, IndexError):
                    total_conversations = 0
        
        # If we computed new stats, save them back to stats.json to avoid re-computing
        if not stats_file.exists() and saved_stats:
            try:
                with open(stats_file, "w") as f:
                    json.dump(saved_stats, f, indent=4, cls=CustomJSONEncoder)
            except Exception:
                pass

        results.append({
            "folder_name": folder.name,
            "date": formatted_date,
            "csv_filename": str(csv_file.relative_to(BASE_DIR)) if csv_file else None,
            "report_filename": str(report_file.relative_to(BASE_DIR)) if report_file.exists() else None,
            "uploaded_files": uploaded_files,
            "total_conversations": total_conversations,
            "id": f"{date_str}_{time_str}"
        })
    return results

def build_results_context(data_file: Path, source_label: str) -> dict:
    result_df = pd.read_csv(data_file)
    preview_rows = result_df.head(12).to_dict(orient="records")
    preview_headers = result_df.columns.tolist()

    chart_labels: list[str] = []
    chart_values: list[int] = []
    if not result_df.empty and preview_headers:
        first_column = result_df.iloc[:, 0].astype(str)
        value_counts = first_column.value_counts(dropna=False).head(10)
        chart_labels = value_counts.index.tolist()
        chart_values = value_counts.values.tolist()

    return {
        "merged_filename": str(data_file.relative_to(BASE_DIR)),
        "total_rows": len(result_df),
        "total_columns": len(preview_headers),
        "headers": preview_headers,
        "preview_rows": preview_rows,
        "chart_labels": chart_labels,
        "chart_values": chart_values,
        "source_label": source_label,
    }

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    uploaded_files = request.files.getlist("files")
    saved_files: list[Path] = []
    uploaded_info: list[dict] = []
    base_headers = None
    
    # Rotate upload folders
    rotate_folders(UPLOAD_DIR, "upload")
    
    # Create new timestamped upload directory
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_upload_dir = UPLOAD_DIR / f"new_upload_{timestamp_str}"
    session_upload_dir.mkdir(parents=True, exist_ok=True)

    for file in uploaded_files:
        if not file.filename:
            continue
        if not is_allowed_file(file.filename):
            continue

        safe_name = secure_filename(file.filename)
        file_path = session_upload_dir / safe_name
        file.save(file_path)
        saved_files.append(file_path)
        
        # Calculate row count and check headers
        df = pd.read_csv(file_path)
        current_headers = list(df.columns)
        
        if base_headers is None:
            base_headers = current_headers
            header_status = "ok"
        else:
            header_status = "ok" if current_headers == base_headers else "mismatch"
            
        uploaded_info.append({
            "name": safe_name, 
            "rows": len(df), 
            "header_status": header_status,
            "headers": current_headers
        })

    if not saved_files:
        return render_template(
            "index.html",
            error="Please upload at least one CSV file."
        )

    total_rows = sum(info["rows"] for info in uploaded_info)
    for info in uploaded_info:
        info["percentage"] = (info["rows"] / total_rows * 100) if total_rows > 0 else 0

    merged_file, report_file = merge_csv_files(saved_files, uploaded_info)
    results = build_results_context(merged_file, source_label="Uploaded & Merged CSV")
    results["uploaded_files"] = uploaded_info
    results["report_filename"] = str(report_file.relative_to(BASE_DIR))

    return render_template(
        "index.html",
        **results,
    )

@app.route("/visualize")
def visualize():
    history = get_all_results()
    return render_template("visualize.html", history=history)

@app.route("/download/<path:filepath>", methods=["GET"])
def download(filepath: str):
    file_path = BASE_DIR / filepath
    
    # Security: check if file is within ALLOWED directories
    is_safe = False
    for parent in file_path.parents:
        if parent == RESULTS_DIR or parent == UPLOAD_DIR:
            is_safe = True
            break
            
    if not is_safe or not file_path.exists():
        return render_template("index.html", error="File not found or access denied.")
        
    return send_file(file_path, as_attachment=True, download_name=file_path.name)

if __name__ == "__main__":
    app.run(debug=True)
