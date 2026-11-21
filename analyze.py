import ast
import difflib
import hashlib
import json
import statistics
import warnings
from datetime import datetime
from pathlib import Path

import nbformat
import pandas as pd

SECONDS_PER_HOUR = 3600.0

ML_LIBRARIES = {
    "pandas", "numpy", "sklearn", "scipy", "statsmodels",
    "matplotlib", "seaborn", "xgboost", "lightgbm", "catboost",
    "tensorflow", "keras", "torch", "optuna", "h2o",
}


def _parse_ast_quietly(source: str) -> ast.AST | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(source)
    except SyntaxError:
        return None


def _collect_ml_imports(tree: ast.AST) -> set[str]:
    detected: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                pkg = alias.name.split(".")[0]
                if pkg in ML_LIBRARIES:
                    detected.add(pkg)
        elif isinstance(node, ast.ImportFrom) and node.module:
            pkg = node.module.split(".")[0]
            if pkg in ML_LIBRARIES:
                detected.add(pkg)
    return detected


def extract_imports_from_code(code_text: str) -> set[str]:
    cleaned_lines = [
        line for line in code_text.splitlines()
        if not line.lstrip().startswith(("%", "!"))
    ]
    tree = _parse_ast_quietly("\n".join(cleaned_lines))
    if tree is not None:
        return _collect_ml_imports(tree)

    detected: set[str] = set()
    for line in cleaned_lines:
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            line_tree = _parse_ast_quietly(stripped)
            if line_tree is not None:
                detected.update(_collect_ml_imports(line_tree))
    return detected


def load_validated_notebook(
    ipynb_path: Path,
    expected_sha256: str | None = None,
) -> nbformat.NotebookNode:
    if not ipynb_path.exists():
        raise FileNotFoundError(f"Snapshot manquant : {ipynb_path}")
    raw_bytes = ipynb_path.read_bytes()
    if expected_sha256:
        actual_sha = hashlib.sha256(raw_bytes).hexdigest()
        if actual_sha != expected_sha256:
            raise ValueError(f"SHA-256 invalide pour {ipynb_path}")
    return nbformat.reads(raw_bytes.decode("utf-8"), as_version=4)


def extract_notebook_info(ipynb_path: Path, expected_sha256: str | None = None) -> dict:
    nb = load_validated_notebook(ipynb_path, expected_sha256)

    code_cells = 0
    markdown_cells = 0
    code_lines: list[str] = []
    markdown_lines = 0
    detected_libs: set[str] = set()

    for cell in nb.cells:
        ctype = cell.get("cell_type", "")
        text = cell.get("source", "")
        if isinstance(text, list):
            text = "".join(text)

        if ctype == "code":
            code_cells += 1
            code_lines.extend(
                line.rstrip()
                for line in text.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
            detected_libs.update(extract_imports_from_code(text))
        elif ctype == "markdown":
            markdown_cells += 1
            # En markdown, '#' désigne un titre de section et doit être compté.
            markdown_lines += sum(1 for line in text.splitlines() if line.strip())

    return {
        "code_cells": code_cells,
        "markdown_cells": markdown_cells,
        "code_lines_count": len(code_lines),
        "markdown_lines_count": markdown_lines,
        "code_lines": code_lines,
        "ml_libraries": sorted(detected_libs),
    }


def compute_sequence_diff(prev_lines: list[str], curr_lines: list[str]) -> dict:
    """Compte les ajouts et suppressions de lignes (un bloc remplacé compte dans les deux)."""
    matcher = difflib.SequenceMatcher(None, prev_lines, curr_lines, autojunk=False)
    inserted = 0
    deleted = 0
    replaced_prev = 0
    replaced_curr = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            inserted += j2 - j1
        elif tag == "delete":
            deleted += i2 - i1
        elif tag == "replace":
            replaced_prev += i2 - i1
            replaced_curr += j2 - j1

    total_added = inserted + replaced_curr
    total_removed = deleted + replaced_prev
    return {
        "lines_inserted": total_added,
        "lines_deleted": total_removed,
        "lines_changed_between_sampled_snapshots": total_added + total_removed,
    }


def parse_iso_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        return None


def compute_elapsed_hours(
    prev_dt: datetime | None,
    curr_dt: datetime | None,
    context: str,
) -> float | None:
    if prev_dt is None or curr_dt is None:
        return None
    delta_hours = (curr_dt - prev_dt).total_seconds() / SECONDS_PER_HOUR
    if delta_hours < 0:
        print(f"Avertissement : horodatage inversé sur {context}, intervalle ignoré.")
        return None
    return round(delta_hours, 3)


def compute_revision_intervals(slug: str, author: str, revisions: list[dict]) -> list[dict]:
    rows: list[dict] = []
    prev_dt: datetime | None = None
    prev_num: int | None = None

    for rev in revisions:
        curr_num = int(rev["version_number"])
        curr_dt = parse_iso_date(rev.get("date_created"))
        if prev_num is not None:
            hours = compute_elapsed_hours(prev_dt, curr_dt, f"{slug} (v{prev_num}->v{curr_num})")
            if hours is not None:
                rows.append({
                    "slug": slug,
                    "notebook": author,
                    "from_version": prev_num,
                    "to_version": curr_num,
                    "hours_since_prev_revision": hours,
                })
        prev_dt = curr_dt
        prev_num = curr_num

    return rows


def analyze_notebook_snapshots(
    slug: str,
    author: str,
    versions: list[dict],
    data_dir: Path,
) -> list[dict]:
    first_vnum = int(versions[0]["version_number"])
    last_vnum = int(versions[-1]["version_number"])
    vnum_span = max(1, last_vnum - first_vnum)

    rows: list[dict] = []
    prev_info: dict | None = None
    prev_dt: datetime | None = None
    prev_vnum: int | None = None

    for idx, version in enumerate(versions):
        ipynb_path = data_dir / version["local_path"]
        info = extract_notebook_info(ipynb_path, expected_sha256=version.get("sha256"))
        curr_dt = parse_iso_date(version.get("date_created"))
        curr_vnum = int(version["version_number"])
        norm_prog = round((curr_vnum - first_vnum) / vnum_span, 4)

        if prev_info is None:
            diff_stats = {
                "lines_inserted": 0,
                "lines_deleted": 0,
                "lines_changed_between_sampled_snapshots": 0,
            }
            version_gap = 0
            delta_cells = 0
            hours_elapsed = None
        else:
            diff_stats = compute_sequence_diff(prev_info["code_lines"], info["code_lines"])
            version_gap = curr_vnum - prev_vnum
            delta_cells = info["code_cells"] - prev_info["code_cells"]
            hours_elapsed = compute_elapsed_hours(prev_dt, curr_dt, f"{slug} (v{prev_vnum}->v{curr_vnum})")

        rows.append({
            "slug": slug,
            "notebook": author,
            "snapshot_rank": idx + 1,
            "version_number": curr_vnum,
            "normalized_progress": norm_prog,
            "version_gap": version_gap,
            "script_version_id": version["script_version_id"],
            "date_created": version.get("date_created"),
            "hours_since_prev_snapshot": hours_elapsed,
            "code_cells": info["code_cells"],
            "markdown_cells": info["markdown_cells"],
            "delta_code_cells": delta_cells,
            "code_lines": info["code_lines_count"],
            "markdown_lines": info["markdown_lines_count"],
            "lines_inserted": diff_stats["lines_inserted"],
            "lines_deleted": diff_stats["lines_deleted"],
            "lines_changed_between_sampled_snapshots": diff_stats["lines_changed_between_sampled_snapshots"],
            "ml_libs_count": len(info["ml_libraries"]),
            "ml_libraries": ",".join(info["ml_libraries"]),
        })

        prev_info = info
        prev_dt = curr_dt
        prev_vnum = curr_vnum

    return rows


def summarize_notebook(
    nb_meta: dict,
    snapshot_rows: list[dict],
    interval_rows: list[dict],
) -> dict:
    gaps = [r["version_gap"] for r in snapshot_rows[1:]]
    changes = [r["lines_changed_between_sampled_snapshots"] for r in snapshot_rows[1:]]
    intervals = [r["hours_since_prev_revision"] for r in interval_rows]

    return {
        "slug": nb_meta["slug"],
        "title": nb_meta.get("title"),
        "raw_versions_kaggle": nb_meta.get("raw_versions_returned_by_kaggle"),
        "valid_python_versions": nb_meta.get("valid_python_versions_count"),
        "sampled_snapshots": len(snapshot_rows),
        "avg_version_gap": round(sum(gaps) / len(gaps), 2) if gaps else 0.0,
        "initial_code_lines": snapshot_rows[0]["code_lines"],
        "final_code_lines": snapshot_rows[-1]["code_lines"],
        "initial_markdown_lines": snapshot_rows[0]["markdown_lines"],
        "final_markdown_lines": snapshot_rows[-1]["markdown_lines"],
        "avg_lines_changed_between_snapshots": round(sum(changes) / len(changes), 1) if changes else 0.0,
        "max_lines_changed_between_snapshots": max(changes, default=0),
        "median_hours_consecutive_revisions": round(statistics.median(intervals), 2) if intervals else None,
        "detected_ml_libraries_final": snapshot_rows[-1]["ml_libraries"],
    }


def analyze_collected_notebooks(data_dir: Path = Path("data")) -> None:
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Fichier {meta_path} introuvable. Exécutez d'abord collect.py.")

    manifest = json.loads(meta_path.read_text(encoding="utf-8"))

    version_rows: list[dict] = []
    interval_rows: list[dict] = []
    summary_rows: list[dict] = []

    for nb in manifest["notebooks"]:
        versions = nb.get("versions") or []
        if not versions:
            continue

        slug = nb["slug"]
        author = slug.split("__")[0]

        nb_intervals = compute_revision_intervals(slug, author, nb.get("all_python_revisions") or [])
        nb_snapshots = analyze_notebook_snapshots(slug, author, versions, data_dir)

        interval_rows.extend(nb_intervals)
        version_rows.extend(nb_snapshots)
        summary_rows.append(summarize_notebook(nb, nb_snapshots, nb_intervals))

    df_versions = pd.DataFrame(version_rows)
    df_intervals = pd.DataFrame(interval_rows)
    df_summary = pd.DataFrame(summary_rows)

    df_versions.to_csv(data_dir / "versions_evolution.csv", index=False)
    df_intervals.to_csv(data_dir / "all_revisions_intervals.csv", index=False)
    df_summary.to_csv(data_dir / "notebooks_summary.csv", index=False)

    print(
        f"Analyse terminée : {len(df_summary)} notebooks, "
        f"{len(df_versions)} snapshots, {len(df_intervals)} intervalles consécutifs."
    )
    summary_cols = [
        "slug", "valid_python_versions", "sampled_snapshots", "avg_version_gap",
        "initial_code_lines", "final_code_lines", "avg_lines_changed_between_snapshots",
        "median_hours_consecutive_revisions",
    ]
    print(df_summary[summary_cols].to_string(index=False))


if __name__ == "__main__":
    analyze_collected_notebooks()
