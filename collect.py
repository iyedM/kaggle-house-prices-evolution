import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import requests

COMPETITION_SLUG = "house-prices-advanced-regression-techniques"
COMPETITION_ID = 5407
BASE_URL = "https://www.kaggle.com"
USER_AGENT = "kaggle-house-prices-evolution/0.1 (academic research mini-project; python-requests)"

DEFAULT_TARGET_NOTEBOOKS = 6
DEFAULT_MAX_VERSIONS = 8
DEFAULT_MAX_PAGES = 10
DEFAULT_PAGE_SIZE = 20
MIN_VALID_VERSIONS = 3

HTTP_TIMEOUT_S = 20
DOWNLOAD_TIMEOUT_S = 30
REQUEST_DELAY_S = 0.25


def create_kaggle_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    })
    resp = session.get(f"{BASE_URL}/competitions/{COMPETITION_SLUG}/code", timeout=HTTP_TIMEOUT_S)
    resp.raise_for_status()

    xsrf_token = session.cookies.get("XSRF-TOKEN")
    if not xsrf_token:
        raise RuntimeError("Cookie XSRF-TOKEN introuvable.")
    session.headers.update({"X-XSRF-TOKEN": xsrf_token})
    return session


def _post_kaggle_json(session: requests.Session, endpoint_name: str, payload: dict) -> dict:
    url = f"{BASE_URL}/api/i/kernels.KernelsService/{endpoint_name}"
    resp = session.post(url, json=payload, timeout=HTTP_TIMEOUT_S)
    if resp.status_code == 429:
        raise RuntimeError(f"HTTP 429 (rate limit Kaggle) : {endpoint_name}")
    resp.raise_for_status()
    return resp.json()


def list_competition_kernels(
    session: requests.Session,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    sort_by: str = "VOTE_COUNT",
) -> list[dict]:
    payload = {
        "kernelFilterCriteria": {
            "search": "",
            "listRequest": {
                "competitionId": COMPETITION_ID,
                "sortBy": sort_by,
                "pageSize": page_size,
                "group": "EVERYONE",
                "page": page,
            },
        },
        "detailFilterCriteria": {
            "deletedAccessBehavior": "RETURN_NOTHING",
            "unauthorizedAccessBehavior": "RETURN_NOTHING",
            "excludeResultsFilesOutputs": True,
            "wantOutputFiles": False,
        },
    }
    return _post_kaggle_json(session, "ListKernels", payload).get("kernels", [])


def list_kernel_versions(session: requests.Session, kernel_id: int) -> list[dict]:
    return _post_kaggle_json(session, "ListKernelVersions", {"kernelId": kernel_id}).get("items", [])


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_python_notebook(data: bytes) -> bool:
    try:
        raw_text = data.decode("utf-8")
        if not raw_text.lstrip().startswith("{"):
            return False
        # Les snapshots Kaggle anciens (2016-2017) ajoutent parfois execution_count sur les
        # cellules markdown ; relax_add_props=True valide le schéma v4 sans rejeter ces snapshots.
        nb = nbformat.reads(raw_text, as_version=4)
        nbformat.validate(nb, relax_add_props=True)
    except (UnicodeDecodeError, ValueError, KeyError, TypeError, nbformat.ValidationError):
        return False

    ks = (nb.get("metadata") or {}).get("kernelspec") or {}
    ks_lang = ks.get("language", "python")
    if not isinstance(ks_lang, str) or ks_lang.lower() != "python":
        return False
    return len(nb.cells) > 0


def download_version_ipynb(session: requests.Session, script_version_id: int, dest_path: Path) -> str | None:
    url = f"{BASE_URL}/kernels/scriptcontent/{script_version_id}/download"
    resp = session.get(url, timeout=DOWNLOAD_TIMEOUT_S)

    if resp.status_code == 429:
        raise RuntimeError(f"HTTP 429 (rate limit Kaggle) : snapshot {script_version_id}")
    if resp.status_code in (403, 404):
        print(f"  Snapshot {script_version_id} inaccessible (HTTP {resp.status_code}), ignoré.")
        return None
    resp.raise_for_status()

    raw_bytes = resp.content
    if not validate_python_notebook(raw_bytes):
        print(f"  Snapshot {script_version_id} rejeté (notebook Python v4 invalide).")
        return None

    dest_path.write_bytes(raw_bytes)
    return sha256_bytes(raw_bytes)


def rehydrate_from_manifest(manifest_path: Path, out_dir: Path) -> None:
    """Vérifie les snapshots locaux d'un manifeste et retélécharge ceux qui manquent."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    session: requests.Session | None = None
    total = 0

    for nb in manifest["notebooks"]:
        slug = nb["slug"]
        for v in nb["versions"]:
            total += 1
            target = out_dir / v["local_path"]
            expected_hash = v.get("sha256")

            if target.exists() and expected_hash and sha256_bytes(target.read_bytes()) == expected_hash:
                continue

            if session is None:
                session = create_kaggle_session()
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = download_version_ipynb(session, v["script_version_id"], target)
            if digest is None:
                raise RuntimeError(f"Échec du téléchargement du snapshot {v['script_version_id']} ({slug}).")
            if expected_hash and digest != expected_hash:
                raise RuntimeError(f"SHA-256 différent pour {target} : attendu {expected_hash}, obtenu {digest}.")
            time.sleep(REQUEST_DELAY_S)

    print(f"Manifeste vérifié : {total} snapshots conformes.")


def extract_valid_python_versions(version_items: list[dict]) -> list[dict]:
    valid: list[dict] = []
    for item in version_items:
        ver_info = item.get("version") or {}
        run_info = item.get("run") or {}
        if run_info.get("language") != "LANGUAGE_PYTHON":
            continue
        if run_info.get("sourceType") != "EDITOR_TYPE_NOTEBOOK":
            continue
        if not run_info.get("id") or ver_info.get("versionNumber") is None:
            continue
        valid.append({
            "version_number": int(ver_info["versionNumber"]),
            "script_version_id": int(run_info["id"]),
            "date_created": run_info.get("dateCreated"),
            "lines_total_kaggle": ver_info.get("linesTotal"),
            "lines_inserted_kaggle": ver_info.get("linesInserted"),
            "lines_deleted_kaggle": ver_info.get("linesDeleted"),
            "runtime_seconds": (item.get("runInfo") or {}).get("runTimeSeconds"),
        })
    valid.sort(key=lambda x: x["version_number"])
    return valid


def sample_versions_evenly(versions: list[dict], max_count: int) -> list[dict]:
    if max_count < MIN_VALID_VERSIONS:
        raise ValueError(f"max_count doit être >= {MIN_VALID_VERSIONS}")
    if len(versions) <= max_count:
        return [dict(v) for v in versions]
    step = (len(versions) - 1) / (max_count - 1)
    indices = sorted({round(i * step) for i in range(max_count)})
    return [dict(versions[idx]) for idx in indices]


def ensure_snapshot_cached(
    session: requests.Session,
    version: dict,
    nb_folder: Path,
    out_dir: Path,
) -> dict | None:
    v_num = version["version_number"]
    sv_id = version["script_version_id"]
    file_path = nb_folder / f"v{v_num:03d}_{sv_id}.ipynb"

    if file_path.exists():
        raw_bytes = file_path.read_bytes()
        if validate_python_notebook(raw_bytes):
            digest = sha256_bytes(raw_bytes)
        else:
            file_path.unlink()
            digest = download_version_ipynb(session, sv_id, file_path)
            time.sleep(REQUEST_DELAY_S)
    else:
        digest = download_version_ipynb(session, sv_id, file_path)
        time.sleep(REQUEST_DELAY_S)

    if digest is None:
        return None
    entry = dict(version)
    entry["local_path"] = str(file_path.relative_to(out_dir))
    entry["sha256"] = digest
    return entry


def collect_notebooks(
    target_notebooks: int = DEFAULT_TARGET_NOTEBOOKS,
    max_versions_per_nb: int = DEFAULT_MAX_VERSIONS,
    max_pages: int = DEFAULT_MAX_PAGES,
    sort_by: str = "VOTE_COUNT",
    out_dir: Path = Path("data"),
) -> None:
    raw_dir = out_dir / "notebooks"
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = create_kaggle_session()

    collected_notebooks: list[dict] = []
    page = 1

    while len(collected_notebooks) < target_notebooks and page <= max_pages:
        kernels = list_competition_kernels(session, page=page, page_size=DEFAULT_PAGE_SIZE, sort_by=sort_by)
        if not kernels:
            break

        for kernel in kernels:
            if len(collected_notebooks) >= target_notebooks:
                break

            kernel_id = kernel.get("id")
            script_url = (kernel.get("scriptUrl") or "").strip("/")
            slug = script_url.removeprefix("code/").replace("/", "__")
            # Écarte d'anciens notebooks R étiquetés LANGUAGE_PYTHON par l'API Kaggle (ex. rtatman)
            if not kernel_id or not slug or "-in-r" in slug:
                continue

            try:
                version_items = list_kernel_versions(session, kernel_id)
            except requests.RequestException as exc:
                print(f"Avertissement : versions inaccessibles pour {slug} ({exc}).")
                continue

            valid_versions = extract_valid_python_versions(version_items)
            if len(valid_versions) < MIN_VALID_VERSIONS:
                continue

            selected_versions = sample_versions_evenly(valid_versions, max_versions_per_nb)
            nb_folder = raw_dir / slug
            nb_folder.mkdir(parents=True, exist_ok=True)

            print(
                f"[{len(collected_notebooks) + 1}/{target_notebooks}] {slug} "
                f"({len(selected_versions)}/{len(valid_versions)} versions Python)"
            )
            downloaded: list[dict] = []
            for v in selected_versions:
                cached = ensure_snapshot_cached(session, v, nb_folder, out_dir)
                if cached is not None:
                    downloaded.append(cached)
            if len(downloaded) < MIN_VALID_VERSIONS:
                continue

            author_info = (version_items[0].get("author") or {}) if version_items else {}
            collected_notebooks.append({
                "kernel_id": kernel_id,
                "slug": slug,
                "author_username": author_info.get("userName", slug.split("__")[0]),
                "author_display_name": author_info.get("displayName", ""),
                "kaggle_url": f"{BASE_URL}/{script_url}",
                "license": "Apache-2.0",
                "title": kernel.get("title", "unknown"),
                "total_votes_at_collection": kernel.get("totalVotes", 0),
                "raw_versions_returned_by_kaggle": len(version_items),
                "valid_python_versions_count": len(valid_versions),
                "sampled_snapshots_count": len(downloaded),
                "all_python_revisions": [
                    {
                        "version_number": v["version_number"],
                        "script_version_id": v["script_version_id"],
                        "date_created": v["date_created"],
                    }
                    for v in valid_versions
                ],
                "versions": downloaded,
            })

        page += 1

    manifest = {
        "collected_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "competition_slug": COMPETITION_SLUG,
        "competition_id": COMPETITION_ID,
        "user_agent": USER_AGENT,
        "selection_criteria": {
            "sort_by": sort_by,
            "min_valid_python_versions": MIN_VALID_VERSIONS,
            "target_notebooks": target_notebooks,
            "max_versions_sampled_per_notebook": max_versions_per_nb,
        },
        "notebooks_count": len(collected_notebooks),
        "notebooks": collected_notebooks,
    }

    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifeste écrit : {meta_path} ({len(collected_notebooks)} notebooks).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collecte de versions historiques de notebooks Kaggle (House Prices).")
    parser.add_argument("--notebooks", type=int, default=DEFAULT_TARGET_NOTEBOOKS, help="Nombre de notebooks (>= 1)")
    parser.add_argument(
        "--max-versions",
        type=int,
        default=DEFAULT_MAX_VERSIONS,
        help=f"Snapshots max par notebook (>= {MIN_VALID_VERSIONS})",
    )
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES, help="Pages Kaggle max (>= 1)")
    parser.add_argument("--out-dir", type=Path, default=Path("data"), help="Dossier de sortie")
    parser.add_argument(
        "--from-manifest",
        type=Path,
        default=None,
        help="Manifeste metadata.json existant à vérifier ou réhydrater",
    )
    args = parser.parse_args()

    if args.from_manifest is not None:
        rehydrate_from_manifest(args.from_manifest, args.out_dir)
        return

    if args.notebooks < 1:
        parser.error("--notebooks doit être >= 1")
    if args.max_versions < MIN_VALID_VERSIONS:
        parser.error(f"--max-versions doit être >= {MIN_VALID_VERSIONS}")
    if args.max_pages < 1:
        parser.error("--max-pages doit être >= 1")

    collect_notebooks(
        target_notebooks=args.notebooks,
        max_versions_per_nb=args.max_versions,
        max_pages=args.max_pages,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
