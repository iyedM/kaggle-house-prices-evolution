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


def create_kaggle_session() -> requests.Session:
    """Initialise une session HTTP et récupère le jeton XSRF nécessaire aux appels JSON de Kaggle."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    })
    url = f"{BASE_URL}/competitions/{COMPETITION_SLUG}/code"
    resp = session.get(url, timeout=20)
    resp.raise_for_status()

    xsrf_token = session.cookies.get("XSRF-TOKEN")
    if not xsrf_token:
        raise RuntimeError("Jeton XSRF-TOKEN absent des cookies renvoyés par Kaggle.")
    session.headers.update({"X-XSRF-TOKEN": xsrf_token})
    return session


def list_competition_kernels(
    session: requests.Session,
    page: int = 1,
    page_size: int = 20,
    sort_by: str = "VOTE_COUNT",
) -> list[dict]:
    endpoint = f"{BASE_URL}/api/i/kernels.KernelsService/ListKernels"
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
    resp = session.post(endpoint, json=payload, timeout=20)
    if resp.status_code == 429:
        raise RuntimeError("Limite de requêtes Kaggle atteinte (HTTP 429) sur ListKernels.")
    resp.raise_for_status()
    return resp.json().get("kernels", [])


def list_kernel_versions(session: requests.Session, kernel_id: int) -> list[dict]:
    endpoint = f"{BASE_URL}/api/i/kernels.KernelsService/ListKernelVersions"
    resp = session.post(endpoint, json={"kernelId": kernel_id}, timeout=20)
    if resp.status_code == 429:
        raise RuntimeError("Limite de requêtes Kaggle atteinte (HTTP 429) sur ListKernelVersions.")
    resp.raise_for_status()
    return resp.json().get("items", [])


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_python_notebook(raw_text: str) -> bool:
    """Vérifie via nbformat que le document est un notebook Jupyter v4 valide en Python."""
    if not raw_text.lstrip().startswith("{"):
        return False
    try:
        # Les snapshots Kaggle anciens (2016-2017) ajoutent parfois execution_count sur les
        # cellules markdown ; relax_add_props=True valide le schéma v4 sans rejeter ces snapshots.
        nb = nbformat.reads(raw_text, as_version=4)
        nbformat.validate(nb, relax_add_props=True)
    except Exception:
        return False

    ks_lang = nb.get("metadata", {}).get("kernelspec", {}).get("language", "python")
    if isinstance(ks_lang, str) and ks_lang.lower() != "python":
        return False
    return len(nb.cells) > 0


def download_version_ipynb(session: requests.Session, script_version_id: int, dest_path: Path) -> str | None:
    """Télécharge un snapshot .ipynb, valide sa structure et retourne son empreinte SHA-256."""
    url = f"{BASE_URL}/kernels/scriptcontent/{script_version_id}/download"
    resp = session.get(url, timeout=30)

    if resp.status_code == 429:
        raise RuntimeError(f"Limite de requêtes Kaggle atteinte (HTTP 429) sur le snapshot {script_version_id}.")
    if resp.status_code in (403, 404):
        print(f"  Snapshot {script_version_id} inaccessible (HTTP {resp.status_code}), ignoré.")
        return None
    resp.raise_for_status()

    raw_bytes = resp.content
    raw_text = raw_bytes.decode("utf-8", errors="replace")
    if not validate_python_notebook(raw_text):
        print(f"  Snapshot {script_version_id} rejeté (format notebook Python v4 invalide).")
        return None

    dest_path.write_bytes(raw_bytes)
    return sha256_bytes(raw_bytes)


def rehydrate_from_manifest(manifest_path: Path, out_dir: Path) -> None:
    """Retélécharge exactement les snapshots figés d'un manifeste existant et vérifie leurs SHA-256."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    session = create_kaggle_session()

    for nb in manifest["notebooks"]:
        slug = nb["slug"]
        print(f"Vérification / téléchargement de {slug}...")
        for v in nb["versions"]:
            target = out_dir / v["local_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            expected_hash = v.get("sha256")

            if target.exists() and expected_hash:
                actual_hash = sha256_bytes(target.read_bytes())
                if actual_hash == expected_hash:
                    continue

            digest = download_version_ipynb(session, v["script_version_id"], target)
            if digest is None:
                raise RuntimeError(f"Impossible de récupérer le snapshot {v['script_version_id']} ({slug}).")
            if expected_hash and digest != expected_hash:
                raise RuntimeError(
                    f"Empreinte SHA-256 différente pour {target}: attendu {expected_hash}, obtenu {digest}."
                )
            time.sleep(0.25)
    print("Tous les snapshots du manifeste sont présents et vérifiés.")


def collect_notebooks(
    target_notebooks: int = 6,
    max_versions_per_nb: int = 8,
    max_pages: int = 10,
    sort_by: str = "VOTE_COUNT",
    out_dir: Path = Path("data"),
) -> None:
    raw_dir = out_dir / "notebooks"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ouverture de la session Kaggle ({COMPETITION_SLUG})...")
    session = create_kaggle_session()

    collected_notebooks = []
    page = 1

    while len(collected_notebooks) < target_notebooks and page <= max_pages:
        kernels = list_competition_kernels(session, page=page, page_size=20, sort_by=sort_by)
        if not kernels:
            break

        for k in kernels:
            if len(collected_notebooks) >= target_notebooks:
                break

            kernel_id = k.get("id")
            title = k.get("title", "unknown")
            script_url = k.get("scriptUrl", "")
            slug = script_url.strip("/").replace("code/", "").replace("/", "__")
            if not kernel_id or not slug or "-in-r" in slug:
                continue

            try:
                version_items = list_kernel_versions(session, kernel_id)
            except requests.RequestException as exc:
                print(f"Avertissement : impossible de lister les versions de {slug} ({exc}).")
                continue

            valid_versions = []
            for item in version_items:
                ver_info = item.get("version", {})
                run_info = item.get("run", {})
                if run_info.get("language") != "LANGUAGE_PYTHON":
                    continue
                if run_info.get("sourceType") != "EDITOR_TYPE_NOTEBOOK":
                    continue
                if not run_info.get("id") or ver_info.get("versionNumber") is None:
                    continue
                valid_versions.append({
                    "version_number": int(ver_info["versionNumber"]),
                    "script_version_id": int(run_info["id"]),
                    "date_created": run_info.get("dateCreated"),
                    "lines_total_kaggle": ver_info.get("linesTotal"),
                    "lines_inserted_kaggle": ver_info.get("linesInserted"),
                    "lines_deleted_kaggle": ver_info.get("linesDeleted"),
                    "runtime_seconds": item.get("runInfo", {}).get("runTimeSeconds"),
                })

            valid_versions.sort(key=lambda x: x["version_number"])
            if len(valid_versions) < 3:
                continue

            # Historique complet des dates sans téléchargement supplémentaire (pour analyser le vrai rythme de révision)
            all_python_revisions = [
                {
                    "version_number": v["version_number"],
                    "script_version_id": v["script_version_id"],
                    "date_created": v["date_created"],
                }
                for v in valid_versions
            ]

            if len(valid_versions) > max_versions_per_nb:
                step = (len(valid_versions) - 1) / (max_versions_per_nb - 1)
                indices = sorted({round(i * step) for i in range(max_versions_per_nb)})
                selected_versions = [dict(valid_versions[idx]) for idx in indices]
            else:
                selected_versions = [dict(v) for v in valid_versions]

            nb_folder = raw_dir / slug
            nb_folder.mkdir(parents=True, exist_ok=True)

            downloaded_versions = []
            print(
                f"[{len(collected_notebooks) + 1}/{target_notebooks}] {slug} "
                f"({len(selected_versions)} snapshots sélectionnés sur {len(valid_versions)} versions Python)..."
            )

            for v in selected_versions:
                v_num = v["version_number"]
                sv_id = v["script_version_id"]
                file_path = nb_folder / f"v{v_num:03d}_{sv_id}.ipynb"

                if file_path.exists():
                    raw_bytes = file_path.read_bytes()
                    if validate_python_notebook(raw_bytes.decode("utf-8", errors="replace")):
                        digest = sha256_bytes(raw_bytes)
                    else:
                        file_path.unlink()
                        digest = download_version_ipynb(session, sv_id, file_path)
                else:
                    digest = download_version_ipynb(session, sv_id, file_path)
                    time.sleep(0.2)

                if digest is not None:
                    v["local_path"] = str(file_path.relative_to(out_dir))
                    v["sha256"] = digest
                    downloaded_versions.append(v)

            if len(downloaded_versions) >= 3:
                author_info = version_items[0].get("author", {}) if version_items else {}
                collected_notebooks.append({
                    "kernel_id": kernel_id,
                    "slug": slug,
                    "author_username": author_info.get("userName", slug.split("__")[0]),
                    "author_display_name": author_info.get("displayName", ""),
                    "kaggle_url": f"{BASE_URL}/code/{slug.replace('__', '/')}",
                    "license": "Apache-2.0",
                    "title": title,
                    "total_votes_at_collection": k.get("totalVotes", 0),
                    "raw_versions_returned_by_kaggle": len(version_items),
                    "valid_python_versions_count": len(valid_versions),
                    "sampled_snapshots_count": len(downloaded_versions),
                    "all_python_revisions": all_python_revisions,
                    "versions": downloaded_versions,
                })

        page += 1

    manifest = {
        "collected_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "competition_slug": COMPETITION_SLUG,
        "competition_id": COMPETITION_ID,
        "user_agent": USER_AGENT,
        "selection_criteria": {
            "sort_by": sort_by,
            "min_valid_python_versions": 3,
            "target_notebooks": target_notebooks,
            "max_versions_sampled_per_notebook": max_versions_per_nb,
        },
        "notebooks_count": len(collected_notebooks),
        "notebooks": collected_notebooks,
    }

    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifeste écrit dans {meta_path} ({len(collected_notebooks)} notebooks).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collecte de snapshots historiques de notebooks Kaggle (House Prices).")
    parser.add_argument("--notebooks", type=int, default=6, help="Nombre de notebooks Python à retenir (>= 1)")
    parser.add_argument("--max-versions", type=int, default=8, help="Nombre max de snapshots par notebook (>= 3)")
    parser.add_argument("--max-pages", type=int, default=10, help="Nombre max de pages Kaggle parcourues (>= 1)")
    parser.add_argument("--out-dir", type=Path, default=Path("data"), help="Dossier de sortie")
    parser.add_argument(
        "--from-manifest",
        type=Path,
        default=None,
        help="Chemin d'un metadata.json existant pour retélécharger exactement les mêmes snapshots",
    )
    args = parser.parse_args()

    if args.from_manifest is not None:
        rehydrate_from_manifest(args.from_manifest, args.out_dir)
    else:
        if args.notebooks < 1:
            parser.error("--notebooks doit être supérieur ou égal à 1.")
        if args.max_versions < 3:
            parser.error("--max-versions doit être supérieur ou égal à 3 (seuil minimal de versions par notebook).")
        if args.max_pages < 1:
            parser.error("--max-pages doit être supérieur ou égal à 1.")
        collect_notebooks(
            target_notebooks=args.notebooks,
            max_versions_per_nb=args.max_versions,
            max_pages=args.max_pages,
            out_dir=args.out_dir,
        )
