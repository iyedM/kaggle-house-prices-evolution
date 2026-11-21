from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIG_DPI = 180
DUAL_FIGSIZE = (13.8, 5.2)
HEATMAP_FIGSIZE = (10.5, 4.8)
LOG_HOURS_FLOOR = 0.01

TRAJ_ABSENT = 0
TRAJ_TRANSIENT = 1
TRAJ_REMOVED = 2
TRAJ_ADDED = 3
TRAJ_CONSTANT = 4

TRAJECTORY_SPECS = [
    (TRAJ_ABSENT, "", "#f7f7f7"),
    (TRAJ_TRANSIENT, "Transitoire", "#fdd0a2"),
    (TRAJ_REMOVED, "Retirée", "#fc9272"),
    (TRAJ_ADDED, "Ajoutée", "#74c476"),
    (TRAJ_CONSTANT, "Constante", "#2b8cbe"),
]
TRAJECTORY_LABELS = {code: label for code, label, _ in TRAJECTORY_SPECS}
TRAJECTORY_COLORS = [color for _, _, color in TRAJECTORY_SPECS]


def parse_ml_libraries(raw_value: object) -> set[str]:
    if raw_value is None or pd.isna(raw_value):
        return set()
    return {lib.strip() for lib in str(raw_value).split(",") if lib.strip()}


def classify_library_trajectory(presence: list[bool]) -> int:
    if not any(presence):
        return TRAJ_ABSENT
    if all(presence):
        return TRAJ_CONSTANT
    transitions = sum(1 for prev, curr in zip(presence, presence[1:]) if prev != curr)
    if transitions == 1:
        return TRAJ_ADDED if presence[-1] else TRAJ_REMOVED
    return TRAJ_TRANSIENT


def build_library_trajectory_matrix(df_ver: pd.DataFrame, author_order: list[str]) -> pd.DataFrame:
    all_libs = sorted({
        lib
        for val in df_ver["ml_libraries"]
        for lib in parse_ml_libraries(val)
    })

    status_by_author: dict[str, list[int]] = {}
    for _, grp in df_ver.groupby("slug", sort=False):
        author = grp.iloc[0]["notebook"]
        snapshots_libs = [parse_ml_libraries(val) for val in grp["ml_libraries"]]
        status_by_author[author] = [
            classify_library_trajectory([lib in snap for snap in snapshots_libs])
            for lib in all_libs
        ]

    matrix = [status_by_author[author] for author in author_order]
    return pd.DataFrame(matrix, index=author_order, columns=all_libs)


def plot_code_evolution(df_ver: pd.DataFrame, label_order: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=DUAL_FIGSIZE)

    sns.lineplot(
        data=df_ver,
        x="normalized_progress",
        y="code_lines",
        hue="notebook_label",
        hue_order=label_order,
        marker="o",
        linewidth=1.9,
        ax=axes[0],
    )
    axes[0].set_title("Évolution du volume de code sur le cycle de vie normalisé", fontsize=11)
    axes[0].set_xlabel("Progression relative dans l'historique (0.0 = 1re version, 1.0 = dernière)")
    axes[0].set_ylabel("Lignes non vides hors '#' (cellules de code)")
    axes[0].legend(title="Auteur (versions Python, pas moyen)", fontsize=8, title_fontsize=8.5, loc="best")

    df_transitions = df_ver[df_ver["snapshot_rank"] > 1]
    sns.lineplot(
        data=df_transitions,
        x="normalized_progress",
        y="lines_changed_between_sampled_snapshots",
        hue="notebook_label",
        hue_order=label_order,
        marker="s",
        linewidth=1.8,
        ax=axes[1],
    )
    axes[1].set_title("Lignes modifiées par transition échantillonnée", fontsize=11)
    axes[1].set_xlabel("Progression relative dans l'historique (0.0 -> 1.0)")
    axes[1].set_ylabel("Lignes insérées + supprimées (SequenceMatcher)")
    axes[1].legend(title="Auteur (versions Python, pas moyen)", fontsize=8, title_fontsize=8.5, loc="best")

    plt.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def plot_intervals_and_cells(
    df_int: pd.DataFrame,
    df_ver: pd.DataFrame,
    author_order: list[str],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=DUAL_FIGSIZE)

    df_int_clean = df_int.dropna(subset=["hours_since_prev_revision"]).copy()
    df_int_clean["hours_plot"] = df_int_clean["hours_since_prev_revision"].clip(lower=LOG_HOURS_FLOOR)

    sns.boxplot(
        data=df_int_clean,
        x="hours_plot",
        y="notebook",
        order=author_order,
        color="#c6dbef",
        fliersize=0,
        ax=axes[0],
    )
    sns.stripplot(
        data=df_int_clean,
        x="hours_plot",
        y="notebook",
        order=author_order,
        color="#08519c",
        size=4.5,
        jitter=0.18,
        alpha=0.75,
        ax=axes[0],
    )
    axes[0].set_xscale("log")
    axes[0].set_title("Délais entre révisions Python consécutives (échelle log)", fontsize=11)
    axes[0].set_xlabel("Heures écoulées entre deux versions consécutives v(k) -> v(k+1)")
    axes[0].set_ylabel("Notebook (auteur)")

    first_last_rows = []
    for _, grp in df_ver.groupby("slug", sort=False):
        first_row = grp.iloc[0]
        last_row = grp.iloc[-1]
        author = first_row["notebook"]
        first_last_rows.append({"Notebook": author, "Snapshot": "1re version", "Cellules de code": first_row["code_cells"]})
        first_last_rows.append({"Notebook": author, "Snapshot": "Dernière version", "Cellules de code": last_row["code_cells"]})

    df_first_last = pd.DataFrame(first_last_rows)
    sns.barplot(
        data=df_first_last,
        x="Cellules de code",
        y="Notebook",
        order=author_order,
        hue="Snapshot",
        palette=["#9ecae1", "#2171b5"],
        ax=axes[1],
    )
    axes[1].set_title("Cellules de code : 1re vs dernière version", fontsize=11)
    axes[1].set_xlabel("Nombre de cellules de code")
    axes[1].set_ylabel("")
    axes[1].legend(title="", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def plot_ml_library_matrix(mat_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE)
    cmap = sns.color_palette(TRAJECTORY_COLORS, as_cmap=True)
    annot_labels = mat_df.replace(TRAJECTORY_LABELS)

    sns.heatmap(
        mat_df,
        cmap=cmap,
        vmin=TRAJ_ABSENT,
        vmax=TRAJ_CONSTANT,
        cbar=False,
        annot=annot_labels,
        fmt="",
        linewidths=0.8,
        linecolor="#d9d9d9",
        annot_kws={"fontsize": 8.5},
        ax=ax,
    )
    ax.set_title(
        "Trajectoire des bibliothèques ML détectées sur l'ensemble des snapshots échantillonnés",
        fontsize=11,
    )
    ax.set_xlabel("Bibliothèque Python détectée par analyse statique (AST)")
    ax.set_ylabel("Notebook (auteur)")
    plt.xticks(rotation=25, ha="right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def generate_plots(data_dir: Path = Path("data"), fig_dir: Path = Path("figures")) -> None:
    versions_csv = data_dir / "versions_evolution.csv"
    intervals_csv = data_dir / "all_revisions_intervals.csv"
    summary_csv = data_dir / "notebooks_summary.csv"

    for req_file in (versions_csv, intervals_csv, summary_csv):
        if not req_file.exists():
            raise FileNotFoundError(f"Fichier {req_file} introuvable. Exécutez d'abord analyze.py.")

    df_ver = pd.read_csv(versions_csv).sort_values(["slug", "version_number"]).reset_index(drop=True)
    df_int = pd.read_csv(intervals_csv)
    df_sum = pd.read_csv(summary_csv).sort_values("valid_python_versions", ascending=False)

    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    authors = df_sum["slug"].str.split("__").str[0]
    author_order = authors.tolist()
    legend_map = {
        slug: f"{author} ({int(n_ver)} v., pas~{gap:.1f})"
        for slug, author, n_ver, gap in zip(
            df_sum["slug"],
            authors,
            df_sum["valid_python_versions"],
            df_sum["avg_version_gap"],
        )
    }
    label_order = [legend_map[slug] for slug in df_sum["slug"]]
    df_ver["notebook_label"] = df_ver["slug"].map(legend_map)

    fig1_path = fig_dir / "01_evolution_code_et_modifications.png"
    fig2_path = fig_dir / "02_intervalles_temporels_et_cellules.png"
    fig3_path = fig_dir / "03_matrice_bibliotheques_ml.png"

    plot_code_evolution(df_ver, label_order, fig1_path)
    plot_intervals_and_cells(df_int, df_ver, author_order, fig2_path)
    plot_ml_library_matrix(build_library_trajectory_matrix(df_ver, author_order), fig3_path)

    print(f"Figures enregistrées dans {fig_dir} :")
    for path in (fig1_path, fig2_path, fig3_path):
        print(f"  - {path.name}")


if __name__ == "__main__":
    generate_plots()
