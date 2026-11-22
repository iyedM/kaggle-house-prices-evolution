import textwrap

import pytest

from analyze import compute_sequence_diff, extract_imports_from_code
from collect import sample_versions_evenly
from visualize import (
    TRAJ_ABSENT,
    TRAJ_ADDED,
    TRAJ_CONSTANT,
    TRAJ_REMOVED,
    TRAJ_TRANSIENT,
    classify_library_trajectory,
    parse_ml_libraries,
)


def test_compute_sequence_diff() -> None:
    assert compute_sequence_diff(["a = 1", "b = 2"], ["a = 1", "b = 2"]) == {
        "lines_inserted": 0,
        "lines_deleted": 0,
        "lines_changed_between_sampled_snapshots": 0,
    }
    # Bloc remplacé asymétrique : 1 ligne remplacée par 2 + 1 ligne conservée
    diff = compute_sequence_diff(["import pandas as pd", "x = 1"], ["import pandas as pd", "x = 2", "y = 3"])
    assert diff == {
        "lines_inserted": 2,
        "lines_deleted": 1,
        "lines_changed_between_sampled_snapshots": 3,
    }


def test_extract_imports_from_code() -> None:
    cell_with_magics = textwrap.dedent("""
        %matplotlib inline
        !pip install torch
        import os
        import pandas as pd
        from sklearn.model_selection import KFold
        from automl_tables_wrapper import AutoMLTablesWrapper
    """)
    assert extract_imports_from_code(cell_with_magics) == {"pandas", "sklearn"}

    # Repli ligne par ligne lorsqu'une cellule ancienne contient de la syntaxe Python 2
    legacy_py2_cell = 'import xgboost as xgb\nprint "legacy python 2"\nfrom scipy import stats\n'
    assert extract_imports_from_code(legacy_py2_cell) == {"xgboost", "scipy"}


@pytest.mark.parametrize(
    ("presence", "expected"),
    [
        ([False, False, False], TRAJ_ABSENT),
        ([True, True, True], TRAJ_CONSTANT),
        ([False, False, True, True], TRAJ_ADDED),
        ([True, True, False, False], TRAJ_REMOVED),
        ([True, False, True], TRAJ_TRANSIENT),
        ([False, True, False], TRAJ_TRANSIENT),
    ],
)
def test_classify_library_trajectory(presence: list[bool], expected: int) -> None:
    assert classify_library_trajectory(presence) == expected


def test_sample_versions_evenly_and_parse_ml_libraries() -> None:
    versions = [{"version_number": i} for i in range(1, 21)]
    sampled = sample_versions_evenly(versions, max_count=8)
    assert len(sampled) == 8
    assert sampled[0]["version_number"] == 1
    assert sampled[-1]["version_number"] == 20

    with pytest.raises(ValueError):
        sample_versions_evenly(versions, max_count=2)

    assert parse_ml_libraries("") == set()
    assert parse_ml_libraries(float("nan")) == set()
    assert parse_ml_libraries("pandas, sklearn , numpy") == {"pandas", "sklearn", "numpy"}
