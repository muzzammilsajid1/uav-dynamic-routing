import csv

import pytest

from evaluation.check_artifact_integrity import _check_event_file, _check_file


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_route_integrity_requires_exact_cartesian_coverage(tmp_path):
    path = tmp_path / "routes.csv"
    _write_csv(
        path,
        [
            {"run_id": "run-1", "manifest_sha256": "abc"},
            {"run_id": "run-2", "manifest_sha256": "abc"},
        ],
    )
    report = _check_file(
        path,
        expected_manifest_hash="abc",
        expected_run_ids={"run-1", "run-2"},
    )
    assert report["rows"] == 2

    with pytest.raises(RuntimeError, match="Cartesian coverage"):
        _check_file(
            path,
            expected_manifest_hash="abc",
            expected_run_ids={"run-1", "run-3"},
        )


def test_event_integrity_matches_each_route_event_count(tmp_path):
    path = tmp_path / "events.csv"
    route_rows = [
        {"run_id": "run-1", "dynamic_event_count": "2"},
        {"run_id": "run-2", "dynamic_event_count": "0"},
    ]
    _write_csv(
        path,
        [
            {"run_id": "run-1", "event_index": "1"},
            {"run_id": "run-1", "event_index": "2"},
        ],
    )
    report = _check_event_file(path, route_rows=route_rows, classical=False)
    assert report["rows"] == 2

    _write_csv(path, [{"run_id": "run-1", "event_index": "1"}])
    with pytest.raises(RuntimeError, match="event coverage mismatch"):
        _check_event_file(path, route_rows=route_rows, classical=False)
