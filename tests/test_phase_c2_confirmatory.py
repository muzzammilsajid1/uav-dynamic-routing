import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from cloud.kaggle.phase_c2_kaggle_runner import KagglePhaseC2Runner
from scripts.run_phase_c2_confirmatory import (
    DEFAULT_SEEDS,
    aligned_target,
    inspect_seed,
    verify_bundle,
)


def test_confirmatory_protocol_constants():
    assert DEFAULT_SEEDS == (11, 22, 33, 44, 55)
    assert aligned_target(150000) == 151552


def test_explicit_seed_reaches_runner_and_generator(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_TEST_OUT_DIR", str(tmp_path))
    runner = KagglePhaseC2Runner("M2", 150000, device="cpu", seed=11)
    assert runner.seed == 11
    assert int(runner.generator.rng.get_state()[1][0]) == 11
    runner.train_env.close()


def _write_bundle(path: Path, payloads: dict[str, bytes]) -> None:
    lines = [
        f"{hashlib.sha256(content).hexdigest()}  {name}"
        for name, content in sorted(payloads.items())
    ]
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in payloads.items():
            archive.writestr(name, content)
        archive.writestr("inventory.txt", "\n".join(lines) + "\n")


def test_confirmatory_completed_seed_inspection(tmp_path):
    out_dir = tmp_path / "seed_011" / "artifacts"
    out_dir.mkdir(parents=True)
    target = aligned_target(150000)
    status = {"completed_interactions": target}
    provenance = {"model_type": "M2", "seed": 11, "git_commit": "abc123"}
    (out_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
    (out_dir / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (out_dir / f"evaluation_{target:06d}.json").write_text("{}", encoding="utf-8")
    (out_dir / f"model_{target:06d}.zip").write_bytes(b"model")
    _write_bundle(out_dir / "latest_checkpoint_bundle.zip", {"state.json": b"{}"})

    result = inspect_seed(out_dir, 11, 150000, "abc123")

    assert result["complete"] is True
    assert result["completed_interactions"] == target
    assert result["bundle_entries"] == 1
    json.dumps(result)


def test_completed_seed_allows_queue_only_commit_change(tmp_path, monkeypatch):
    import scripts.run_phase_c2_confirmatory as queue_module

    out_dir = tmp_path / "seed_011" / "artifacts"
    out_dir.mkdir(parents=True)
    target = aligned_target(150000)
    (out_dir / "status.json").write_text(
        json.dumps({"completed_interactions": target}), encoding="utf-8"
    )
    (out_dir / "provenance.json").write_text(
        json.dumps({"model_type": "M2", "seed": 11, "git_commit": "prior"}),
        encoding="utf-8",
    )
    (out_dir / f"evaluation_{target:06d}.json").write_text("{}", encoding="utf-8")
    (out_dir / f"model_{target:06d}.zip").write_bytes(b"model")
    _write_bundle(out_dir / "latest_checkpoint_bundle.zip", {"state.json": b"{}"})
    monkeypatch.setattr(
        queue_module, "core_training_sources_match", lambda provenance: (True, [])
    )

    result = inspect_seed(out_dir, 11, 150000, "repaired")

    assert result["complete"] is True
    assert result["prior_queue_only_commit"] is True
    json.dumps(result)


def test_bundle_verification_rejects_changed_payload(tmp_path):
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("state.json", b"changed")
        archive.writestr("inventory.txt", f"{'0' * 64}  state.json\n")

    with pytest.raises(RuntimeError, match="mismatched"):
        verify_bundle(bundle)
