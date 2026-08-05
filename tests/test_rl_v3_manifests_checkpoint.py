import json

from rl_v3.checkpointing import save_smoke_checkpoint
from rl_v3.scenario_generation import (
    build_training_generator_asset,
    generate_training_episode,
    generate_validation_manifest,
    verify_manifest_separation,
)


def _config():
    return json.loads(open("configs/rl_v3_phase_a.json", encoding="utf-8").read())


def test_training_generator_and_validation_manifest_are_separate():
    config = _config()
    asset = build_training_generator_asset(config)
    validation = generate_validation_manifest(config)
    separation = verify_manifest_separation(config, validation)
    assert asset["asset_type"] == "rl_v3_training_generator"
    assert len(validation["scenarios"]) == 36
    assert separation["passed"]
    assert not validation["final_test"]
    assert "final" not in validation["suite_id"]


def test_training_episode_generation_is_reproducible():
    config = _config()
    first = generate_training_episode(config, 12)
    second = generate_training_episode(config, 12)
    assert first == second
    assert first["scenario_id"] == "TRAIN-000000000012"


def test_checkpoint_smoke_contract(tmp_path):
    config = _config()
    status = save_smoke_checkpoint(
        run_dir=tmp_path / "run",
        config=config,
        policy_seed=123,
        generator_state={"episode_index": 1},
        manifest_hashes={"validation": "abc"},
        generator_hash="def",
        timesteps=2,
    )
    assert status["resume_verified"]
