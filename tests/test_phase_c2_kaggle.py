import json

def local_runner_env(out_dir=None):
    import os
    env = os.environ.copy()
    env.pop("KAGGLE_KERNEL_RUN_TYPE", None)
    if out_dir:
        env["KAGGLE_TEST_OUT_DIR"] = str(out_dir)
    return env

import pytest
import os
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_notebook_json_validity():
    nb_path = ROOT / "cloud/kaggle/phase_c2_kaggle.ipynb"
    assert nb_path.exists()
    with open(nb_path) as f:
        nb = json.load(f)
    assert nb["nbformat"] == 4
    
    # Check for placeholders
    text = json.dumps(nb).lower()
    for word in ["todo", "placeholder", "implement here", "dummy", "mocked success", "hard-coded successful output", "tiny benchmark logic"]:
        assert word not in text, f"Found placeholder '{word}' in notebook"

def test_notebook_code_cells_parse():
    nb_path = ROOT / "cloud/kaggle/phase_c2_kaggle.ipynb"
    with open(nb_path) as f:
        nb = json.load(f)
        
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            # Remove magic commands
            src = "\\n".join([line for line in src.split("\\n") if not line.strip().startswith(("!", "%"))])
            try:
                ast.parse(src)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in notebook cell: {e}\\n{src}")

def test_no_windows_paths():
    runner_path = ROOT / "cloud/kaggle/phase_c2_kaggle_runner.py"
    with open(runner_path) as f:
        src = f.read()
    assert "\\\\" not in src.replace("\\\\n", ""), "Found backslashes suggesting hardcoded Windows paths"
    assert "C:" not in src, "Found C: drive path"
    
def test_m1_architecture():
    import json

    from rl_v3.run_phase_c2 import PhaseC2Runner
    from sb3_contrib import MaskablePPO
    
    runner = PhaseC2Runner(ROOT / "configs/rl_v3_phase_c2.json", out_dir=str(ROOT / "runs/test_arch"), model_type="M1")
    model = runner.model
    
    # M1 must use global-local feature extractor
    assert (model.policy.features_extractor.__class__.__name__ == "PhaseBMaskableActorCriticPolicy" or
            model.policy.features_extractor.__class__.__name__ == "PhaseBFeatureExtractor")
           
def test_m2_architecture():
    import json

    from rl_v3.run_phase_c2 import PhaseC2Runner
    from sb3_contrib import MaskablePPO
    
    runner = PhaseC2Runner(ROOT / "configs/rl_v3_phase_c2.json", out_dir=str(ROOT / "runs/test_arch"), model_type="M2")
    model = runner.model
    
    # M2 must use MultiInputPolicy and CombinedExtractor (SB3 default for Dict)
    assert "MultiInput" in str(model.policy.__class__)
    assert "CombinedExtractor" in str(model.policy.features_extractor.__class__)
    
def test_preflight_command_exists():
    # We just run the preflight from a subprocess
    import subprocess
    import sys
    out_dir = ROOT / "runs" / "uav_phase_c2_preflight_test"
    res = subprocess.run([sys.executable, "-m", "rl_v3.run_phase_c2", "preflight", str(out_dir)], capture_output=True)
    assert res.returncode == 0
    assert (out_dir / "preflight_verification.json").exists()
    with open(out_dir / "preflight_verification.json") as f:
        v = json.load(f)
    assert v["status"] == "PASS"
    assert v["oracle_success_sizes"] == [15, 30, 50, 100]

def test_bundle_creation_and_resume(tmp_path):
    import sys
    import subprocess
    import shutil
    
    # Run 10 steps of M2 to create bundle
    out_dir = tmp_path
    
    
    cmd = [
        sys.executable, str(ROOT / "cloud/kaggle/phase_c2_kaggle_runner.py"),
        "--model", "M2",
        "--interactions", "10"
    ]
    subprocess.run(cmd, check=True, env=local_runner_env(out_dir))
    
    # Bundle must be INSIDE out_dir (v4 fix)
    bundle_path = out_dir / "latest_checkpoint_bundle.zip"
    assert bundle_path.exists(), f"Bundle not found at {bundle_path}"
    
    # Resume it for 10 more steps
    cmd = [
        sys.executable, str(ROOT / "cloud/kaggle/phase_c2_kaggle_runner.py"),
        "--model", "M2",
        "--interactions", "20",
        "--resume",
        "--bundle-path", str(bundle_path)
    ]
    res = subprocess.run(cmd, capture_output=True, env=local_runner_env(out_dir))
    assert res.returncode == 0
    
def test_git_token_hygiene():
    nb_path = ROOT / "cloud/kaggle/phase_c2_kaggle.ipynb"
    with open(nb_path) as f:
        src = f.read()
    assert "set-url" in src and "origin" in src, "Notebook does not clean origin URL"
    
def test_hashes_exist_in_notebook():
    nb_path = ROOT / "cloud/kaggle/phase_c2_kaggle.ipynb"
    with open(nb_path) as f:
        src = f.read()
    assert "EXPECTED_TAG =" in src
    assert "HASH_VALIDATION =" in src
    assert "HASH_TRAIN_GEN =" in src
    assert "HASH_CONFIG =" in src
    assert "HASH_KAGGLE_RUNNER =" in src

def test_m2_no_map_tensors():
    import json

    from rl_v3.run_phase_c2 import PhaseC2Runner
    
    runner = PhaseC2Runner(ROOT / "configs/rl_v3_phase_c2.json", out_dir=str(ROOT / "runs/test_arch"), model_type="M2")
    
    obs, _ = runner.train_env.reset()
    assert "global_map" not in obs, "M2 received global_map!"
    assert "local_map" not in obs, "M2 received local_map!"
    assert "scalars" in obs, "M2 did not receive scalars!"

def test_output_root_enforcement():
    runner_path = ROOT / "cloud/kaggle/phase_c2_kaggle_runner.py"
    with open(runner_path) as f:
        src = f.read()
    
    assert "/kaggle/working/uav_phase_c2" in src
    assert "rl_v3_phase_c2_" in src and "_raw_artifacts" in src

def test_m2_scalar_contract():
    import numpy as np
    from rl_v3.run_phase_c2 import PhaseC2Runner
    
    runner = PhaseC2Runner(ROOT / "configs/rl_v3_phase_c2.json", out_dir=str(ROOT / "runs/test_arch2"), model_type="M2")
    env = runner.train_env
    
    obs, info = env.reset(seed=42)
    scalars = obs["scalars"]
    
    assert scalars.shape == (4,), f"Expected 4 scalars, got {scalars.shape}"
    assert scalars.dtype == np.float32, f"Expected float32, got {scalars.dtype}"
    assert "local_map" not in obs
    assert "global_map" not in obs
    assert env.observation_space.contains(obs), "Observation space does not contain the observation"
    
    invalid_obs = {"scalars": np.array([0, 0, -1, 0], dtype=np.float32)}
    assert not env.observation_space.contains(invalid_obs), "Observation space incorrectly contains invalid observation"


# ---------------------------------------------------------------------------
# v3: device propagation tests
# ---------------------------------------------------------------------------

def test_device_propagation():
    import numpy as np
    from rl_v3.run_phase_c2 import PhaseC2Runner
    
    runner = PhaseC2Runner(ROOT / "configs/rl_v3_phase_c2.json", out_dir=str(ROOT / "runs/test_dev"), model_type="M2", device="cpu")
    assert runner.device == "cpu"
    assert runner.model.device.type == "cpu"
    
    try:
        PhaseC2Runner(ROOT / "configs/rl_v3_phase_c2.json", out_dir=str(ROOT / "runs/test_dev2"), model_type="M2", device="invalid_dev")
        assert False, "Should reject invalid device"
    except ValueError:
        pass

def test_kaggle_runner_device():
    import subprocess
    import sys
    
    res = subprocess.run([sys.executable, str(ROOT / "cloud/kaggle/phase_c2_kaggle_runner.py"), "--help"], capture_output=True, text=True)
    assert "--device" in res.stdout

def test_inventory_path():
    with open(ROOT / "cloud/kaggle/phase_c2_kaggle.ipynb") as f:
        src = f.read()
    assert "/kaggle/working/final_inventory.txt" not in src
    assert "/kaggle/working/uav_phase_c2/final_inventory.txt" in src

def test_pytest_in_notebook():
    with open(ROOT / "cloud/kaggle/phase_c2_kaggle.ipynb") as f:
        src = f.read()
    assert "pytest_kaggle.log" in src
    assert "pytest" in src

def test_preflight_command():
    import subprocess
    import sys
    
    out_dir = ROOT / "runs/test_preflight_cli"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    res = subprocess.run([sys.executable, "-m", "rl_v3.run_phase_c2", "preflight", str(out_dir)], check=True)
    assert res.returncode == 0


# ---------------------------------------------------------------------------
# v4: bundle-path regression tests
# ---------------------------------------------------------------------------

def test_bundle_inside_out_dir_m1():
    """latest_checkpoint_bundle.zip must be inside out_dir, not in parent."""
    import shutil
    out_dir = ROOT / "runs/test_bundle_path_m1"
    

    from cloud.kaggle.phase_c2_kaggle_runner import KagglePhaseC2Runner
    from rl_v3.run_phase_c2 import PhaseC2Runner
    config_path = ROOT / "configs" / "rl_v3_phase_c2.json"
    runner = KagglePhaseC2Runner.__new__(KagglePhaseC2Runner)
    PhaseC2Runner.__init__(runner, config_path=config_path, out_dir=str(out_dir),
                            model_type="M1", resume=False, device="cpu")
    runner.max_interactions = 10
    runner.bundle_path = None
    runner.run(10)

    latest = out_dir / "latest_checkpoint_bundle.zip"
    assert latest.exists(), f"Bundle not found inside out_dir: {latest}"

    parent_bundle = out_dir.parent / "latest_checkpoint_bundle.zip"
    assert not parent_bundle.exists(), f"Bundle leaked to parent: {parent_bundle}"

    shutil.rmtree(out_dir)


def test_bundle_inside_out_dir_m2():
    """Same check for M2."""
    import shutil
    out_dir = ROOT / "runs/test_bundle_path_m2"
    

    from cloud.kaggle.phase_c2_kaggle_runner import KagglePhaseC2Runner
    from rl_v3.run_phase_c2 import PhaseC2Runner
    config_path = ROOT / "configs" / "rl_v3_phase_c2.json"
    runner = KagglePhaseC2Runner.__new__(KagglePhaseC2Runner)
    PhaseC2Runner.__init__(runner, config_path=config_path, out_dir=str(out_dir),
                            model_type="M2", resume=False, device="cpu")
    runner.max_interactions = 10
    runner.bundle_path = None
    runner.run(10)

    latest = out_dir / "latest_checkpoint_bundle.zip"
    assert latest.exists(), f"Bundle not found inside out_dir: {latest}"

    parent_bundle = out_dir.parent / "latest_checkpoint_bundle.zip"
    assert not parent_bundle.exists(), f"Bundle leaked to parent: {parent_bundle}"

    shutil.rmtree(out_dir)


def test_no_nested_zip_in_bundle():
    """checkpoint bundle must contain no nested .zip file."""
    import zipfile
    import shutil
    out_dir = ROOT / "runs/test_no_nested_zip"
    

    from cloud.kaggle.phase_c2_kaggle_runner import KagglePhaseC2Runner
    from rl_v3.run_phase_c2 import PhaseC2Runner
    config_path = ROOT / "configs" / "rl_v3_phase_c2.json"
    runner = KagglePhaseC2Runner.__new__(KagglePhaseC2Runner)
    PhaseC2Runner.__init__(runner, config_path=config_path, out_dir=str(out_dir),
                            model_type="M1", resume=False, device="cpu")
    runner.max_interactions = 10
    runner.bundle_path = None
    runner.run(10)

    latest = out_dir / "latest_checkpoint_bundle.zip"
    assert latest.exists()
    with zipfile.ZipFile(latest, "r") as zf:
        names = zf.namelist()
    nested_zips = [n for n in names if "bundle" in n or "_raw_artifacts" in n]
    assert not nested_zips, f"Nested zips found in bundle: {nested_zips}"

    shutil.rmtree(out_dir)


def test_source_hash_is_line_ending_independent(tmp_path):
    from cloud.kaggle.phase_c2_kaggle_runner import (
        _legacy_source_hash_candidates,
        hash_file,
        hash_source_file,
    )

    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes(b"alpha\nbeta\n")
    crlf.write_bytes(b"alpha\r\nbeta\r\n")

    assert hash_source_file(lf) == hash_source_file(crlf)
    assert hash_file(crlf) in _legacy_source_hash_candidates(lf)


def test_final_archives_exclude_prior_archives_and_are_repeatable(tmp_path):
    import zipfile

    from cloud.kaggle.phase_c2_kaggle_runner import create_final_archives

    out_dir = tmp_path / "uav_phase_c2"
    out_dir.mkdir()
    (out_dir / "evaluation_002048.json").write_text("{}", encoding="utf-8")
    (out_dir / "model_002048.zip").write_bytes(b"model archive")
    (out_dir / "latest_checkpoint_bundle.zip").write_bytes(b"bundle")
    (out_dir / "checkpoint_bundle_002048.zip").write_bytes(b"timestamped bundle")
    (out_dir / "rl_v3_phase_c2_M2_raw_artifacts.zip").write_bytes(b"old raw")

    raw_archive, complete_archive = create_final_archives(out_dir, "M2")
    raw_archive, complete_archive = create_final_archives(out_dir, "M2")

    expected = {"evaluation_002048.json", "model_002048.zip", "final_inventory.txt"}
    for archive in (raw_archive, complete_archive):
        with zipfile.ZipFile(archive) as handle:
            assert set(handle.namelist()) == expected
            inventory = handle.read("final_inventory.txt").decode("utf-8")
            assert "evaluation_002048.json" in inventory
            assert "model_002048.zip" in inventory
            assert "checkpoint_bundle" not in inventory
            assert "raw_artifacts" not in inventory


def test_resume_from_in_root_bundle(tmp_path):
    """Resume must work when bundle is inside out_dir (not parent)."""
    import shutil
    import subprocess
    import sys

    actual_out = tmp_path / "uav_phase_c2_local_test"
    

    subprocess.run(
        [sys.executable, "cloud/kaggle/phase_c2_kaggle_runner.py",
         "--model", "M1", "--interactions", "10", "--device", "cpu"],
        check=True, cwd=str(ROOT), env=local_runner_env(actual_out)
    )

    bundle_in_root = actual_out / "latest_checkpoint_bundle.zip"
    assert bundle_in_root.exists(), f"Bundle not found at in-root path: {bundle_in_root}"
    assert not (actual_out.parent / "latest_checkpoint_bundle.zip").exists(), \
        "Bundle should not exist in parent"

    subprocess.run(
        [sys.executable, "cloud/kaggle/phase_c2_kaggle_runner.py",
         "--model", "M1", "--interactions", "20", "--device", "cpu",
         "--resume", "--bundle-path", str(bundle_in_root)],
        check=True, cwd=str(ROOT), env=local_runner_env(actual_out)
    )

    


def test_no_parent_dir_bundles(tmp_path):
    """No checkpoint_bundle_*.zip should exist in parent of out_dir after a run."""
    import shutil
    import subprocess
    import sys

    actual_out = tmp_path / "uav_phase_c2_local_test"
    

    subprocess.run(
        [sys.executable, "cloud/kaggle/phase_c2_kaggle_runner.py",
         "--model", "M2", "--interactions", "10", "--device", "cpu"],
        check=True, cwd=str(ROOT), env=local_runner_env(actual_out)
    )
    parent = actual_out.parent

    leaked = list(parent.glob("checkpoint_bundle_*.zip")) + \
             list(parent.glob("latest_checkpoint_bundle.zip"))
    assert not leaked, f"Bundles leaked to parent dir: {leaked}"

    


def test_runner_no_parent_zip_leakage():
    """Verify the runner source does not reference out_dir.parent for bundles."""
    runner_src = (ROOT / "cloud/kaggle/phase_c2_kaggle_runner.py").read_text()
    assert 'out_dir).parent / "latest_checkpoint_bundle' not in runner_src
    assert 'out_dir).parent / f"checkpoint_bundle_' not in runner_src
import sys
from pathlib import Path
import json

import subprocess
import hashlib

ROOT = Path(__file__).parent.parent

def test_canonical_hashes():
    nb_path = ROOT / "cloud/kaggle/phase_c2_kaggle.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    
    hashes = {}
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            for line in cell["source"]:
                if line.startswith("HASH_"):
                    parts = line.split("=")
                    if len(parts) == 2:
                        name = parts[0].strip()
                        val = parts[1].strip().strip('",\\n')
                        hashes[name] = val

    expected_vars = [
        "HASH_VALIDATION", "HASH_TRAIN_GEN", "HASH_CONFIG", 
        "HASH_REWARD", "HASH_OBSERVATION",
        "HASH_C2_ENV", "HASH_C2_RUNNER", "HASH_C0_ENV", "HASH_ACTION_MASK",
        "HASH_KAGGLE_RUNNER"
    ]
    for var in expected_vars:
        assert var in hashes, f"Missing {var} in notebook"

    file_mapping = {
        "HASH_VALIDATION": "evaluation/manifests/rl_v3_phase_c2_validation.json",
        "HASH_TRAIN_GEN": "evaluation/manifests/rl_v3_phase_c2_train_generator.json",
        "HASH_CONFIG": "configs/rl_v3_phase_c2.json",
        "HASH_REWARD": "tools/verification/r2_pb_wrapper.py",
        "HASH_OBSERVATION": "rl_v3/observations.py",
        "HASH_C2_ENV": "rl_v3/phase_c2_env.py",
        "HASH_C2_RUNNER": "rl_v3/run_phase_c2.py",
        "HASH_C0_ENV": "rl_v3/phase_c0_env.py",
        "HASH_ACTION_MASK": "rl_v3/action_masking.py",
        "HASH_KAGGLE_RUNNER": "cloud/kaggle/phase_c2_kaggle_runner.py",
    }
    
    for var, fpath in file_mapping.items():
        canonical_bytes = (ROOT / fpath).read_bytes().replace(b"\r\n", b"\n")
        canonical_hash = hashlib.sha256(canonical_bytes).hexdigest()
        assert canonical_hash == hashes[var], f"Hash mismatch for {var} ({fpath}): expected {canonical_hash}, got {hashes[var]}"



def test_kaggle_environment_regression(tmp_path):
    import subprocess
    import sys
    import os
    
    os.environ["KAGGLE_KERNEL_RUN_TYPE"] = "Interactive"
    
    # Create the simulated production root
    prod_root = tmp_path / "simulated_working"
    prod_root.mkdir()
    prod_dir = prod_root / "uav_phase_c2"
    prod_dir.mkdir()
    
    # Create sentinel log file
    sentinel_file = prod_dir / "pytest_kaggle.log"
    sentinel_file.write_text("Test Kaggle Log\n")
    
    # Set the test prod dir variable
    os.environ["KAGGLE_TEST_PROD_DIR"] = str(prod_root)
    
    try:
        out_dir = tmp_path / "uav_phase_c2_out"
        
        env = local_runner_env(out_dir)
        env["KAGGLE_TEST_PROD_DIR"] = str(prod_root)
        env["KAGGLE_KERNEL_RUN_TYPE"] = "Interactive"
        
        cmd = [
            sys.executable, str(ROOT / "cloud/kaggle/phase_c2_kaggle_runner.py"),
            "--model", "M2",
            "--interactions", "10"
        ]
        subprocess.run(cmd, check=True, env=env)
        
        assert out_dir.exists(), "Output not in local test directory"
        bundle_path = out_dir / "latest_checkpoint_bundle.zip"
        assert bundle_path.exists(), "Bundle not found in isolated directory"
        
        # Verify production directory is untouched but sentinel log still exists
        assert sentinel_file.exists()
        assert sentinel_file.read_text() == "Test Kaggle Log\n"
        
        assert not (prod_dir / "latest_checkpoint_bundle.zip").exists()
        assert not list(prod_dir.glob("checkpoint_bundle_*.zip"))
        assert not list(prod_dir.glob("rl_v3_phase_c2_*_raw_artifacts.zip"))
        assert not list(prod_dir.glob("model_*.zip"))
        assert not (prod_dir / "provenance.json").exists()
        assert not (prod_dir / "status.json").exists()
        
        resume_cmd = [
            sys.executable, str(ROOT / "cloud/kaggle/phase_c2_kaggle_runner.py"),
            "--model", "M2",
            "--interactions", "20",
            "--resume",
            "--bundle-path", str(bundle_path)
        ]
        res = subprocess.run(resume_cmd, capture_output=True, env=env)
        assert res.returncode == 0
        
    finally:
        os.environ.pop("KAGGLE_KERNEL_RUN_TYPE", None)
        os.environ.pop("KAGGLE_TEST_PROD_DIR", None)

def test_notebook_contract():
    import json
    path = ROOT / 'cloud/kaggle/phase_c2_kaggle.ipynb'
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    full_src = ""
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            full_src += "".join(cell['source']) + "\n"
            
    assert 'Path("/kaggle/working/uav_phase_c2").mkdir(' in full_src
    assert 'open("/kaggle/working/uav_phase_c2/pytest_kaggle.log"' in full_src
    # Check that pytest is invoked before preflight
    pytest_idx = full_src.find('["pytest"')
    if pytest_idx == -1:
        pytest_idx = full_src.find('"-m", "pytest"')
    preflight_idx = full_src.find('"preflight"')
    assert pytest_idx != -1
    assert preflight_idx != -1
    assert pytest_idx < preflight_idx
    
    assert "pytest_kaggle.log" in full_src
    assert 'os.environ["KAGGLE_TEST_OUT_DIR"]' not in full_src
    assert 'os.environ.setdefault("KAGGLE_TEST_OUT_DIR"' not in full_src
    assert 'KAGGLE_TEST_OUT_DIR' not in full_src
    assert 'if not Path("/kaggle/working/uav_phase_c2/pytest_kaggle.log").exists():' in full_src
    assert 'KAGGLE SETTINGS: enable Persistence' in full_src
    assert 'WAIT UNTIL phase_c2_{MODEL_TO_RUN}_COMPLETE.zip HAS BEEN DOWNLOADED' in full_src
    assert '--basetemp=/kaggle/working/pytest-phase-c2-temp' in full_src
    assert '--basetemp=/kaggle/working/uav_phase_c2' not in full_src
    assert '=== ISOLATED END-TO-END RELEASE SMOKE (2,048 interactions) ===' in full_src
    assert '"--smoke"' in full_src
    assert 'evaluation_002048.json' in full_src
    assert 'checkpoint_is_ppo_update_aligned' in full_src
    assert 'invalid_action_count' in full_src
def test_durability_and_complete_backup(tmp_path):
    out_dir = tmp_path / "uav_phase_c2"
    out_dir.mkdir(parents=True)
    
    env = os.environ.copy()
    env["KAGGLE_TEST_OUT_DIR"] = str(out_dir)
    env["PYTHONPATH"] = str(ROOT)
    
    # Run a short train session via import to force checkpoints at 10 and 20
    sys.path.insert(0, str(ROOT))
    from cloud.kaggle.phase_c2_kaggle_runner import KagglePhaseC2Runner
    os.environ["KAGGLE_TEST_OUT_DIR"] = str(out_dir)
    os.environ.setdefault("KAGGLE_TEST_PROD_DIR", str(tmp_path))
    
    runner = KagglePhaseC2Runner("M2", 20, device="cpu")
    
    # We fake the history keys so it doesn't crash
    runner.history[10] = {"success_rate": 0.5}
    runner.history[20] = {"success_rate": 0.5}
    runner.evaluate_and_save(10)
    runner.evaluate_and_save(20)
    
    # also call the archiving logic that is usually at the bottom of the script
    with open(out_dir / "final_inventory.txt", "w") as f:
        f.write("test")

    import shutil
    import tempfile as _tf
    complete_archive_base = out_dir.parent / "phase_c2_M2_COMPLETE"
    with _tf.TemporaryDirectory() as _stage2:
        _stage_path2 = Path(_stage2)
        _complete_base = _stage_path2 / complete_archive_base.name
        shutil.make_archive(str(_complete_base), 'zip', str(out_dir))
        final_complete_archive = complete_archive_base.with_suffix(".zip")
        shutil.move(str(_complete_base) + ".zip", str(final_complete_archive))
    
    # Verify both timestamped bundles exist (10 and 20 interactions)
    bundle_10k = out_dir / "checkpoint_bundle_000010.zip"
    bundle_20k = out_dir / "checkpoint_bundle_000020.zip"
    latest_bundle = out_dir / "latest_checkpoint_bundle.zip"
    
    assert bundle_10k.exists()
    assert bundle_20k.exists()
    assert latest_bundle.exists()
    
    # latest should match the newest (20)
    import hashlib
    def get_hash(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    assert get_hash(latest_bundle) == get_hash(bundle_20k)
    
    # Verify complete backup
    complete_zip = tmp_path / "phase_c2_M2_COMPLETE.zip"
    assert complete_zip.exists()
    
    # Ensure it's not recursive inside itself
    import zipfile
    with zipfile.ZipFile(complete_zip, 'r') as zf:
        names = zf.namelist()
        assert complete_zip.name not in names
        assert "final_inventory.txt" in names
        assert bundle_10k.name in names
        assert latest_bundle.name in names

