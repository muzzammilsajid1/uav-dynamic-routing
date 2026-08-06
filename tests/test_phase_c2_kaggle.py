import json
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

def test_bundle_creation_and_resume():
    import sys
    import subprocess
    import shutil
    
    # Run 10 steps of M2 to create bundle
    out_dir = ROOT / "runs" / "uav_phase_c2_local_test"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    
    cmd = [
        sys.executable, str(ROOT / "cloud/kaggle/phase_c2_kaggle_runner.py"),
        "--model", "M2",
        "--interactions", "10"
    ]
    subprocess.run(cmd, check=True)
    
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
    res = subprocess.run(cmd, capture_output=True)
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
    if out_dir.exists():
        shutil.rmtree(out_dir)

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
    if out_dir.exists():
        shutil.rmtree(out_dir)

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
    if out_dir.exists():
        shutil.rmtree(out_dir)

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


def test_resume_from_in_root_bundle():
    """Resume must work when bundle is inside out_dir (not parent)."""
    import shutil
    import subprocess
    import sys

    actual_out = ROOT / "runs/uav_phase_c2_local_test"
    if actual_out.exists():
        shutil.rmtree(actual_out)

    subprocess.run(
        [sys.executable, "cloud/kaggle/phase_c2_kaggle_runner.py",
         "--model", "M1", "--interactions", "10", "--device", "cpu"],
        check=True, cwd=str(ROOT)
    )

    bundle_in_root = actual_out / "latest_checkpoint_bundle.zip"
    assert bundle_in_root.exists(), f"Bundle not found at in-root path: {bundle_in_root}"
    assert not (actual_out.parent / "latest_checkpoint_bundle.zip").exists(), \
        "Bundle should not exist in parent"

    subprocess.run(
        [sys.executable, "cloud/kaggle/phase_c2_kaggle_runner.py",
         "--model", "M1", "--interactions", "20", "--device", "cpu",
         "--resume", "--bundle-path", str(bundle_in_root)],
        check=True, cwd=str(ROOT)
    )

    if actual_out.exists():
        shutil.rmtree(actual_out)


def test_no_parent_dir_bundles():
    """No checkpoint_bundle_*.zip should exist in parent of out_dir after a run."""
    import shutil
    import subprocess
    import sys

    actual_out = ROOT / "runs/uav_phase_c2_local_test"
    if actual_out.exists():
        shutil.rmtree(actual_out)

    subprocess.run(
        [sys.executable, "cloud/kaggle/phase_c2_kaggle_runner.py",
         "--model", "M2", "--interactions", "10", "--device", "cpu"],
        check=True, cwd=str(ROOT)
    )
    parent = actual_out.parent

    leaked = list(parent.glob("checkpoint_bundle_*.zip")) + \
             list(parent.glob("latest_checkpoint_bundle.zip"))
    assert not leaked, f"Bundles leaked to parent dir: {leaked}"

    if actual_out.exists():
        shutil.rmtree(actual_out)


def test_runner_no_parent_zip_leakage():
    """Verify the runner source does not reference out_dir.parent for bundles."""
    runner_src = (ROOT / "cloud/kaggle/phase_c2_kaggle_runner.py").read_text()
    assert 'out_dir).parent / "latest_checkpoint_bundle' not in runner_src
    assert 'out_dir).parent / f"checkpoint_bundle_' not in runner_src
