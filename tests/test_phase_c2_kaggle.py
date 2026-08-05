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
    
    # Run 1 step of M2 to create bundle
    out_dir = ROOT / "runs" / "uav_phase_c2_resume_test"
    if out_dir.exists(): shutil.rmtree(out_dir)
    
    cmd = [
        sys.executable, str(ROOT / "cloud/kaggle/phase_c2_kaggle_runner.py"),
        "--model", "M2",
        "--interactions", "10"
    ]
    subprocess.run(cmd, check=True)
    
    bundle_path = out_dir.parent / "latest_checkpoint_bundle.zip"
    assert bundle_path.exists()
    
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
    assert "EXPECTED_COMMIT =" in src
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
