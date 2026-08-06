import os
import sys
import shutil
import json
import logging
import zipfile
import hashlib
from pathlib import Path
from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from rl_v3.run_phase_c2 import PhaseC2Runner

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

EXPECTED_TAG = "rl-v3-c2-kaggle-v4"

def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class KagglePhaseC2Runner(PhaseC2Runner):
    def __init__(self, model_type, max_interactions, resume=False, bundle_path=None, device="auto"):
        is_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
        test_out = os.environ.get("KAGGLE_TEST_OUT_DIR")
        if test_out:
            out_dir = test_out
        else:
            out_dir = "/kaggle/working/uav_phase_c2" if is_kaggle else str(ROOT / "runs" / "uav_phase_c2_local_test")
        
        config_path = ROOT / "configs" / "rl_v3_phase_c2.json"
        
        self.bundle_path = bundle_path
        if resume:
            self._handle_resume(bundle_path, out_dir, model_type)
            
        super().__init__(config_path=config_path, out_dir=out_dir, model_type=model_type, resume=resume, device=device)
        self.max_interactions = max_interactions
        
        if resume:
            self.model = MaskablePPO.load(self.resume_model_path, env=self.train_env, device=self.device)
            # Override num_timesteps
            self.model.num_timesteps = self.resume_ts
            self.history = self.resume_history
            self.generator.set_state(self.resume_gen_state)
            logger.info(f"Successfully resumed at timestep {self.resume_ts} on device {self.device}")

    def _handle_resume(self, bundle_path, out_dir, expected_model):
        if not bundle_path or not Path(bundle_path).exists():
            raise FileNotFoundError(f"Resume bundle not found at {bundle_path}")
            
        logger.info(f"Extracting resume bundle from {bundle_path}")
        extract_dir = Path(out_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle_path, 'r') as zf:
            zf.extractall(extract_dir)
            
        # Verify Inventory
        with open(extract_dir / "inventory.txt", "r") as f:
            for line in f:
                if not line.strip(): continue
                expected_hash, fname = line.strip().split("  ")
                fname = fname.strip()
                if fname.startswith("./"): fname = fname[2:]
                fpath = extract_dir / fname
                if not fpath.exists():
                    raise RuntimeError(f"Missing file from bundle: {fname}")
                h = hash_file(fpath)
                if h != expected_hash:
                    raise RuntimeError(f"Hash mismatch in bundle for {fname}: expected {expected_hash}, got {h}")
                    
        # Load Provenance
        with open(extract_dir / "provenance.json", "r") as f:
            prov = json.load(f)
            
        if prov["model_type"] != expected_model:
            raise ValueError(f"Bundle model type {prov['model_type']} != {expected_model}")
            
        # Verify hashes
        curr_cfg_hash = hash_file(ROOT / "configs" / "rl_v3_phase_c2.json")
        curr_val_hash = hash_file(ROOT / "evaluation/manifests/rl_v3_phase_c2_validation.json")
        
        if prov["hashes"]["config"] != curr_cfg_hash:
            raise ValueError("Config hash mismatch on resume")
        if prov["hashes"]["validation_manifest"] != curr_val_hash:
            raise ValueError("Validation manifest mismatch on resume")
            
        self.resume_ts = prov["completed_interactions"]
        self.resume_model_path = extract_dir / f"model_{self.resume_ts:06d}.zip"
        
        with open(extract_dir / f"generator_{self.resume_ts:06d}.json", "r") as f:
            self.resume_gen_state = json.load(f)
            
        with open(extract_dir / "status.json", "r") as f:
            self.resume_history = json.load(f)["history"]
            
        # Also clean up string keys back to int if needed
        self.resume_history = {int(k): v for k, v in self.resume_history.items()}

    def evaluate_and_save(self, ts):
        super().evaluate_and_save(ts)
        
        logger.info("\n==============================================")
        logger.info(f" CHECKPOINT SAVED: {ts} interactions")
        logger.info(f" Validation Success Rate: {self.history[ts]['success_rate']*100:.1f}%")
        logger.info(f" Output Directory: {self.out_dir}")
        logger.info("==============================================\n")
        
        # Provenance
        prov = {
            "model_type": self.model_type,
            "device": self.device,
            "completed_interactions": ts,
            "hashes": {
                "config": hash_file(ROOT / "configs" / "rl_v3_phase_c2.json"),
                "validation_manifest": hash_file(ROOT / "evaluation/manifests/rl_v3_phase_c2_validation.json"),
                "train_generator": hash_file(ROOT / "evaluation/manifests/rl_v3_phase_c2_train_generator.json")
            }
        }
        with open(self.out_dir / "provenance.json", "w") as f:
            json.dump(prov, f, indent=2)
            
        # Atomic Bundle Creation — bundles live inside self.out_dir, never in parent.
        # Build from an explicit whitelist of checkpoint-state files only.
        # Excluded from bundles: any *.zip file (avoids nested archives and raw-artifact
        # archives), final_inventory.txt (only written at run end), and
        # partial/temporary files.
        import tempfile

        _BUNDLE_EXCLUDES = {
            "latest_checkpoint_bundle.zip",
            "final_inventory.txt",
        }

        def _is_excluded(fpath: Path) -> bool:
            """Return True if this file must not appear in the checkpoint bundle."""
            name = fpath.name
            if name in _BUNDLE_EXCLUDES:
                return True
            # Exclude timestamped checkpoint bundles and raw-artifact archives.
            # Include model checkpoint zips (model_<ts>.zip) — they are needed for resume.
            if name.startswith("checkpoint_bundle_") and name.endswith(".zip"):
                return True
            if "_raw_artifacts" in name and name.endswith(".zip"):
                return True
            return False

        # 1. Build inventory from whitelisted files.
        inventory_lines = []
        checkpoint_files = []
        for fpath in sorted(self.out_dir.iterdir()):
            if not fpath.is_file():
                continue
            if _is_excluded(fpath):
                continue
            if fpath.name == "inventory.txt":
                continue
            h = hash_file(fpath)
            inventory_lines.append(f"{h}  {fpath.name}")
            checkpoint_files.append(fpath)

        with open(self.out_dir / "inventory.txt", "w") as f:
            f.write("\n".join(inventory_lines) + "\n")
        checkpoint_files.append(self.out_dir / "inventory.txt")

        # 2. Stage whitelisted files in a temp directory and zip from there.
        #    The archive base MUST be outside the staging dir to avoid the zip
        #    including itself (make_archive writes <base>.zip before it finishes).
        with tempfile.TemporaryDirectory() as staging_dir:
            staging = Path(staging_dir)
            for src in checkpoint_files:
                shutil.copy2(str(src), str(staging / src.name))

            # Write zip base to a sibling temp file, not inside staging_dir.
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as _tf:
                tmp_zip_path = Path(_tf.name)
            tmp_zip_base = str(tmp_zip_path)[:-4]  # strip .zip for make_archive
            shutil.make_archive(tmp_zip_base, "zip", str(staging))
            # tmp_zip_base + ".zip" now exists outside staging_dir.

        # 3. Move into self.out_dir — both paths are inside the output root.
        bundle_path = self.out_dir / "latest_checkpoint_bundle.zip"
        shutil.move(str(tmp_zip_base) + ".zip", str(bundle_path))

        timestamped_bundle = self.out_dir / f"checkpoint_bundle_{ts:06d}.zip"
        shutil.copy2(str(bundle_path), str(timestamped_bundle))

        logger.info(f"Bundle archived to: {bundle_path}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="M1")
    parser.add_argument("--interactions", type=int, default=150000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bundle-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()
    
    logger.info(f"\n Launching Phase C2 Kaggle Runner")
    logger.info(f"Model: {args.model}")
    logger.info(f"Interactions: {args.interactions}")
    logger.info(f"Resume: {args.resume}")
    logger.info(f"Device: {args.device}\n")
    
    runner = KagglePhaseC2Runner(args.model, args.interactions, args.resume, args.bundle_path, args.device)
    runner.run(args.interactions)
    
    is_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
    test_out = os.environ.get("KAGGLE_TEST_OUT_DIR")
    if test_out:
        out_dir = Path(test_out)
    else:
        out_dir = Path("/kaggle/working/uav_phase_c2" if is_kaggle else str(ROOT / "runs" / "uav_phase_c2_local_test"))

    # Stage the raw-artifact archive in a tempdir, then move into out_dir.
    # This prevents shutil.make_archive from recursing into out_dir while
    # the archive is being created inside it.
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _stage:
        _stage_path = Path(_stage)
        _archive_base = _stage_path / f"rl_v3_phase_c2_{args.model}_raw_artifacts"
        shutil.make_archive(str(_archive_base), 'zip', str(out_dir))
        final_archive = out_dir / f"rl_v3_phase_c2_{args.model}_raw_artifacts.zip"
        shutil.move(str(_archive_base) + ".zip", str(final_archive))

    # Final Inventory — skip all .zip files (bundles + raw-artifact archives).
    lines = []
    for root, dirs, files in os.walk(str(out_dir)):
        for file in files:
            p = Path(root) / file
            if p.suffix == ".zip":
                continue
            h = hash_file(p)
            rel = p.relative_to(out_dir)
            lines.append(f"{h}  {rel}  {p.stat().st_size} bytes")

    with open(out_dir / "final_inventory.txt", "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Final artifacts archived to: {final_archive}")
