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

EXPECTED_COMMIT = "8531041"

def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class KagglePhaseC2Runner(PhaseC2Runner):
    def __init__(self, model_type, max_interactions, resume=False, bundle_path=None):
        is_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
        out_dir = "/kaggle/working/uav_phase_c2" if is_kaggle else str(ROOT / "runs" / "uav_phase_c2_local_test")
        
        config_path = ROOT / "configs" / "rl_v3_phase_c2.json"
        
        self.bundle_path = bundle_path
        if resume:
            self._handle_resume(bundle_path, out_dir, model_type)
            
        super().__init__(config_path=config_path, out_dir=out_dir, model_type=model_type, resume=resume)
        self.max_interactions = max_interactions
        
        if resume:
            self.model = MaskablePPO.load(self.resume_model_path, env=self.train_env, device="auto")
            # Override num_timesteps
            self.model.num_timesteps = self.resume_ts
            self.history = self.resume_history
            self.generator.set_state(self.resume_gen_state)
            logger.info(f"Successfully resumed at timestep {self.resume_ts}")

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
            "completed_interactions": ts,
            "hashes": {
                "config": hash_file(ROOT / "configs" / "rl_v3_phase_c2.json"),
                "validation_manifest": hash_file(ROOT / "evaluation/manifests/rl_v3_phase_c2_validation.json"),
                "train_generator": hash_file(ROOT / "evaluation/manifests/rl_v3_phase_c2_train_generator.json")
            }
        }
        with open(self.out_dir / "provenance.json", "w") as f:
            json.dump(prov, f, indent=2)
            
        # Atomic Bundle Creation
        import tempfile
        tmp_zip = Path(tempfile.mktemp(suffix=".zip"))
        
        # inventory
        inventory_lines = []
        for fpath in self.out_dir.iterdir():
            if fpath.is_file() and fpath.name != "inventory.txt":
                h = hash_file(fpath)
                inventory_lines.append(f"{h}  {fpath.name}")
        with open(self.out_dir / "inventory.txt", "w") as f:
            f.write("\n".join(inventory_lines) + "\n")
            
        shutil.make_archive(str(tmp_zip)[:-4], 'zip', str(self.out_dir))
        
        bundle_path = Path(self.out_dir).parent / "latest_checkpoint_bundle.zip"
        shutil.move(str(tmp_zip), str(bundle_path))
        
        timestamped_bundle = Path(self.out_dir).parent / f"checkpoint_bundle_{ts:06d}.zip"
        shutil.copy2(str(bundle_path), str(timestamped_bundle))
        
        logger.info(f"Bundle archived to: {bundle_path}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="M1")
    parser.add_argument("--interactions", type=int, default=150000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bundle-path", type=str, default=None)
    args = parser.parse_args()
    
    logger.info(f"\n Launching Phase C2 Kaggle Runner")
    logger.info(f"Model: {args.model}")
    logger.info(f"Interactions: {args.interactions}")
    logger.info(f"Resume: {args.resume}\n")
    
    runner = KagglePhaseC2Runner(args.model, args.interactions, args.resume, args.bundle_path)
    runner.run(args.interactions)
    
    is_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
    out_dir = Path("/kaggle/working/uav_phase_c2" if is_kaggle else str(ROOT / "runs" / "uav_phase_c2_local_test"))
    
    final_archive = out_dir.parent / f"rl_v3_phase_c2_{args.model}_raw_artifacts"
    shutil.make_archive(str(final_archive), 'zip', str(out_dir))
    
    # Final Inventory
    lines = []
    for root, dirs, files in os.walk(str(out_dir)):
        for file in files:
            p = Path(root) / file
            h = hash_file(p)
            rel = p.relative_to(out_dir)
            lines.append(f"{h}  {rel}  {p.stat().st_size} bytes")
    
    with open(out_dir.parent / "final_inventory.txt", "w") as f:
        f.write("\n".join(lines))
        
    logger.info(f"Final artifacts archived to: {final_archive}.zip")
