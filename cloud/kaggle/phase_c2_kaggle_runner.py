import os
import sys
import shutil
import json
import logging
import zipfile
import hashlib
import tempfile
from pathlib import Path
from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from rl_v3.run_phase_c2 import PhaseC2Runner, _write_json_atomic

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SOURCE_HASH_MODE = "lf_normalized_sha256_v1"


def _lf_normalized_bytes(path):
    data = Path(path).read_bytes()
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def hash_source_file(path):
    """Hash text source independently of Windows versus Linux line endings."""
    return hashlib.sha256(_lf_normalized_bytes(path)).hexdigest()


def _legacy_source_hash_candidates(path):
    """Accept a v11 raw hash on either platform without weakening content checks."""
    raw = Path(path).read_bytes()
    lf = _lf_normalized_bytes(path)
    crlf = lf.replace(b"\n", b"\r\n")
    return {
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(lf).hexdigest(),
        hashlib.sha256(crlf).hexdigest(),
    }


def source_hash_files():
    return {
        "config": ROOT / "configs" / "rl_v3_phase_c2.json",
        "validation_manifest": ROOT / "evaluation" / "manifests" / "rl_v3_phase_c2_validation.json",
        "train_generator": ROOT / "evaluation" / "manifests" / "rl_v3_phase_c2_train_generator.json",
        "reward_wrapper": ROOT / "tools" / "verification" / "r2_pb_wrapper.py",
        "observations": ROOT / "rl_v3" / "observations.py",
        "phase_c2_env": ROOT / "rl_v3" / "phase_c2_env.py",
        "phase_c2_runner": ROOT / "rl_v3" / "run_phase_c2.py",
        "phase_c0_env": ROOT / "rl_v3" / "phase_c0_env.py",
        "action_masking": ROOT / "rl_v3" / "action_masking.py",
        "kaggle_runner": Path(__file__).resolve(),
        "confirmatory_queue": ROOT / "scripts" / "run_phase_c2_confirmatory.py",
    }


def _is_final_evidence_file(path):
    """Select direct evidence while excluding prior/container archives."""
    name = Path(path).name
    if name.endswith(".tmp") or ".tmp." in name:
        return False
    if name == "latest_checkpoint_bundle.zip":
        return False
    if name.startswith("checkpoint_bundle_") and name.endswith(".zip"):
        return False
    if "_raw_artifacts" in name and name.endswith(".zip"):
        return False
    if name.startswith("phase_c2_") and name.endswith("_COMPLETE.zip"):
        return False
    return True


def _archive_files(files, destination):
    """Create an archive atomically from an explicit, flat file list."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory() as staging_dir,
        tempfile.TemporaryDirectory(dir=destination.parent) as archive_dir,
    ):
        staging = Path(staging_dir)
        for source in files:
            shutil.copy2(source, staging / Path(source).name)
        temporary_base = Path(archive_dir) / f"{destination.stem}.building"
        temporary_zip = Path(shutil.make_archive(str(temporary_base), "zip", str(staging)))
        temporary_zip.replace(destination)


def create_final_archives(out_dir, model_type, smoke=False):
    """Create non-nesting raw and complete backups plus a direct-file inventory."""
    out_dir = Path(out_dir)
    inventory_path = out_dir / "final_inventory.txt"
    payload = [
        path for path in sorted(out_dir.iterdir())
        if path.is_file() and path.name != inventory_path.name and _is_final_evidence_file(path)
    ]
    lines = [f"{hash_file(path)}  {path.name}  {path.stat().st_size} bytes" for path in payload]
    inventory_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    archive_payload = payload + [inventory_path]

    archive_stem = f"rl_v3_phase_c2_{model_type}_{'smoke_' if smoke else ''}raw_artifacts"
    raw_archive = out_dir / f"{archive_stem}.zip"
    _archive_files(archive_payload, raw_archive)

    complete_suffix = "SMOKE_COMPLETE" if smoke else "COMPLETE"
    complete_archive = out_dir.parent / f"phase_c2_{model_type}_{complete_suffix}.zip"
    _archive_files(archive_payload, complete_archive)
    return raw_archive, complete_archive


def resolve_output_dir(smoke=False):
    is_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
    if smoke:
        override = os.environ.get("KAGGLE_SMOKE_OUT_DIR")
        if override:
            return Path(override)
        if is_kaggle:
            return Path("/kaggle/working/uav_phase_c2_smoke")
        return ROOT / "runs" / "uav_phase_c2_smoke"

    test_out = os.environ.get("KAGGLE_TEST_OUT_DIR")
    if test_out:
        return Path(test_out)
    test_prod = os.environ.get("KAGGLE_TEST_PROD_DIR")
    kaggle_path = (
        Path(test_prod) / "uav_phase_c2"
        if test_prod
        else Path("/kaggle/working/uav_phase_c2")
    )
    return kaggle_path if is_kaggle else ROOT / "runs" / "uav_phase_c2_local_test"


class KagglePhaseC2Runner(PhaseC2Runner):
    def __init__(
        self,
        model_type,
        max_interactions,
        resume=False,
        bundle_path=None,
        device="auto",
        smoke=False,
        seed=None,
    ):
        out_dir = resolve_output_dir(smoke)
        
        config_path = ROOT / "configs" / "rl_v3_phase_c2.json"
        
        self.bundle_path = bundle_path
        if resume:
            self._handle_resume(bundle_path, out_dir, model_type, seed)
            seed = self.resume_seed
            
        super().__init__(
            config_path=config_path,
            out_dir=out_dir,
            model_type=model_type,
            resume=resume,
            device=device,
            seed=seed,
        )
        self.max_interactions = max_interactions
        
        if resume:
            self.generator.set_state(self.resume_gen_state)
            # SB3 does not serialize the partially active episode. The resumed
            # environment therefore starts a fresh episode from the curriculum
            # stage governing the next absolute interaction.
            self.generator.set_active_sizes(self.active_sizes_for_interaction(self.resume_ts + 1))
            self.model = MaskablePPO.load(self.resume_model_path, env=self.train_env, device=self.device)
            self.model.num_timesteps = self.resume_ts
            self.history = self.resume_history
            self.restore_rng_state(self.resume_rng_path)
            logger.info(f"Successfully resumed at timestep {self.resume_ts} on device {self.device}")

    def _handle_resume(self, bundle_path, out_dir, expected_model, expected_seed=None):
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
        self.resume_seed = int(prov["seed"])
        if expected_seed is not None and self.resume_seed != int(expected_seed):
            raise ValueError(
                f"Bundle seed {self.resume_seed} != requested seed {int(expected_seed)}"
            )
            
        # Verify every training/evaluation-relevant source captured by v11.
        source_hash_mode = prov.get("source_hash_mode")
        for name, path in source_hash_files().items():
            expected = prov.get("hashes", {}).get(name)
            if expected is None:
                raise ValueError(f"Resume provenance is missing source hash: {name}")
            if source_hash_mode == SOURCE_HASH_MODE:
                matches = expected == hash_source_file(path)
            elif source_hash_mode is None:
                matches = expected in _legacy_source_hash_candidates(path)
            else:
                raise ValueError(f"Unsupported resume source hash mode: {source_hash_mode}")
            if not matches:
                raise ValueError(f"Resume source hash mismatch for {name}")
            
        self.resume_ts = prov["completed_interactions"]
        self.resume_model_path = extract_dir / f"model_{self.resume_ts:06d}.zip"
        self.resume_rng_path = extract_dir / f"rng_{self.resume_ts:06d}.pt"
        if not self.resume_rng_path.exists():
            raise RuntimeError(f"Resume bundle is missing RNG state: {self.resume_rng_path.name}")
        
        with open(extract_dir / f"generator_{self.resume_ts:06d}.json", "r") as f:
            self.resume_gen_state = json.load(f)
            
        with open(extract_dir / "status.json", "r") as f:
            self.resume_history = json.load(f)["history"]
            
        # Also clean up string keys back to int if needed
        self.resume_history = {int(k): v for k, v in self.resume_history.items()}

    def evaluate_and_save(self, ts, requested_ts=None):
        super().evaluate_and_save(ts, requested_ts)
        
        logger.info("\n==============================================")
        logger.info(f" CHECKPOINT SAVED: {ts} interactions")
        logger.info(f" Validation Success Rate: {self.history[ts]['success_rate']*100:.1f}%")
        logger.info(f" Output Directory: {self.out_dir}")
        logger.info("==============================================\n")
        
        # Provenance
        import subprocess
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode("utf-8").strip()
        except Exception:
            git_commit = "unknown"
            
        import torch
        prov = {
            "model_type": self.model_type,
            "device": self.device,
            "resolved_device": str(self.model.device),
            "completed_interactions": ts,
            "requested_interactions": int(ts if requested_ts is None else requested_ts),
            "rollout_size": self.rollout_size(),
            "checkpoint_is_ppo_update_aligned": int(ts) % self.rollout_size() == 0,
            "seed": getattr(self, "seed", None),
            "deterministic_cuda_requested": getattr(self, "deterministic_cuda", False),
            "deterministic_backend_flags": {
                "cudnn.deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn.benchmark": bool(torch.backends.cudnn.benchmark),
                "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled())
            },
            "is_resumed_run": self.resume,
            "resume_semantics": "statistically_equivalent",
            "resume_limit": "SB3 does not serialize the partially active environment episode or rollout buffer.",
            "git_commit": git_commit,
            "source_hash_mode": SOURCE_HASH_MODE,
            "hashes": {
                name: hash_source_file(path) for name, path in source_hash_files().items()
            },
            "checkpoint_schedule": getattr(self, "runtime_checkpoint_plan", None),
        }
        _write_json_atomic(self.out_dir / "provenance.json", prov)
            
        # Atomic Bundle Creation — bundles live inside self.out_dir, never in parent.
        # Build from an explicit whitelist of checkpoint-state files only.
        # Excluded from bundles: any *.zip file (avoids nested archives and raw-artifact
        # archives), final_inventory.txt (only written at run end), and
        # partial/temporary files.
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

        logger.info("\n DURABILITY BACKUP SAVED:")
        logger.info(f"  Timestep: {ts}")
        logger.info(f"  Latest bundle: {bundle_path} ({bundle_path.stat().st_size} bytes)")
        logger.info(f"  Latest SHA-256: {hash_file(bundle_path)}")
        logger.info(f"  Timestamped bundle: {timestamped_bundle} ({timestamped_bundle.stat().st_size} bytes)")
        logger.info(f"  Timestamped SHA-256: {hash_file(timestamped_bundle)}\n")

        logger.info(f"Bundle archived to: {bundle_path}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="M1")
    parser.add_argument("--interactions", type=int, default=150000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bundle-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--smoke", action="store_true", help="Use an isolated smoke output directory")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    
    logger.info(f"\n Launching Phase C2 Kaggle Runner")
    logger.info(f"Model: {args.model}")
    logger.info(f"Interactions: {args.interactions}")
    logger.info(f"Resume: {args.resume}")
    logger.info(f"Seed: {args.seed if args.seed is not None else 'config default'}")
    logger.info(f"Device: {args.device}\n")
    
    runner = KagglePhaseC2Runner(
        args.model,
        args.interactions,
        args.resume,
        args.bundle_path,
        args.device,
        args.smoke,
        args.seed,
    )
    runner.run(args.interactions)
    
    out_dir = resolve_output_dir(args.smoke)

    final_archive, final_complete_archive = create_final_archives(
        out_dir, args.model, smoke=args.smoke
    )
    logger.info(f"Final artifacts archived to: {final_archive}")
    logger.info(f"COMPLETE BACKUP CREATED: {final_complete_archive}")
    logger.info("********************************************************************************")
    logger.info(" DO NOT DISCONNECT THE NOTEBOOK KERNEL YET!")
    logger.info(f" WAIT UNTIL {final_complete_archive.name} HAS BEEN DOWNLOADED")
    logger.info(" OR THE NOTEBOOK VERSION HAS BEEN SAVED SUCCESSFULLY WITH ALL OUTPUTS.")
    logger.info("********************************************************************************")
