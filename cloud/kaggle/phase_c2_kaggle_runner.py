import os
import sys
import shutil
import logging
from pathlib import Path

# Add root to sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from rl_v3.run_phase_c2 import PhaseC2Runner

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class KagglePhaseC2Runner(PhaseC2Runner):
    def __init__(self, model_type, max_interactions, resume=False):
        # Always output to /kaggle/working unless testing locally
        is_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
        out_dir = "/kaggle/working/uav_phase_c2" if is_kaggle else str(ROOT / "runs" / "uav_phase_c2_local_test")
        
        config_path = ROOT / "configs" / "rl_v3_phase_c2.json"
        super().__init__(config_path=config_path, out_dir=out_dir, model_type=model_type, resume=resume)
        self.max_interactions = max_interactions

    def evaluate_and_save(self, ts):
        super().evaluate_and_save(ts)
        
        logger.info("\n==============================================")
        logger.info(f" CHECKPOINT SAVED: {ts} interactions")
        logger.info(f" Validation Success Rate: {self.history[ts]['success_rate']*100:.1f}%")
        logger.info(f" Output Directory: {self.out_dir}")
        logger.info("==============================================\n")
        
        # Package bundle
        bundle_path = Path(self.out_dir).parent / "latest_checkpoint_bundle"
        shutil.make_archive(str(bundle_path), 'zip', str(self.out_dir))
        logger.info(f"Bundle archived to: {bundle_path}.zip\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="M1")
    parser.add_argument("--interactions", type=int, default=150000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    logger.info(f"\n🚀 Launching Phase C2 Kaggle Runner")
    logger.info(f"Model: {args.model}")
    logger.info(f"Interactions: {args.interactions}")
    logger.info(f"Resume: {args.resume}\n")
    
    runner = KagglePhaseC2Runner(args.model, args.interactions, args.resume)
    runner.run(args.interactions)
    
    # Final archive logic
    is_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
    out_dir = Path("/kaggle/working/uav_phase_c2" if is_kaggle else str(ROOT / "runs" / "uav_phase_c2_local_test"))
    
    final_archive = out_dir.parent / f"rl_v3_phase_c2_{args.model}_raw_artifacts"
    shutil.make_archive(str(final_archive), 'zip', str(out_dir))
    logger.info(f"Final artifacts archived to: {final_archive}.zip")
