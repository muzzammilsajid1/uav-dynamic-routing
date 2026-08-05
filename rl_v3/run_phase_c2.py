import json
import logging
from pathlib import Path
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from rl_v3.phase_c2_env import PhaseC2Env, PhaseC2EndpointGenerator
from tools.verification.r2_pb_wrapper import PotentialShapingWrapper

logger = logging.getLogger(__name__)

class CurriculumCallback(BaseCallback):
    def __init__(self, stages, generator, runner, verbose=0):
        super().__init__(verbose)
        self.stages = stages
        self.generator = generator
        self.runner = runner

    def _on_step(self):
        ts = self.num_timesteps
        active_sizes = None
        for stage in self.stages:
            if ts <= stage["max_interactions"]:
                active_sizes = stage["active_sizes"]
                break
        
        if active_sizes is None:
            active_sizes = self.stages[-1]["active_sizes"]
            
        if self.generator.active_sizes != active_sizes:
            logger.info(f"[{ts}] Curriculum shift: Active sizes now {active_sizes}")
            self.generator.set_active_sizes(active_sizes)
            
        # Checkpoint logic
        if ts in self.runner.checkpoints and ts not in self.runner.history:
            self.runner.evaluate_and_save(ts)
            
        return True

class PhaseC2Runner:
    def __init__(self, config_path, out_dir, model_type="M1", resume=False):
        with open(config_path) as f:
            self.config = json.load(f)
            
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model_type = model_type
        self.resume = resume
        
        self.checkpoints = [25000, 50000, 75000, 100000, 150000]
        self.history = {}
        
        self.generator = PhaseC2EndpointGenerator(seed=42)
        
        # Initialize initial active sizes
        self.generator.set_active_sizes(self.config["curriculum"]["stages"][0]["active_sizes"])
        
        # Create Train Env
        self.train_env = PhaseC2Env(self.config, mode="train", generator=self.generator)
        if self.config["reward"]["type"] == "R2-PB-empty-v1":
            self.train_env = PotentialShapingWrapper(
                self.train_env,
                gamma=self.config["reward"]["gamma"],
                lambda_=self.config["reward"]["lambda_"]
            )
            
        # Load or create model
        if resume:
            # Handle resume later
            pass
        else:
            if model_type == "M1":
                from rl_v3.phase_b_policy import PhaseBMaskableActorCriticPolicy
                self.model = MaskablePPO(
                    PhaseBMaskableActorCriticPolicy,
                    self.train_env,
                    **self.config["model"]
                )
            else:
                self.model = MaskablePPO(
                    "MultiInputPolicy",
                    self.train_env,
                    **self.config["model"]
                )
                
    def evaluate_and_save(self, ts):
        logger.info(f"[{ts}] Evaluating 240 validation routes...")
        val_env = PhaseC2Env(self.config, mode="eval", generator=self.generator)
        if self.config["reward"]["type"] == "R2-PB-empty-v1":
            val_env = PotentialShapingWrapper(
                val_env,
                gamma=self.config["reward"]["gamma"],
                lambda_=self.config["reward"]["lambda_"]
            )
            
        successes = 0
        collisions = 0
        timeouts = 0
        n_eval = len(self.generator.val_manifest)
        
        for i in range(n_eval):
            obs, info = val_env.reset()
            done = False
            steps = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = val_env.step(int(action))
                done = term or trunc
                steps += 1
                
            if info.get("is_success", False): successes += 1
            elif info.get("is_collision", False): collisions += 1
            else: timeouts += 1
            
        val_sr = successes / n_eval
        logger.info(f"[{ts}] Val SR: {val_sr:.3f}")
        
        self.history[ts] = {
            "success_rate": val_sr,
            "collisions": collisions,
            "timeouts": timeouts
        }
        
        self.model.save(str(self.out_dir / f"model_{ts:06d}.zip"))
        
        # Save generator state
        gen_state = self.generator.get_state()
        with open(self.out_dir / f"generator_{ts:06d}.json", "w") as f:
            json.dump(gen_state, f)
            
        # Save full status
        with open(self.out_dir / "status.json", "w") as f:
            json.dump({"history": self.history}, f, indent=2)

    def run(self, max_interactions=150000):
        cb = CurriculumCallback(self.config["curriculum"]["stages"], self.generator, self)
        logger.info(f"Starting Phase C2 training for {max_interactions} interactions")
        self.model.learn(total_timesteps=max_interactions, callback=cb)
        
        # Final save if not done
        if max_interactions not in self.history:
            self.evaluate_and_save(max_interactions)
