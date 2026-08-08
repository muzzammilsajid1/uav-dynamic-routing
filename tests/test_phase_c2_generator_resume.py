def test_generator_state_restoration():
    from rl_v3.phase_c2_env import PhaseC2EndpointGenerator
    
    # Initialize generator with seed
    gen1 = PhaseC2EndpointGenerator(seed=1234)
    gen1.set_active_sizes([15, 30])
    
    # advance it
    for _ in range(10):
        gen1.sample_train()
        
    # serialize state
    state = gen1.get_state()
    assert state["active_sizes"] == [15, 30]
    
    # generate next N endpoints and record them
    expected_endpoints = []
    for _ in range(5):
        expected_endpoints.append(gen1.sample_train())
        
    # construct a new generator
    gen2 = PhaseC2EndpointGenerator(seed=5678)  # Different seed to ensure state overwrites it
    gen2.set_active_sizes([100])
    
    # restore serialized state
    gen2.set_state(state)
    assert gen2.active_sizes == [15, 30]
    
    # verify next N endpoints are exactly identical
    actual_endpoints = []
    for _ in range(5):
        actual_endpoints.append(gen2.sample_train())
        
    assert expected_endpoints == actual_endpoints

def test_runner_level_resume(tmp_path, monkeypatch):
    import os
    import sys
    from cloud.kaggle.phase_c2_kaggle_runner import KagglePhaseC2Runner
    
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    
    out_dir = tmp_path / "uav_phase_c2"
    out_dir.mkdir(parents=True)
    monkeypatch.setenv("KAGGLE_TEST_OUT_DIR", str(out_dir))
    
    # Run a tiny 5-step run
    runner = KagglePhaseC2Runner("M1", 5, device="cpu")
    runner.checkpoints = [5]
    # Use one interaction per rollout so this unit test exercises exact resume
    # arithmetic without invoking the production 2,048-step alignment policy.
    runner.model.n_steps = 1
    
    # Mock learn to just increment num_timesteps
    def mock_learn(total_timesteps, callback, reset_num_timesteps):
        runner.model.num_timesteps += total_timesteps
    monkeypatch.setattr(runner.model, "learn", mock_learn)
    
    runner.run(5)
    
    bundle_path = out_dir / "latest_checkpoint_bundle.zip"
    assert bundle_path.exists()
    
    # Resume it for 5 more steps (total 10)
    runner2 = KagglePhaseC2Runner("M1", 10, resume=True, bundle_path=str(bundle_path), device="cpu")
    runner2.checkpoints = [5, 10]
    
    assert runner2.model.num_timesteps == 5
    
    def mock_learn2(total_timesteps, callback, reset_num_timesteps):
        runner2.model.num_timesteps += total_timesteps
    monkeypatch.setattr(runner2.model, "learn", mock_learn2)
    
    runner2.run(10)
    
    assert runner2.model.num_timesteps == 10
