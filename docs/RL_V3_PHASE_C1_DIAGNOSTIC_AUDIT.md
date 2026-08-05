# Phase C1 Learning Diagnostic Audit

## 1. Checkpoint Optimization Metrics

| Metric | 10k | 25k | 50k | 100k |
|--------|-----|-----|-----|------|
| **Training Rollout Success** | 15.0% | 13.0% | 11.0% | 15.0% |
| **Validation Success (Overall)** | 7.5% | 9.2% | 8.3% | 5.8% |
| **Validation Success (Short)** | 15.0% | 12.5% | 17.5% | 12.5% |
| **Validation Success (Medium)** | 0.0% | 7.5% | 5.0% | 0.0% |
| **Validation Success (Long)** | 7.5% | 7.5% | 2.5% | 5.0% |
| **Policy Entropy Loss** | -1.63 | -1.45 | -1.12 | -1.33 |
| **Value Loss** | 0.0064 | 0.0199 | 0.0101 | 0.0168 |
| **Explained Variance** | 0.365 | 0.370 | 0.346 | 0.024 |
| **Policy Gradient Loss** | -0.0425 | -0.0682 | -0.0570 | -0.0667 |
| **Approximate KL Divergence** | 0.0235 | 0.0741 | 0.1450 | 0.0914 |
| **Clipping Fraction** | 0.222 | 0.408 | 0.411 | 0.432 |

### Missing Metrics
The following metrics were **not saved** by the default Stable-Baselines3 logger configuration (and therefore cannot be retrospectively audited):
- Mean training episode length (`ep_len_mean` was omitted due to `Monitor` wrapper or tracking absence).
- Mean validation episode length.
- Action distribution (PPO does not log raw action distributions without a custom callback).
- Success rate stratified by training distance bin.
- Number of unique training endpoint pairs encountered.
- Mean and distribution of terminal goal rewards.
- Fraction of episodes receiving a terminal goal reward.

## 2. Diagnostic Determinations

1. **Did entropy collapse?**
   - **No.** The starting maximum entropy for 9 discrete actions is $\ln(9) \approx 2.197$. At 10k, it was 1.63, and it ended at 1.33. It dropped, indicating the policy became more deterministic, but it did not entirely collapse to zero (e.g., $H < 0.1$). It still maintained some stochasticity.

2. **Did one or two actions dominate?**
   - **Likely, but unconfirmed by explicit metric.** Given the high timeout rate and oscillatory behavior (constantly looping between two cells), the policy likely locked onto a 2-step repeating cycle of opposing actions (e.g., UP and DOWN). This explains the moderate remaining entropy (the policy mixes between a few actions to oscillate) rather than a single action mode-collapse (which would hit a wall/boundary).

3. **Did the value function fail to learn?**
   - **Yes, eventually.** `explained_variance` started around `0.36` (meaning the value function explained 36% of the variance in returns) but collapsed entirely to `0.024` at 100k. The value network completely lost its predictive power by the end of training.

4. **Did validation peak before the final checkpoint?**
   - **Yes.** Validation peaked at 9.2% at 25k interactions and then steadily declined to 5.8% by 100k.

5. **Did training success and validation success diverge?**
   - **No, both failed.** Training success hovered between 11-15%. It never reached a high value. This is a failure to fit the training distribution, not just a generalization gap.

6. **Did optimization become unstable?**
   - **Yes.** The Approximate KL divergence spiked to `0.145` at 50k, far exceeding the typical target KL range of 0.01-0.03 for stable PPO. The clipping fraction also stayed very high (over 40% of updates were clipped). This indicates the policy was thrashing against the PPO clip bounds repeatedly, a sign of unstable optimization likely caused by sparse, highly delayed, or inconsistent credit assignment.
