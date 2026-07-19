import io

with open('train_her_colab_v2.py', 'r', encoding='utf-8') as f:
    colab = f.read()

ddqn_code = '''
import torch
import torch.nn.functional as F

class DoubleDQN(DQN):
    """Subclass of SB3\\'s DQN that implements Double DQN."""
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            
            with torch.no_grad():
                # 1. Select action with highest value from ONLINE network
                next_q_values_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_values_online.argmax(dim=1, keepdim=True)
                
                # 2. Evaluate that action\\'s value using TARGET network
                next_q_values_target = self.q_net_target(replay_data.next_observations)
                next_q_values = torch.gather(next_q_values_target, dim=1, index=next_actions)
                
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

'''

# Inject DoubleDQN code before Model instantiation in Cell 3
model_marker = '# ---- Model -------------------------------------------------------------------'
if 'class DoubleDQN' not in colab:
    new_colab = colab.replace(model_marker, ddqn_code + model_marker)
else:
    new_colab = colab

# Replace DQN(...) with DoubleDQN(...)
new_colab = new_colab.replace('model = DQN(', 'model = DoubleDQN(')

# In Cell 4 and Cell 5 and Cell 6, replace DQN.load with DoubleDQN.load
new_colab = new_colab.replace('DQN.load(', 'DoubleDQN.load(')

with open('train_her_colab_v2.py', 'w', encoding='utf-8') as f:
    f.write(new_colab)
