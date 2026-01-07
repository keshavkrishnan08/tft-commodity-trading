
# src/models/rl_meta_learner.py
"""
Reinforcement Learning Meta-Learner (PPO).

replaces the Random Forest Meta-Learner with a PPO Agent that 
optimizes trading decisions via trial and error.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import pickle
from pathlib import Path

class RLTradingEnv:
    """
    Simulates a trading environment for the RL agent.
    Iterates through meta-training data (potential trades).
    """
    def __init__(self, data, config):
        """
        Args:
            data: DataFrame containing 'features' and 'actual_return'
            config: Config object
        """
        self.data = data.reset_index(drop=True)
        self.config = config
        self.current_step = 0
        self.max_steps = len(data)
        
        # Account state tracking for simulation
        self.balance = 10000.0
        self.initial_balance = 10000.0
        self.peak_balance = 10000.0
        self.wins = 0
        self.losses = 0
        
        # Pre-convert features to float32
        # Assuming last column is actual_return, rest are features
        feature_cols = [c for c in data.columns if c not in ['actual_return', 'asset', 'optimal_position']]
        self.features = data[feature_cols].values.astype(np.float32)
        self.returns = data['actual_return'].values.astype(np.float32)
        
        self.obs_dim = self.features.shape[1]

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.wins = 0
        self.losses = 0
        return self._get_obs()

    def _get_obs(self):
        if self.current_step >= self.max_steps:
            return np.zeros(self.obs_dim, dtype=np.float32)
        return self.features[self.current_step]

    def step(self, action):
        """
        Modified Reward Function: Optimize for Sharpe/Sortino Ratio.
        Uses Stationary Returns (non-compounding) for training stability.
        """
        # 1. Execute Trade
        position_size = np.clip(action, 0.0, 0.10)
        
        actual_ret = self.returns[self.current_step]
        
        # Dollar PnL based on INITIAL BALANCE (Non-compounding for stability)
        # This prevents the balance from reaching 10^15 and exploding gradients
        position_dollars = self.initial_balance * position_size
        pnl_dollars = position_dollars * actual_ret
        
        # Update Account (Compounding still tracked for 'info', but reward is stationary)
        # Actually, for training, let's keep balance growth linear
        self.balance += pnl_dollars
        
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
            
        current_drawdown = (self.peak_balance - self.balance) / self.peak_balance
        
        # 2. Advanced Reward Function (Sharpe-Optimized)
        # Base: Return in Basis Points relative to initial capital
        ret_bps = (pnl_dollars / self.initial_balance) * 10000 
        
        # Penalty 1: Asymmetric Loss Weighting (Sortino-style)
        if pnl_dollars < 0:
            reward = ret_bps * 2.0 # Heavier penalty for losses
        else:
            reward = ret_bps * 1.0
            
        # Penalty 2: Drawdown Punishment
        # Stationary DD penalty
        dd_penalty = (current_drawdown * 50) 
        reward -= dd_penalty
        
        # 3. Reward Clipping for Gradient Stability
        reward = np.clip(reward, -20.0, 20.0)
        
        # 4. Next Step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        next_obs = self._get_obs()
        
        info = {
            'pnl': pnl_dollars,
            'balance': self.balance,
            'drawdown': current_drawdown
        }
        
        return next_obs, reward, done, info


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim=1, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # Actor: Outputs Mean and Std for Action Distribution
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.actor_mu = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim)) # Learnable log_std
        
        # Critic: Estimates Value Function V(s)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        # Value
        value = self.critic(state)
        
        # Policy
        x = self.actor(state)
        mu = torch.sigmoid(self.actor_mu(x)) * 0.10 # Sigmoid -> [0, 1] -> Scaled to [0, 0.10]
        # Alternative: Tanh -> [-1, 1] -> shifted to [0, 0.10]
        # Using Sigmoid * MaxPosSize ensures constraints
        
        std = self.actor_log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        
        return dist, value

class PPOAgent:
    def __init__(self, state_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4, total_episodes=200):
        self.pixel_gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim).float()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Learning Rate Scheduler: Linear decay to stabilization
        # Requires PyTorch 1.10+. If this fails, we can swap to StepLR.
        try:
            self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_episodes)
        except AttributeError:
            # Fallback for older PyTorch
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=total_episodes//4, gamma=0.5)
            
        # Entropy Decay for exploration cooling
        self.entropy_coef = 0.02 # Start slightly higher
        self.entropy_decay = 0.98
        self.min_entropy = 0.001

        self.policy_old = ActorCritic(state_dim).float()
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            dist, value = self.policy_old(state)
            action = dist.sample()
            
        return action.item(), dist.log_prob(action).item()

    def update(self, memory):
        # memory: list of (state, action, log_prob, reward, done)
        states = torch.FloatTensor(np.array([t[0] for t in memory]))
        actions = torch.FloatTensor(np.array([t[1] for t in memory])).unsqueeze(1)
        old_log_probs = torch.FloatTensor(np.array([t[2] for t in memory])).unsqueeze(1)
        rewards = torch.FloatTensor(np.array([t[3] for t in memory])).unsqueeze(1)
        
        # Discounted Rewards (Rewards-to-Go)
        discounted_rewards = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + (self.pixel_gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)
            
        discounted_rewards = torch.stack(discounted_rewards).squeeze(1).detach()
        # Normalize
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        
        # PPO Update
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            dist, state_values = self.policy(states)
            log_probs = dist.log_prob(actions)
            dist_entropy = dist.entropy()
            
            # Match tensor shapes
            state_values = torch.squeeze(state_values)
            
            # Ratio
            ratios = torch.exp(log_probs - old_log_probs.detach())
            
            # Surrogate Loss
            advantages = discounted_rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Total Loss (Dynamic Entropy)
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, discounted_rewards) - self.entropy_coef*dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Step Schedulers
        self.scheduler.step()
        self.entropy_coef = max(self.min_entropy, self.entropy_coef * self.entropy_decay)

class RLMetaLearner:
    """Wrapper to replace the RF MetaLearner"""
    def __init__(self, config):
        self.config = config
        self.agent = None
        self.scaler = None # Will be fitted during training
        
    def _extract_features(self, direction_prob, market_state, account_state):
        # Same as RF MetaLearner
        features = [
            direction_prob,
            abs(direction_prob - 0.5) * 2,
            market_state.get('volatility', 0.02),
            market_state.get('momentum_20d', 0.0),
            market_state.get('momentum_60d', 0.0),
            market_state.get('rsi', 50.0) / 100.0,
            market_state.get('macd_hist', 0.0),
            market_state.get('volume_surge', 1.0),
            account_state.get('capital', 10000) / 10000,
            account_state.get('drawdown', 0.0),
            account_state.get('recent_win_rate', 0.5),
            account_state.get('recent_sharpe', 0.0),
            account_state.get('consecutive_wins', 0),
            account_state.get('consecutive_losses', 0),
        ]
        return np.array(features, dtype=np.float32)

    def fit(self, training_data, episodes=500):
        print("\nTRAINING RL AGENT (PPO)...")
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        
        # 1. Prepare Data
        # Build feature matrix for scaling fitting
        raw_features = []
        for _, row in training_data.iterrows():
             # Re-construct context to use _extract_features logic
             ms = {k: row.get(k, 0) for k in ['volatility', 'momentum_20d', 'momentum_60d', 'rsi', 'macd_hist', 'volume_surge']}
             as_ = {k: row.get(k, 0) for k in ['capital', 'drawdown', 'recent_win_rate', 'recent_sharpe', 'consecutive_wins', 'consecutive_losses']}
             feat = self._extract_features(row['direction_prob'], ms, as_)
             raw_features.append(feat)
        
        raw_features = np.array(raw_features)
        self.scaler.fit(raw_features)
        
        # Create environment-ready dataframe
        # We replace the raw columns with the SCALED features for the Env to consume directly
        scaled_features = self.scaler.transform(raw_features)
        env_data = pd.DataFrame(scaled_features) # Columns 0..N
        env_data['actual_return'] = training_data['actual_return'].values
        
        env = RLTradingEnv(env_data, self.config)
        self.agent = PPOAgent(state_dim=env.obs_dim, total_episodes=episodes)
        
        # 2. Training Loop
        best_reward = -float('inf')
        patience = 20
        no_improvement_count = 0
        
        for ep in range(episodes):
            state = env.reset()
            memory = []
            episode_reward = 0
            
            while True:
                action, log_prob = self.agent.select_action(state)
                # Clip action to valid range for step
                action_clipped = np.clip(action, 0.0, 0.10)
                
                next_state, reward, done, _ = env.step(action_clipped)
                
                memory.append((state, action, log_prob, reward, done))
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Update Policy
            self.agent.update(memory)
            
            # Early Stopping Check
            if episode_reward > best_reward:
                best_reward = episode_reward
                no_improvement_count = 0
                # Save the BEST model state locally
                best_policy_state = {k: v.cpu().clone() for k, v in self.agent.policy.state_dict().items()}
            else:
                no_improvement_count += 1
            
            if (ep+1) % 1 == 0:
                print(f"Episode {ep+1}/{episodes} | Reward: {episode_reward:.2f} | Best: {best_reward:.2f} | Balance: ${env.balance:.0f}")
                
            if no_improvement_count >= patience:
                print(f"\nEARLY STOPPING: No improvement for {patience} episodes. Convergence reached at episode {ep+1}.")
                # Restore best weights
                self.agent.policy.load_state_dict(best_policy_state)
                break
        
        # After training, ensure we have the best weights loaded
        if 'best_policy_state' in locals():
            self.agent.policy.load_state_dict(best_policy_state)

    def predict_decision(self, direction_prob, market_state, account_state):
        if self.agent is None:
            return {'should_take': False, 'position_size': 0.0, 'confidence': 0.0}
            
        features = self._extract_features(direction_prob, market_state, account_state)
        # Scale
        features_scaled = self.scaler.transform(features.reshape(1, -1))[0]
        
        # Select action
        action, _ = self.agent.select_action(features_scaled)
        
        # Interpret Action
        # Action is position size [0.0, 0.10]
        position_size = np.clip(action, 0.0, 0.10)
        
        # GUARDRAIL: Volatility Check
        # If market is crazy (Vol > 5% daily), force FLAT.
        volatility = market_state.get('volatility', 0.0)
        if volatility > 0.05:
            position_size = 0.0
            
        should_take = position_size > 0.005 # Min threshold
        
        return {
            'should_take': bool(should_take),
            'take_probability': 1.0 if should_take else 0.0, # RL is deterministic approx
            'position_size': float(position_size),
            'confidence': float(position_size * 10) # Proxy
        }
    
    def update_confidence_feedback(self, *args):
        pass # Not implementing online learning for RL yet
        
    def save(self, path):
        # Save PyTorch model + Scaler
        state = {
            'policy': self.agent.policy.state_dict(),
            'scaler': self.scaler
        }
        torch.save(state, path)
        
    def load(self, path):
        state = torch.load(path)
        # Re-init agent with correct dim if needed, for now assume standard
        # Standard dim is 14 (from _extract_features)
        self.agent = PPOAgent(state_dim=14) 
        self.agent.policy.load_state_dict(state['policy'])
        self.scaler = state['scaler']
