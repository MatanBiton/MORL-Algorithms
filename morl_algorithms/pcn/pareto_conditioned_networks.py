"""
Pareto Conditioned Networks (PCN) for Multi-Objective Reinforcement Learning.

Implementation adapted from the original PCN paper and codebase to match the 
structure and dependencies of the MOSAC agent in this repository.

Reference:
Reymond, M., Bargiacchi, E., & Nowé, A. (2022, May). Pareto Conditioned Networks.
In Proceedings of the 21st International Conference on Autonomous Agents
and Multiagent Systems (pp. 1110-1118).
https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1110.pdf

Key Features:
- Continuous action space support (as required for EnergyNet environment)
- Experience replay with crowding distance-based prioritization
- Command selection based on non-dominated solutions
- Direct policy learning conditioned on desired returns and horizons
"""

import heapq
import random
from collections import deque
from typing import List, Tuple, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


# ==============================
# Utilities
# ==============================

def crowding_distance(points: np.ndarray) -> np.ndarray:
    """
    Compute the crowding distance of a set of points for diversity preservation.
    
    Args:
        points: Array of shape (n_points, n_objectives) containing the points
        
    Returns:
        Array of crowding distances for each point
    """
    if len(points) <= 2:
        return np.ones(len(points))
    
    # Normalize across dimensions to ensure equal scaling
    points_norm = (points - points.min(axis=0)) / (points.ptp(axis=0) + 1e-8)
    
    # Sort points per dimension
    dim_sorted = np.argsort(points_norm, axis=0)
    point_sorted = np.take_along_axis(points_norm, dim_sorted, axis=0)
    
    # Compute distances between neighboring points
    distances = np.abs(point_sorted[:-2] - point_sorted[2:])
    
    # Pad extrema with maximum distance (1) for each dimension
    distances = np.pad(distances, ((1, 1), (0, 0)), constant_values=1)
    
    # Sum distances across dimensions for each point
    crowding = np.zeros(points_norm.shape)
    crowding[dim_sorted, np.arange(points_norm.shape[-1])] = distances
    crowding = np.sum(crowding, axis=-1)
    
    return crowding


def get_non_dominated_indices(points: np.ndarray) -> np.ndarray:
    """
    Get indices of non-dominated points (Pareto front).
    
    Args:
        points: Array of shape (n_points, n_objectives)
        
    Returns:
        Boolean array indicating which points are non-dominated
    """
    n_points = points.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    
    for i, point in enumerate(points):
        if is_efficient[i]:
            # Check if any other point dominates this one
            is_efficient[is_efficient] = np.any(points[is_efficient] > point, axis=1)
            is_efficient[i] = True  # Keep the current point
    
    return is_efficient


# ==============================
# PCN Model Architecture
# ==============================

class PCNModel(nn.Module):
    """
    Pareto Conditioned Network model for continuous action spaces.
    
    The model takes as input:
    - State observation
    - Desired return vector
    - Desired horizon (remaining time steps)
    
    And outputs the action to take.
    """
    
    def __init__(self, state_dim: int, action_dim: int, reward_dim: int, 
                 scaling_factor: np.ndarray, hidden_sizes: Tuple[int, ...] = (256, 256)):
        """
        Initialize the PCN model.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space  
            reward_dim: Number of reward objectives
            scaling_factor: Scaling factors for desired return and horizon
            hidden_sizes: Hidden layer sizes
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        
        # Scaling factor is fixed and not learned
        self.register_buffer('scaling_factor', torch.tensor(scaling_factor, dtype=torch.float32))
        
        # State embedding network
        state_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_sizes:
            state_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        self.state_embedding = nn.Sequential(*state_layers)
        
        # Command embedding network (desired return + horizon)
        command_dim = reward_dim + 1  # +1 for horizon
        command_layers = []
        prev_dim = command_dim
        for hidden_dim in hidden_sizes:
            command_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        self.command_embedding = nn.Sequential(*command_layers)
        
        # Final layers that combine state and command embeddings
        self.final_layers = nn.Sequential(
            nn.Linear(prev_dim, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], action_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor, desired_return: torch.Tensor, 
                desired_horizon: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PCN model.
        
        Args:
            state: State observations (batch_size, state_dim)
            desired_return: Desired return vectors (batch_size, reward_dim)
            desired_horizon: Desired horizons (batch_size, 1)
            
        Returns:
            Action predictions (batch_size, action_dim)
        """
        # Combine desired return and horizon into command vector
        command = torch.cat([desired_return, desired_horizon], dim=-1)
        
        # Scale command by fixed scaling factors
        command = command * self.scaling_factor
        
        # Compute embeddings
        state_emb = self.state_embedding(state.float())
        command_emb = self.command_embedding(command)
        
        # Element-wise multiplication and final prediction
        # This follows the original PCN architecture where state and command
        # embeddings are multiplied element-wise before final layers
        combined = state_emb * command_emb
        action = self.final_layers(combined)
        
        return action


# ==============================
# Experience Replay with Prioritization
# ==============================

class Transition:
    """Single transition in an episode."""
    
    def __init__(self, observation: np.ndarray, action: np.ndarray, 
                 reward: np.ndarray, next_observation: np.ndarray, done: bool):
        self.observation = observation
        self.action = action
        self.reward = reward  # Will be modified to contain cumulative return
        self.next_observation = next_observation
        self.done = done


class PrioritizedExperienceReplay:
    """
    Experience replay buffer that maintains episodes and prioritizes them
    based on crowding distance in the objective space.
    """
    
    def __init__(self, max_size: int = 100, gamma: float = 0.99):
        """
        Initialize the experience replay buffer.
        
        Args:
            max_size: Maximum number of episodes to store
            gamma: Discount factor for computing returns
        """
        self.max_size = max_size
        self.gamma = gamma
        # Heap structure: (priority, timestep, episode)
        # Priority is negative crowding distance (for max-heap behavior)
        self.buffer = []
        
    def add_episode(self, transitions: List[Transition], timestep: int):
        """
        Add a complete episode to the buffer.
        
        Args:
            transitions: List of transitions forming an episode
            timestep: Global timestep when episode was collected
        """
        # Compute discounted returns backwards through the episode
        for i in reversed(range(len(transitions) - 1)):
            transitions[i].reward += self.gamma * transitions[i + 1].reward
        
        # Add episode to heap buffer
        if len(self.buffer) >= self.max_size:
            # Replace least prioritized episode
            heapq.heappushpop(self.buffer, (1.0, timestep, transitions))
        else:
            heapq.heappush(self.buffer, (1.0, timestep, transitions))
    
    def update_priorities(self, threshold: float = 0.2):
        """
        Update episode priorities based on crowding distance.
        
        Args:
            threshold: Threshold for crowding distance penalty
        """
        if len(self.buffer) < 2:
            return
            
        # Extract returns from all episodes
        returns = np.array([episode[2][0].reward for episode in self.buffer])
        
        # Compute crowding distances
        distances = crowding_distance(returns)
        
        # Identify points that are too close together
        close_indices = np.where(distances <= threshold)[0]
        
        # Get non-dominated indices
        non_dominated_mask = get_non_dominated_indices(returns)
        non_dominated_returns = returns[non_dominated_mask]
        
        # Compute distance to closest non-dominated point for each episode
        priorities = np.zeros(len(returns))
        for i, ret in enumerate(returns):
            if len(non_dominated_returns) > 0:
                distances_to_front = np.linalg.norm(non_dominated_returns - ret, axis=1)
                priorities[i] = -np.min(distances_to_front)  # Negative for max-heap
            else:
                priorities[i] = -1.0
        
        # Apply penalties for overcrowded regions
        priorities[close_indices] *= 2.0
        
        # Apply penalty for duplicate points
        unique_returns, unique_indices = np.unique(returns, axis=0, return_index=True)
        duplicate_mask = np.ones(len(returns), dtype=bool)
        duplicate_mask[unique_indices] = False
        priorities[duplicate_mask] -= 1e-5
        
        # Update heap with new priorities
        for i, (_, timestep, episode) in enumerate(self.buffer):
            self.buffer[i] = (priorities[i], timestep, episode)
        
        heapq.heapify(self.buffer)
    
    def get_best_episodes(self, n: int) -> List:
        """Get the n best episodes based on current priorities."""
        if n >= len(self.buffer):
            return [episode for _, _, episode in self.buffer]
        
        # Update priorities first
        self.update_priorities()
        
        # Get n largest (most negative priorities = highest actual priorities)
        best = heapq.nlargest(n, self.buffer, key=lambda x: x[0])
        return [episode for _, _, episode in best]
    
    def sample_batch(self, batch_size: int) -> List[Tuple]:
        """
        Sample a batch of transitions for training.
        
        Args:
            batch_size: Size of batch to sample
            
        Returns:
            List of (state, action, desired_return, desired_horizon) tuples
        """
        if len(self.buffer) == 0:
            return []
        
        batch = []
        
        # Randomly sample episodes
        episode_indices = np.random.choice(len(self.buffer), size=batch_size, replace=True)
        
        for idx in episode_indices:
            _, _, episode = self.buffer[idx]
            
            # Randomly sample timestep from episode
            t = np.random.randint(0, len(episode))
            
            # Get transition data
            transition = episode[t]
            state = transition.observation
            action = transition.action
            
            # Desired return is the actual return achieved from this timestep
            desired_return = transition.reward.copy()
            
            # Desired horizon is remaining timesteps in episode
            desired_horizon = float(len(episode) - t)
            
            batch.append((state, action, desired_return, desired_horizon))
        
        return batch
    
    def __len__(self):
        return len(self.buffer)


# ==============================
# Main PCN Agent
# ==============================

class PCN:
    """
    Pareto Conditioned Networks agent for multi-objective reinforcement learning.
    
    This implementation is adapted to work with the same environment interface
    and structure as the MOSAC agent, making it a drop-in replacement.
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_low: np.ndarray,
        act_high: np.ndarray,
        objectives: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 256,
        buffer_capacity: int = 100,  # Number of episodes, not transitions
        max_steps_per_episode: int = 1000,
        training_frequency: int = 1000,
        noise_std: float = 0.1,
        scaling_factor: Optional[np.ndarray] = None,
        writer_filename: str = 'pcn_runs',
        verbose: bool = False
    ):
        """
        Initialize PCN agent.
        
        Args:
            obs_dim: Dimension of observation space
            act_dim: Dimension of action space
            act_low: Lower bounds for actions
            act_high: Upper bounds for actions
            objectives: Number of reward objectives
            hidden_sizes: Hidden layer sizes for networks
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            batch_size: Batch size for training
            buffer_capacity: Maximum number of episodes in replay buffer
            max_steps_per_episode: Maximum steps per episode
            training_frequency: How often to train (in environment steps)
            noise_std: Standard deviation of exploration noise
            scaling_factor: Scaling factors for desired returns and horizon
            writer_filename: Tensorboard log directory
            verbose: Whether to print training progress
        """
        self.objectives = int(objectives)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.max_steps = max_steps_per_episode
        self.training_frequency = training_frequency
        self.noise_std = noise_std
        self.verbose = verbose
        
        # Set default scaling factor if not provided
        if scaling_factor is None:
            # Default scaling: normalize desired returns to [0,1] and horizon to [0,1]
            self.scaling_factor = np.ones(self.objectives + 1, dtype=np.float32) * 0.01
        else:
            assert len(scaling_factor) == self.objectives + 1, \
                f"Scaling factor must have length {self.objectives + 1}"
            self.scaling_factor = np.array(scaling_factor, dtype=np.float32)
        
        # Action bounds for clipping
        self.action_low = torch.tensor(act_low, dtype=torch.float32)
        self.action_high = torch.tensor(act_high, dtype=torch.float32)
        
        # Network and optimizer
        self.model = PCNModel(
            self.obs_dim, self.act_dim, self.objectives, 
            self.scaling_factor, hidden_sizes
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.replay_buffer = PrioritizedExperienceReplay(
            max_size=buffer_capacity, gamma=gamma
        )
        
        # Current desired return and horizon for evaluation
        self.desired_return = None
        self.desired_horizon = None
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.action_low = self.action_low.to(self.device)
        self.action_high = self.action_high.to(self.device)
        
        # Tensorboard writer
        self.writer = SummaryWriter(writer_filename)
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
    
    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Predict action for given observation using current desired return/horizon.
        
        Args:
            obs: Observation array
            deterministic: Whether to add exploration noise
            
        Returns:
            Action array
        """
        if self.desired_return is None or self.desired_horizon is None:
            # If no desired return/horizon set, use random values for exploration
            self.desired_return = np.random.uniform(0, 10, self.objectives)
            self.desired_horizon = np.random.uniform(1, 100)
        
        # Convert to tensors
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return_tensor = torch.tensor(self.desired_return, dtype=torch.float32, device=self.device).unsqueeze(0)
        horizon_tensor = torch.tensor([[self.desired_horizon]], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            action = self.model(obs_tensor, return_tensor, horizon_tensor)
            action = action.squeeze(0).cpu().numpy()
        
        # Add exploration noise if not deterministic
        if not deterministic:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = action + noise
        
        # Clip to action bounds
        action = np.clip(action, self.action_low.cpu().numpy(), self.action_high.cpu().numpy())
        
        return action
    
    def _choose_desired_commands(self, num_episodes: int = 20) -> Tuple[np.ndarray, float]:
        """
        Choose desired return and horizon based on best episodes in buffer.
        
        Args:
            num_episodes: Number of best episodes to consider
            
        Returns:
            Tuple of (desired_return, desired_horizon)
        """
        if len(self.replay_buffer) == 0:
            # Random initialization if no episodes yet
            desired_return = np.random.uniform(0, 10, self.objectives).astype(np.float32)
            desired_horizon = float(np.random.uniform(10, self.max_steps))
            return desired_return, desired_horizon
        
        # Get best episodes based on crowding distance
        best_episodes = self.replay_buffer.get_best_episodes(min(num_episodes, len(self.replay_buffer)))
        
        # Extract returns and horizons from best episodes
        returns = np.array([episode[0].reward for episode in best_episodes])
        horizons = np.array([len(episode) for episode in best_episodes])
        
        # Filter to non-dominated solutions only
        if len(returns) > 1:
            non_dominated_mask = get_non_dominated_indices(returns)
            returns = returns[non_dominated_mask]
            horizons = horizons[non_dominated_mask]
        
        if len(returns) == 0:
            # Fallback if no non-dominated solutions
            desired_return = np.random.uniform(0, 10, self.objectives).astype(np.float32)
            desired_horizon = float(np.random.uniform(10, self.max_steps))
            return desired_return, desired_horizon
        
        # Select random point from non-dominated set
        idx = np.random.randint(0, len(returns))
        base_return = returns[idx].copy()
        base_horizon = horizons[idx]
        
        # Try to improve on the selected return by adding small random amounts
        # This encourages the agent to try to do better than past performance
        mean_return = np.mean(returns, axis=0)
        std_return = np.std(returns, axis=0) + 1e-6  # Avoid division by zero
        
        desired_return = base_return.copy()
        # Randomly choose one objective to improve
        obj_to_improve = np.random.randint(0, self.objectives)
        improvement = np.random.uniform(0, std_return[obj_to_improve])
        desired_return[obj_to_improve] += improvement
        
        # Set desired horizon slightly less than the original to encourage efficiency
        desired_horizon = float(max(base_horizon - 2, 1.0))
        
        return desired_return.astype(np.float32), desired_horizon
    
    def _run_episode(self, env: gym.Env, desired_return: np.ndarray, desired_horizon: float, 
                    max_return: Optional[np.ndarray] = None) -> List[Transition]:
        """
        Run a single episode with given desired return and horizon.
        
        Args:
            env: Environment to run episode in
            desired_return: Target return vector
            desired_horizon: Target episode length
            max_return: Maximum return for clipping (if None, no clipping)
            
        Returns:
            List of transitions
        """
        transitions = []
        obs, _ = env.reset()
        done = False
        
        # Set current desired commands for action prediction
        current_return = desired_return.copy()
        current_horizon = desired_horizon
        
        while not done and len(transitions) < self.max_steps:
            # Update desired commands for action selection
            self.desired_return = current_return
            self.desired_horizon = current_horizon
            
            # Take action
            action = self.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            transitions.append(Transition(
                observation=obs.copy(),
                action=action.copy(),
                reward=reward.copy().astype(np.float32),
                next_observation=next_obs.copy(),
                done=terminated
            ))
            
            obs = next_obs
            self.global_step += 1
            
            # Update remaining desired return and horizon
            current_return = current_return - reward
            if max_return is not None:
                current_return = np.clip(current_return, None, max_return)
            current_horizon = max(current_horizon - 1, 1.0)
        
        return transitions
    
    def update(self) -> float:
        """
        Update the PCN model using a batch of experience.
        
        Returns:
            Training loss value
        """
        if len(self.replay_buffer) == 0:
            return 0.0
        
        # Sample batch of experience
        batch = self.replay_buffer.sample_batch(self.batch_size)
        if len(batch) == 0:
            return 0.0
        
        # Unpack batch
        states, actions, desired_returns, desired_horizons = zip(*batch)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        desired_returns = torch.tensor(np.array(desired_returns), dtype=torch.float32, device=self.device)
        desired_horizons = torch.tensor(np.array(desired_horizons), dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # Forward pass
        predicted_actions = self.model(states, desired_returns, desired_horizons)
        
        # Compute loss (MSE between predicted and actual actions)
        loss = F.mse_loss(predicted_actions, actions)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, env: gym.Env, num_episodes: int):
        """
        Train the PCN agent.
        
        Args:
            env: Environment to train in
            num_episodes: Number of episodes to train for
        """
        print(f"Starting PCN training for {num_episodes} episodes...")
        
        # Warm-up phase: collect random episodes
        warmup_episodes = min(20, num_episodes // 4)
        print(f"Warmup phase: collecting {warmup_episodes} random episodes...")
        
        for ep in range(warmup_episodes):
            # Random desired commands for exploration
            desired_return = np.random.uniform(0, 20, self.objectives).astype(np.float32)
            desired_horizon = float(np.random.uniform(10, self.max_steps))
            
            transitions = self._run_episode(env, desired_return, desired_horizon)
            self.replay_buffer.add_episode(transitions, self.global_step)
            self.episode_count += 1
            
            # Compute episode return for logging
            ep_return = transitions[0].reward  # Already computed as cumulative return
            
            if self.verbose or ep % 5 == 0:
                print(f"Warmup episode {ep + 1}: Steps={len(transitions)}, Return={ep_return}")
            
            # Log to tensorboard
            for i, r in enumerate(ep_return):
                self.writer.add_scalar(f'Warmup/Return_Obj_{i}', r, ep)
        
        print("Warmup complete. Starting main training...")
        
        # Main training loop
        update_count = 0
        
        for ep in range(warmup_episodes, num_episodes):
            # Choose desired commands based on best past episodes
            desired_return, desired_horizon = self._choose_desired_commands()
            
            # Run episode
            transitions = self._run_episode(env, desired_return, desired_horizon)
            self.replay_buffer.add_episode(transitions, self.global_step)
            self.episode_count += 1
            
            # Compute episode return for logging
            ep_return = transitions[0].reward
            
            # Train model multiple times per episode
            if self.global_step % self.training_frequency == 0 and len(self.replay_buffer) > 0:
                losses = []
                num_updates = max(1, len(transitions) // 10)  # Scale updates with episode length
                
                for _ in range(num_updates):
                    loss = self.update()
                    losses.append(loss)
                    update_count += 1
                
                avg_loss = np.mean(losses) if losses else 0.0
                
                # Log training metrics
                self.writer.add_scalar('Train/Loss', avg_loss, update_count)
                
                if self.verbose:
                    print(f"Update {update_count}: Loss={avg_loss:.4f}")
            
            # Logging
            for i, r in enumerate(ep_return):
                self.writer.add_scalar(f'Train/Return_Obj_{i}', r, ep)
            
            self.writer.add_scalar('Train/Episode_Length', len(transitions), ep)
            self.writer.add_scalar('Train/Total_Return', ep_return.sum(), ep)
            self.writer.add_scalar('Train/Desired_Return_Distance', 
                                 np.linalg.norm(ep_return - desired_return), ep)
            
            # Progress reporting
            if ep % 50 == 0 or self.verbose:
                print(f"Episode {ep + 1}: Steps={len(transitions)}, Return={ep_return}, "
                      f"Desired={desired_return}, Buffer size={len(self.replay_buffer)}")
            
            # Flush writer periodically
            if ep % 50 == 0:
                self.writer.flush()
        
        print(f"Training completed! Total episodes: {num_episodes}, Total updates: {update_count}")
        self.writer.flush()
    
    def evaluate(self, env: gym.Env, episodes: int = 10, num_eval_points: int = 10) -> np.ndarray:
        """
        Evaluate the agent by running episodes with different desired returns.
        
        Args:
            env: Environment to evaluate in
            episodes: Number of episodes per evaluation point
            num_eval_points: Number of different desired returns to evaluate
            
        Returns:
            Array of achieved returns for each evaluation point
        """
        if len(self.replay_buffer) == 0:
            print("Warning: No episodes in buffer for evaluation")
            return np.array([])
        
        # Get diverse evaluation points from buffer
        best_episodes = self.replay_buffer.get_best_episodes(num_eval_points)
        eval_returns = []
        
        for episode in best_episodes:
            episode_returns = []
            
            # Extract desired return and horizon from this episode
            desired_return = episode[0].reward
            desired_horizon = float(len(episode))
            
            for _ in range(episodes):
                transitions = self._run_episode(env, desired_return, desired_horizon)
                achieved_return = transitions[0].reward
                episode_returns.append(achieved_return)
            
            # Average over episodes for this evaluation point
            avg_return = np.mean(episode_returns, axis=0)
            eval_returns.append(avg_return)
        
        return np.array(eval_returns)
    
    def set_desired_return_and_horizon(self, desired_return: np.ndarray, desired_horizon: float):
        """Set desired return and horizon for subsequent action predictions."""
        self.desired_return = desired_return.copy()
        self.desired_horizon = desired_horizon
    
    def save(self, filepath: str):
        """Save the PCN model weights."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'objectives': self.objectives,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'scaling_factor': self.scaling_factor,
            'global_step': self.global_step,
            'episode_count': self.episode_count
        }
        
        torch.save(save_dict, filepath)
        if self.verbose:
            print(f"PCN model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load PCN model weights."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'global_step' in checkpoint:
                self.global_step = checkpoint['global_step']
            if 'episode_count' in checkpoint:
                self.episode_count = checkpoint['episode_count']
            
            if self.verbose:
                print(f"PCN model loaded from {filepath}")
                
        except FileNotFoundError:
            raise FileNotFoundError(f"No model file found at {filepath}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {filepath}: {str(e)}")