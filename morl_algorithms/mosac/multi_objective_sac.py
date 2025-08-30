"""
Multi-Objective SAC (SAC-D template) with optional CAGrad (SAC-D-CAGrad).

Major changes vs. previous file:
1) Twin-critics with composite-min selection (Alg. 1, lines 4–6)
   - BEFORE: one critic per objective, one target set; element-wise target without twin-min.
   - NOW   : two critic networks (j=1,2), each with (m+1) heads; pick a single target
             network j* by the minimum *composite* Q over all heads (incl. entropy head),
             then use *all* its heads for bootstrapping.
   - WHY   : Paper shows element-wise min is suboptimal; use network with min composite Q.
             Proof & statement: “use all the predictions from the network with the minimum
             composite Q-function (Alg. 1, lines 5–6).”
             Ref: 2206.13901v2.pdf, Sec. 3.2 “Twin-network minimums in value decomposition”,
             lines 120–128.  fileciteturn3file1L120-L128

2) Entropy as an extra (m+1)-th head (Alg. 1, lines 3–7)
   - BEFORE: entropy handled via Normal.entropy() in actor loss; not part of critics/targets.
   - NOW   : critics have an extra head for the “entropy reward”; we extend reward with
             r_{m+1} = γ α log π(a′|s′); targets include this head; actor loss sums over m+1 heads.
   - WHY   : Matches Algorithm 1 lines 3, 7 and weight vector w ∈ R^{m+1}.
             Ref: 2206.13901v2.pdf, Algorithm 1 lines 35–66.  fileciteturn3file3L35-L66

3) Actor loss uses squashed log-prob and twin-network composite min (Alg. 1, line 7)
   - BEFORE: π loss used mean of heads and Normal.entropy() of pre-tanh Normal (incorrect)
   - NOW   : Lπ = E[ α log π(u|s) − min_{j∈{1,2}} Σ_{i=1..m+1} w_i Q_i(s,u;θ_j) ]
             with correct tanh log-prob correction.
   - WHY   : Matches Algorithm 1 line 7; squashed Gaussian details in App. C: actions are
             tanh-squashed; evaluation uses tanh(mean); scale to env bounds if needed.
             Ref: 2206.13901v2.pdf, Alg. 1 line 62–65; App. C lines 1–8. 
             fileciteturn3file3L62-L66 fileciteturn3file0L1-L8

4) Correct squashed Gaussian log π(a|s)
   - BEFORE: used Normal.entropy() (pre-tanh); no tanh correction.
   - NOW   : log π(a|s) computed as sum(Normal.log_prob(z)) − Σ log(1 − tanh(z)^2 + ε)
             where a = tanh(z). (Standard SAC practice.)

5) Deterministic evaluation uses tanh(mean) (and scaling to env bounds)
   - BEFORE: returned unsquashed mean that can exceed bounds.
   - NOW   : deterministic action = scale(tanh(mean)) as in App. C.
             Ref: 2206.13901v2.pdf, App. C lines 1–8.  fileciteturn3file0L1-L8

6) Loss aggregation per Algorithm 1, line 6 & 11
   - We average L_Q across heads (1/(m+1) Σ_i L_Qi) and sum across two critics with 1/2 factor.

Optional detail from App. C (we keep the behavior but do not register per-layer hooks):
- Paper divides hidden-layer gradients by reward dimension for SAC-D/Naive (App. C, line 10–15).
  fileciteturn3file0L10-L15
- Here, we scale the *critic loss* as Alg. 1 suggests; if you want per-layer gradient scaling,
  add backward hooks that divide grads by (m+1).

This file is self-contained and keeps the class name MOSAC for drop-in replacement.

CAGrad usage:
- Set use_cagrad=True at init to enable SAC-D-CAGrad (Alg. 1, line 8–12).
- Hyperparameter cagrad_c controls the constraint (c=0 -> GD, large -> MGDA) per CAGrad Sec. 3.1 (arXiv:2110.14048).
- Our implementation solves the dual approximately via a penalized QP with projected gradient onto the simplex.
"""

from collections import deque
import random
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# ==============================
# Utilities & Building Blocks
# ==============================

def _mlp(sizes: List[int], activation=nn.ReLU, out_act=None) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes) - 2:
            layers.append(activation())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)


def _project_simplex(v: torch.Tensor) -> torch.Tensor:
        """Project v onto the probability simplex."""
        if v.ndim > 1:
            v = v.squeeze()
        n = v.shape[0]
        u, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(u, dim=0) - 1
        ind = torch.arange(1, n + 1, device=v.device)
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / rho
        w = torch.clamp(v - theta, min=0)
        return w


def _orthogonal_init(module: nn.Module, gain: float = 1.0):
    """Apply orthogonal initialization to Linear layers in a module."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Sequential):
        for layer in module:
            _orthogonal_init(layer, gain)
    elif isinstance(module, nn.ModuleList):
        for layer in module:
            _orthogonal_init(layer, gain)


def cagrad_weights(G: torch.Tensor, c: float, iters: int = 200, lr: float = 0.05) -> torch.Tensor:
    """
    Compute CAGrad weights for gradient matrix G (flattened grads as columns).
    Args:
        G: tensor of shape (d, m) where m = #tasks (heads)
        c: CAGrad's hyperparameter
        iters: projected gradient steps
        lr: learning rate for projected gradient solver
    Returns:
        w: tensor of shape (m,)
    """
    GG = G.t() @ G   # m x m Gram matrix
    m = GG.shape[0]
    w = torch.ones(m, device=G.device) / m
    for _ in range(iters):
        # Gradient of objective: maximize min_w G norm^2 subject to ||Gw|| <= (1 + c)g0
        gw = GG @ w
        g0_norm = torch.sqrt((gw * w).sum()).detach()
        grad_norm = torch.sqrt((w @ GG) @ w + 1e-8)
        obj_grad = gw / grad_norm - (1 + c) * gw / g0_norm
        w = _project_simplex(w + lr * obj_grad)
    return w




class CriticHead(nn.Module):
    """Deprecated: replaced by shared-backbone TwinCritics. Kept for backward-compat."""
    def __init__(self, in_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = _mlp([in_dim, *hidden_sizes, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwinCritics(nn.Module):
    """One critic network with a shared backbone and (m+1) scalar heads.
    SAC-D uses two such networks (j=1,2). A shared trunk is essential so that
    CAGrad can mitigate gradient conflicts across heads.
    """
    def __init__(self, sa_dim: int, num_heads: int, hidden_sizes=(256, 256), 
                 use_orthogonal_init: bool = False, orthogonal_gain: float = 1.0):
        super().__init__()
        # Shared backbone
        layers = []
        last = sa_dim
        for hs in hidden_sizes:
            layers.append(nn.Linear(last, hs))
            layers.append(nn.ReLU())
            last = hs
        self.backbone = nn.Sequential(*layers)
        # Per-head linear outputs from the shared representation
        self.heads = nn.ModuleList([nn.Linear(last, 1) for _ in range(num_heads)])
        
        # Apply orthogonal initialization if requested
        if use_orthogonal_init:
            _orthogonal_init(self.backbone, gain=orthogonal_gain)
            _orthogonal_init(self.heads, gain=orthogonal_gain)

    def forward(self, sa: torch.Tensor) -> torch.Tensor:
        h = self.backbone(sa)
        qs = [head(h) for head in self.heads]  # list of (B,1)
        return torch.cat(qs, dim=-1)           # (B, num_heads)




class Actor(nn.Module):
    """Gaussian policy with Tanh squashing (outputs mean and log_std)."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256),
                 use_orthogonal_init: bool = False, orthogonal_gain: float = 1.0):
        super().__init__()
        self.net = _mlp([obs_dim, *hidden_sizes, 2*act_dim])
        self.act_dim = act_dim
        
        # Apply orthogonal initialization if requested
        if use_orthogonal_init:
            _orthogonal_init(self.net, gain=orthogonal_gain)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_logstd = self.net(obs)
        mu, log_std = mu_logstd.split(self.act_dim, dim=-1)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        return mu, log_std


class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]):
        # (obs, act, reward_vec(m,), next_obs, done)
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, acts, rews, next_obs, dones = map(np.stack, zip(*batch))
        return obs, acts, rews, next_obs, dones

    def __len__(self):
        return len(self.buffer)


# ==============================
# MOSAC (SAC-D without CAGrad)
# ==============================

class MOSAC:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_low: np.ndarray,
        act_high: np.ndarray,
        objectives: int,
        hidden_sizes=(256, 256),
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha=0.2,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        buffer_capacity=1_000_000,
        max_steps_per_episode=1000,
        training_frequency = 1000,
        writer_filename='mosac_runs',
        reward_weights: List[float] = None,  # length m+1, default all ones,
        use_cagrad: bool = False,
        cagrad_c: float = 0.4,
        cagrad_iters: int = 200,
        cagrad_lr: float = 0.05,
        # New parameters for orthogonal initialization
        use_orthogonal_init: bool = False,
        orthogonal_gain: float = 1.0,
        # New parameters for learning rate annealing
        use_lr_annealing: bool = False,
        lr_anneal_factor: float = 0.1,
        lr_anneal_step_size: int = 100000,  # steps at which to anneal
        lr_anneal_type: str = 'step',  # 'step', 'exponential', or 'cosine'
        verbose=False
    ):
        """
        Implements Algorithm 1 (SAC-D / SAC-D-CAGrad) from 2206.13901 (with optional CAGrad).

        Key features:
        - Twin critics (two networks), each with (m+1) heads (m reward components + entropy head).
        - Composite-min selection to pick *one* target network for all heads.
        - Actor loss uses α log π(u|s) minus min over twin composite-Q.
        - Squashed Gaussian with correct tanh log-prob; deterministic = tanh(mean) (then scaled).
        - Optional orthogonal initialization for improved training stability.
        - Optional learning rate annealing for fine-tuned convergence.

        Citations to the paper are included in the header block above.
        """
        self.objectives = int(objectives)
        self.num_heads = self.objectives + 1  # +1 for entropy head
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_steps = max_steps_per_episode
        self.training_frequency = training_frequency
        self.verbose = verbose
        self.global_step = 0

        # CAGrad flags (Alg. 1 line 8–12)
        self.use_cagrad = bool(use_cagrad)
        self.cagrad_c = float(cagrad_c)
        self.cagrad_iters = int(cagrad_iters)
        self.cagrad_lr = float(cagrad_lr)
        
        # Orthogonal initialization parameters
        self.use_orthogonal_init = bool(use_orthogonal_init)
        self.orthogonal_gain = float(orthogonal_gain)
        
        # Learning rate annealing parameters
        self.use_lr_annealing = bool(use_lr_annealing)
        self.lr_anneal_factor = float(lr_anneal_factor)
        self.lr_anneal_step_size = int(lr_anneal_step_size)
        self.lr_anneal_type = str(lr_anneal_type)
        self.initial_actor_lr = float(actor_lr)
        self.initial_critic_lr = float(critic_lr)

        # Reward weights w ∈ R^{m+1}; default all ones
        if reward_weights is None:
            self.w = torch.ones(self.num_heads)
        else:
            assert len(reward_weights) == self.num_heads, "reward_weights must have length m+1"
            self.w = torch.tensor(reward_weights, dtype=torch.float32)

        # Action scaling to env bounds
        self._act_low = torch.as_tensor(act_low, dtype=torch.float32)
        self._act_high = torch.as_tensor(act_high, dtype=torch.float32)
        self._act_scale = (self._act_high - self._act_low) / 2.0
        self._act_bias = (self._act_high + self._act_low) / 2.0

        # Networks
        self.actor = Actor(self.obs_dim, self.act_dim, hidden_sizes, 
                          self.use_orthogonal_init, self.orthogonal_gain)
        sa_dim = self.obs_dim + self.act_dim
        self.critic1 = TwinCritics(sa_dim, self.num_heads, hidden_sizes,
                                  self.use_orthogonal_init, self.orthogonal_gain)
        self.critic2 = TwinCritics(sa_dim, self.num_heads, hidden_sizes,
                                  self.use_orthogonal_init, self.orthogonal_gain)
        self.target_critic1 = TwinCritics(sa_dim, self.num_heads, hidden_sizes,
                                         self.use_orthogonal_init, self.orthogonal_gain)
        self.target_critic2 = TwinCritics(sa_dim, self.num_heads, hidden_sizes,
                                         self.use_orthogonal_init, self.orthogonal_gain)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # Learning rate schedulers (if enabled)
        self.actor_scheduler = None
        self.critic1_scheduler = None
        self.critic2_scheduler = None
        
        if self.use_lr_annealing:
            if self.lr_anneal_type == 'step':
                self.actor_scheduler = optim.lr_scheduler.StepLR(
                    self.actor_opt, step_size=self.lr_anneal_step_size, 
                    gamma=self.lr_anneal_factor)
                self.critic1_scheduler = optim.lr_scheduler.StepLR(
                    self.critic1_opt, step_size=self.lr_anneal_step_size, 
                    gamma=self.lr_anneal_factor)
                self.critic2_scheduler = optim.lr_scheduler.StepLR(
                    self.critic2_opt, step_size=self.lr_anneal_step_size, 
                    gamma=self.lr_anneal_factor)
            elif self.lr_anneal_type == 'exponential':
                self.actor_scheduler = optim.lr_scheduler.ExponentialLR(
                    self.actor_opt, gamma=self.lr_anneal_factor)
                self.critic1_scheduler = optim.lr_scheduler.ExponentialLR(
                    self.critic1_opt, gamma=self.lr_anneal_factor)
                self.critic2_scheduler = optim.lr_scheduler.ExponentialLR(
                    self.critic2_opt, gamma=self.lr_anneal_factor)
            elif self.lr_anneal_type == 'cosine':
                self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.actor_opt, T_max=self.lr_anneal_step_size)
                self.critic1_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.critic1_opt, T_max=self.lr_anneal_step_size)
                self.critic2_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.critic2_opt, T_max=self.lr_anneal_step_size)

        # Replay buffer and TensorBoard writer
        self.buffer = ReplayBuffer(buffer_capacity)
        self.writer = SummaryWriter(writer_filename)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._to(self.device)

    def _to(self, device):
        self.actor.to(device)
        self.critic1.to(device); self.critic2.to(device)
        self.target_critic1.to(device); self.target_critic2.to(device)
        self.w = self.w.to(device)
        self._act_low = self._act_low.to(device)
        self._act_high = self._act_high.to(device)
        self._act_scale = self._act_scale.to(device)
        self._act_bias = self._act_bias.to(device)

    # ---------- Policy sampling utilities ----------

    def _squash(self, u: torch.Tensor) -> torch.Tensor:
        return torch.tanh(u)

    def _unsquashed_gaussian(self, obs: torch.Tensor):
        mu, log_std = self.actor(obs)
        std = log_std.exp()
        return mu, std

    def _sample_action_and_logp(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          a_scaled: scaled action in env domain (B, act_dim)
          z: pre-tanh sample (B, act_dim)
          logp: log π(a|s) with tanh correction (B,)
        """
        mu, std = self._unsquashed_gaussian(obs)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()                    # reparameterized
        a = self._squash(z)
        # Tanh log-prob correction: sum over action dims
        logp = dist.log_prob(z).sum(dim=-1) - torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1)
        # Scale to environment action bounds
        a_scaled = self._act_bias + self._act_scale * a  # linear scaling
        return a_scaled, z, logp

    def _deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _ = self._unsquashed_gaussian(obs)
        a = self._squash(mu)
        a_scaled = self._act_bias + self._act_scale * a
        return a_scaled

    def predict(self, obs: np.ndarray | torch.Tensor, deterministic=False) -> np.ndarray:
        if isinstance(obs, np.ndarray):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_t = obs.to(self.device)
        with torch.no_grad():
            if deterministic:
                a = self._deterministic_action(obs_t.unsqueeze(0)).squeeze(0)
            else:
                a, _, _ = self._sample_action_and_logp(obs_t.unsqueeze(0))
                a = a.squeeze(0)
        return a.cpu().numpy()

    # ---------- Update ----------

    def update(self, step: int):
        if len(self.buffer) < self.batch_size:
            if self.verbose and step % 100 == 0:
                print(f"Skipping update {step}: buffer size {len(self.buffer)} < batch size {self.batch_size}")
            return
            
        try:
            obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        except Exception as e:
            print(f"Failed to sample from buffer at step {step}: {e}")
            return

        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)  # (B, m)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)  # (B,1)

        # Debug: Check for NaN values
        if torch.isnan(obs).any() or torch.isnan(actions).any() or torch.isnan(rewards).any():
            print(f"Warning: NaN values detected in batch at step {step}")
            return

        # (1) Compute next action and log prob
        with torch.no_grad():
            next_actions, _, logp_next = self._sample_action_and_logp(next_obs)  # (B, act), (B,)
            sa_next = torch.cat([next_obs, next_actions], dim=-1)                # (B, sa_dim)

            # (2) Evaluate target critics on next (s', a')
            q1_next = self.target_critic1(sa_next)   # (B, m+1)
            q2_next = self.target_critic2(sa_next)   # (B, m+1)

            # (3) Entropy reward component r_{m+1} = γ α log π(a'|s') (Alg.1 line 3)
            r_entropy = (self.gamma * self.alpha * logp_next).unsqueeze(-1)      # (B,1)

            # (4) Build extended reward vector r_ext = [r_1..r_m, r_{m+1}] (B, m+1)
            r_ext = torch.cat([rewards, r_entropy], dim=-1)                      # (B, m+1)

            # (5) Choose target network j* via minimum *composite* Q (Alg.1 line 4)
            # composite_j = sum_i w_i * Q_i^{target_j}(s',a')
            comp1 = (q1_next * self.w).sum(dim=-1, keepdim=True)  # (B,1)
            comp2 = (q2_next * self.w).sum(dim=-1, keepdim=True)  # (B,1)
            use_q1 = (comp1 <= comp2)                              # (B,1) boolean
            q_next_chosen = torch.where(use_q1, q1_next, q2_next)  # (B, m+1)

            # (6) Targets yi = ri + γ (1-done) * Q_i^{target_j*}(s',a') (Alg.1 line 5)
            y = r_ext + self.gamma * (1.0 - dones) * q_next_chosen # (B, m+1)

        # (7) Critic updates: L_Qi = (1/2) * sum_j (Qi(s,a;θ_j) - y_i)^2  (Alg.1 line 6)
        sa = torch.cat([obs, actions], dim=-1)  # (B, sa_dim)

        q1_pred = self.critic1(sa)  # (B, m+1)
        q2_pred = self.critic2(sa)  # (B, m+1)

        
        # Per-head MSE for both critics
        lq1 = F.mse_loss(q1_pred, y, reduction='none').mean(dim=0)  # (m+1,)
        lq2 = F.mse_loss(q2_pred, y, reduction='none').mean(dim=0)  # (m+1,)

        self.critic1_opt.zero_grad()
        self.critic2_opt.zero_grad()

        if self.use_cagrad:
            # Build per-head losses that include BOTH critics (Alg. 1 updates all θ)
            per_head_losses = [0.5 * (lq1[i] + lq2[i]) for i in range(self.num_heads)]

            # Collect flattened grads per head
            params = list(self.critic1.parameters()) + list(self.critic2.parameters())
            grads = []
            for i, li in enumerate(per_head_losses):
                # Zero gradients before computing each head's gradient
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()
                
                # Compute gradient for this head
                gi = torch.autograd.grad(li, params, retain_graph=True, create_graph=False, allow_unused=False)
                gi_flat = torch.cat([g.contiguous().view(-1) for g in gi if g is not None])
                grads.append(gi_flat)
            
            if len(grads) == self.num_heads:
                G = torch.stack(grads, dim=1)  # (P, m+1)

                # Solve for CAGrad simplex weights
                with torch.no_grad():
                    w = cagrad_weights(G, c=self.cagrad_c, iters=self.cagrad_iters, lr=self.cagrad_lr)

                # Backprop weighted sum: Σ_i w_i ∇L_i
                weighted_loss = sum(w[i] * per_head_losses[i] for i in range(self.num_heads))
                lq_total = weighted_loss.detach()
                weighted_loss.backward()
            else:
                # Fallback to standard averaging if gradient collection fails
                lq_total = 0.5 * (lq1.mean() + lq2.mean())
                lq_total.backward()
        else:
            # Aggregate per Alg. 1: average across heads, and average across the two critics
            lq_total = 0.5 * (lq1.mean() + lq2.mean())
            lq_total.backward()

        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=5.0)
        self.critic1_opt.step()
        self.critic2_opt.step()

        # (8) Actor update (Alg.1 line 7): Lπ = α log π(u|s) − min_j Σ_i w_i Q_i(s,u;θ_j)
        self.actor_opt.zero_grad()
        u_scaled, z, logp = self._sample_action_and_logp(obs)          # (B, act), (B,), reparam
        sau = torch.cat([obs, u_scaled], dim=-1)
        q1_u = self.critic1(sau)                                       # (B, m+1)
        q2_u = self.critic2(sau)                                       # (B, m+1)
        comp1_u = (q1_u * self.w).sum(dim=-1)                          # (B,)
        comp2_u = (q2_u * self.w).sum(dim=-1)                          # (B,)
        comp_min = torch.minimum(comp1_u, comp2_u)                     # (B,)
        actor_loss = (self.alpha * logp - comp_min).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
        self.actor_opt.step()

        # (9) Soft update target networks (Alg.1 line 14)
        with torch.no_grad():
            for tp, p in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
            for tp, p in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        # (10) Logging
        self.writer.add_scalar('Loss/Actor', actor_loss.item(), step)
        self.writer.add_scalar('Loss/Critic', lq_total.item(), step)
        
        # (11) Learning rate annealing (if enabled)
        if self.use_lr_annealing:
            if self.actor_scheduler is not None:
                self.actor_scheduler.step()
            if self.critic1_scheduler is not None:
                self.critic1_scheduler.step()
            if self.critic2_scheduler is not None:
                self.critic2_scheduler.step()
            
            # Log learning rates
            if step % 100 == 0:  # Log every 100 steps to avoid spam
                self.writer.add_scalar('LR/Actor', self.actor_opt.param_groups[0]['lr'], step)
                self.writer.add_scalar('LR/Critic1', self.critic1_opt.param_groups[0]['lr'], step)
                self.writer.add_scalar('LR/Critic2', self.critic2_opt.param_groups[0]['lr'], step)

    # ---------- Training Loop ----------

    def train(self, env: gym.Env, num_episodes: int):
        step = 0
        total_steps = 0
        warmup_steps = max(self.batch_size * 2, 1000)  # Warmup period before training starts
        
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Warmup steps: {warmup_steps}")
        
        # Log initialization and annealing settings
        if self.use_orthogonal_init:
            print(f"Using orthogonal initialization with gain: {self.orthogonal_gain}")
        if self.use_lr_annealing:
            print(f"Using learning rate annealing: {self.lr_anneal_type}")
            print(f"  Initial actor LR: {self.initial_actor_lr}")
            print(f"  Initial critic LR: {self.initial_critic_lr}")
            print(f"  Anneal factor: {self.lr_anneal_factor}")
            print(f"  Anneal step size: {self.lr_anneal_step_size}")
        
        for ep in range(num_episodes):
            obs, _ = env.reset()
            ep_reward = np.zeros(self.objectives, dtype=np.float32)
            ep_steps = 0
            
            for t in range(self.max_steps):
                action = self.predict(obs, deterministic=False)
                next_obs, reward_vec, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)

                total_steps += 1
                ep_steps += 1

                self.buffer.add((obs, action, reward_vec, next_obs, done))
                ep_reward += reward_vec

                obs = next_obs

                # Start training after warmup and train frequently
                if len(self.buffer) >= warmup_steps and total_steps % self.training_frequency == 0:
                    try:
                        self.update(step)
                        step += 1
                    except Exception as e:
                        print(f"Update failed at step {step}: {e}")
                        if self.verbose:
                            import traceback
                            traceback.print_exc()

                if done:
                    break

            # Log every episode (not just every 100)
            for i, r in enumerate(ep_reward):
                self.writer.add_scalar(f'Reward/Objective_{i}', r, ep)
            
            self.writer.add_scalar('Episode/Length', ep_steps, ep)
            self.writer.add_scalar('Episode/Total_Reward', ep_reward.sum(), ep)
            
            # Verbose logging every 50 episodes
            if ep % 50 == 0 or self.verbose:
                print(f"Episode {ep}: Steps={ep_steps}, Rewards={ep_reward}, Buffer size={len(self.buffer)}, Updates={step}")
            
            # Flush writer periodically
            if ep % 50 == 0:
                self.writer.flush()
                
        print(f"Training completed! Total episodes: {ep + 1}, Total updates: {step}")
        self.writer.flush()  # Final flush

    def evaluate(self, env: gym.Env, episodes=10):
        rewards = []
        for _ in range(episodes):
            obs, _ = env.reset()
            ep_reward = np.zeros(self.objectives, dtype=np.float32)
            for _ in range(self.max_steps):
                action = self.predict(obs, deterministic=True)
                obs, reward_vec, terminated, truncated, _ = env.step(action)
                ep_reward += reward_vec
                if terminated or truncated:
                    break
            rewards.append(ep_reward)
        return np.mean(rewards, axis=0)

    def save(self, filepath: str):
        """
        Save only the neural network weights (state_dicts) to a file.
        Does not save optimizers, schedulers, or other training state.
        
        Args:
            filepath (str): Path to save the weights file (should end with .pth or .pt)
        """
        weights_dict = {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            # Save important hyperparameters needed for reconstruction
            'objectives': self.objectives,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'num_heads': self.num_heads,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'tau': self.tau,
            'w': self.w.cpu(),  # reward weights
        }
        
        torch.save(weights_dict, filepath)
        if self.verbose:
            print(f"MOSAC weights saved to {filepath}")

    def load(self, filepath: str):
        """
        Load neural network weights from a file.
        Only loads the network weights, not optimizers or training state.
        
        Args:
            filepath (str): Path to the weights file
        """
        try:
            weights_dict = torch.load(filepath, map_location=self.device)
            
            # Load network weights
            self.actor.load_state_dict(weights_dict['actor'])
            self.critic1.load_state_dict(weights_dict['critic1'])
            self.critic2.load_state_dict(weights_dict['critic2'])
            self.target_critic1.load_state_dict(weights_dict['target_critic1'])
            self.target_critic2.load_state_dict(weights_dict['target_critic2'])
            
            # Load important hyperparameters (optional, for verification)
            if 'w' in weights_dict:
                self.w = weights_dict['w'].to(self.device)
            if 'alpha' in weights_dict:
                self.alpha = weights_dict['alpha']
            if 'gamma' in weights_dict:
                self.gamma = weights_dict['gamma']
            if 'tau' in weights_dict:
                self.tau = weights_dict['tau']
                
            if self.verbose:
                print(f"MOSAC weights loaded from {filepath}")
                
        except FileNotFoundError:
            raise FileNotFoundError(f"No weights file found at {filepath}")
        except Exception as e:
            raise RuntimeError(f"Failed to load weights from {filepath}: {str(e)}")

    def save_checkpoint(self, filepath: str, episode: int = None, step: int = None):
        """
        Save a complete checkpoint including weights and training metadata.
        Useful for resuming training from a specific point.
        
        Args:
            filepath (str): Path to save the checkpoint file
            episode (int, optional): Current episode number
            step (int, optional): Current training step number
        """
        checkpoint = {
            # Network weights
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            
            # Optimizer states (for resuming training)
            'actor_optimizer': self.actor_opt.state_dict(),
            'critic1_optimizer': self.critic1_opt.state_dict(),
            'critic2_optimizer': self.critic2_opt.state_dict(),
            
            # Scheduler states (if using learning rate annealing)
            'actor_scheduler': self.actor_scheduler.state_dict() if self.actor_scheduler else None,
            'critic1_scheduler': self.critic1_scheduler.state_dict() if self.critic1_scheduler else None,
            'critic2_scheduler': self.critic2_scheduler.state_dict() if self.critic2_scheduler else None,
            
            # Training metadata
            'episode': episode,
            'step': step,
            'global_step': self.global_step,
            
            # Hyperparameters
            'objectives': self.objectives,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'num_heads': self.num_heads,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'tau': self.tau,
            'w': self.w.cpu(),
        }
        
        torch.save(checkpoint, filepath)
        if self.verbose:
            print(f"MOSAC checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str, load_optimizers: bool = True):
        """
        Load a complete checkpoint including weights and optionally optimizer states.
        
        Args:
            filepath (str): Path to the checkpoint file
            load_optimizers (bool): Whether to load optimizer states (for resuming training)
            
        Returns:
            dict: Training metadata (episode, step, etc.)
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Load network weights
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic1.load_state_dict(checkpoint['critic1'])
            self.critic2.load_state_dict(checkpoint['critic2'])
            self.target_critic1.load_state_dict(checkpoint['target_critic1'])
            self.target_critic2.load_state_dict(checkpoint['target_critic2'])
            
            # Load optimizer states if requested (for resuming training)
            if load_optimizers:
                if 'actor_optimizer' in checkpoint:
                    self.actor_opt.load_state_dict(checkpoint['actor_optimizer'])
                if 'critic1_optimizer' in checkpoint:
                    self.critic1_opt.load_state_dict(checkpoint['critic1_optimizer'])
                if 'critic2_optimizer' in checkpoint:
                    self.critic2_opt.load_state_dict(checkpoint['critic2_optimizer'])
                    
                # Load scheduler states if they exist
                if checkpoint.get('actor_scheduler') and self.actor_scheduler:
                    self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler'])
                if checkpoint.get('critic1_scheduler') and self.critic1_scheduler:
                    self.critic1_scheduler.load_state_dict(checkpoint['critic1_scheduler'])
                if checkpoint.get('critic2_scheduler') and self.critic2_scheduler:
                    self.critic2_scheduler.load_state_dict(checkpoint['critic2_scheduler'])
            
            # Load hyperparameters
            if 'w' in checkpoint:
                self.w = checkpoint['w'].to(self.device)
            if 'alpha' in checkpoint:
                self.alpha = checkpoint['alpha']
            if 'gamma' in checkpoint:
                self.gamma = checkpoint['gamma']
            if 'tau' in checkpoint:
                self.tau = checkpoint['tau']
            if 'global_step' in checkpoint:
                self.global_step = checkpoint['global_step']
                
            if self.verbose:
                print(f"MOSAC checkpoint loaded from {filepath}")
                
            # Return training metadata
            metadata = {
                'episode': checkpoint.get('episode'),
                'step': checkpoint.get('step'),
                'global_step': checkpoint.get('global_step', 0)
            }
            return metadata
            
        except FileNotFoundError:
            raise FileNotFoundError(f"No checkpoint file found at {filepath}")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {filepath}: {str(e)}")
