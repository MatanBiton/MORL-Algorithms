print("Starting imports...")
from morl_algorithms.mosac.multi_objective_sac import MOSAC
from EnergyNetMoISO.MoISOEnv import MultiObjectiveISOEnv
from stable_baselines3.ppo import PPO
from energy_net.env.pcs_unit_v0 import PCSUnitEnv
import os

print("Entered alternating training script")

# Training parameters
num_cycles = 5  # Number of alternating training cycles
iso_training_episodes = 50000  # Episodes per ISO training cycle
pcs_training_timesteps = 100000  # Timesteps per PCS training cycle

# Initialize ISO environment without PCS agent
print("Initializing ISO environment without PCS agent...")
iso_env = MultiObjectiveISOEnv()

# Initialize ISO agent
print("Initializing ISO agent...")
iso_agent = MOSAC(obs_dim=3, act_dim=2,
                  act_low=iso_env.action_space.low,
                  act_high=iso_env.action_space.high,
                  objectives=2,
                  hidden_sizes=(256,256,256),
                  training_frequency=48,
                  batch_size=512,
                  use_cagrad=True,
                  use_orthogonal_init=True,
                  use_lr_annealing=True,
                  lr_anneal_type="cosine")

# Initialize PCS agent (will be trained later)
pcs_agent = None

print("Starting alternating training...")

for cycle in range(num_cycles):
    print(f"\n=== Training Cycle {cycle + 1}/{num_cycles} ===")
    
    # Phase 1: Train ISO agent
    print(f"Phase 1: Training ISO agent for {iso_training_episodes} episodes...")
    
    if pcs_agent is not None:
        # Use trained PCS agent in ISO environment
        print("Using trained PCS agent in ISO environment...")
        iso_env_with_pcs = MultiObjectiveISOEnv(trained_pcs_model=pcs_agent)
        iso_agent.train(iso_env_with_pcs, iso_training_episodes)
    else:
        # First cycle: train ISO without PCS agent
        print("Training ISO agent without PCS agent (first cycle)...")
        iso_agent.train(iso_env, iso_training_episodes)
    
    # Save ISO agent after training
    iso_save_path = f"trained_iso_cycle_{cycle + 1}.pth"
    iso_agent.save(iso_save_path)
    print(f"ISO agent saved to {iso_save_path}")
    
    # Phase 2: Train PCS agent using current ISO agent
    print(f"Phase 2: Training PCS agent for {pcs_training_timesteps} timesteps...")
    
    # Create PCS environment with current ISO agent
    pcs_env = PCSUnitEnv(trained_iso_model_instance=iso_agent)
    
    # Initialize or update PCS agent
    if pcs_agent is None:
        # First cycle: create new PCS agent
        print("Creating new PCS agent...")
        pcs_agent = PPO(policy="MlpPolicy", 
                       env=pcs_env, 
                       learning_rate=0.1,
                       device='cpu',
                       n_epochs=10)
    else:
        # Update environment for existing PCS agent
        print("Updating PCS agent environment...")
        pcs_agent.set_env(pcs_env)
    
    # Train PCS agent
    pcs_agent.learn(total_timesteps=pcs_training_timesteps)
    
    # Save PCS agent after training
    pcs_save_path = f"trained_pcs_cycle_{cycle + 1}"
    pcs_agent.save(pcs_save_path)
    print(f"PCS agent saved to {pcs_save_path}")
    
    print(f"Cycle {cycle + 1} completed!")

# Final save with standard names
print("\nSaving final models...")
iso_agent.save("trained_iso.pth")
pcs_agent.save("trained_pcs")
print("Final models saved as 'trained_iso.pth' and 'trained_pcs'")

print("Alternating training completed!")
