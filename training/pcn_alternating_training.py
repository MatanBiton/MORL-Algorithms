print("Starting imports...")
from morl_algorithms.pcn.pareto_conditioned_networks import PCN
from EnergyNetMoISO.MoISOEnv import MultiObjectiveISOEnv
from stable_baselines3.ppo import PPO
from energy_net.env.pcs_unit_v0 import PCSUnitEnv
import os

print("Entered alternating training script")

# Training parameters
num_cycles = 5  # Number of alternating training cycles
iso_training_episodes = 100000  # Episodes per ISO training cycle
pcs_training_timesteps = 100000  # Timesteps per PCS training cycle

# Initialize ISO environment without PCS agent
print("Initializing ISO environment without PCS agent...")
iso_env = MultiObjectiveISOEnv()

# Initialize ISO agent (will be recreated for each cycle with unique tensorboard logging)
print("ISO agent will be initialized for each training cycle with unique tensorboard logging...")
iso_agent = None

# Initialize PCS agent (will be trained later)
pcs_agent = None

print("Starting alternating training...")

for cycle in range(num_cycles):
    print(f"\n=== Training Cycle {cycle + 1}/{num_cycles} ===")
    
    # Create new ISO agent for this cycle with unique tensorboard logging
    print(f"Initializing ISO agent for cycle {cycle + 1} with tensorboard logging...")
    tensorboard_filename = f"pcn_run_iter_{cycle + 1}"
    
    if iso_agent is not None:
        # Close previous tensorboard writer before creating new one
        iso_agent.writer.close()
    
    iso_agent = PCN(obs_dim=3, act_dim=2,
                    act_low=iso_env.action_space.low,
                    act_high=iso_env.action_space.high,
                    objectives=2,
                    hidden_sizes=(256,256,256),
                    training_frequency=48,
                    batch_size=512,
                    learning_rate=3e-4,
                    gamma=0.99,
                    buffer_capacity=100,
                    max_steps_per_episode=1000,
                    noise_std=0.1,
                    writer_filename=tensorboard_filename,
                    verbose=True)
    print(f"ISO agent initialized with tensorboard logging to: {tensorboard_filename}")
    
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
    iso_save_path = f"pcn_trained_iso_cycle_{cycle + 1}.pth"
    iso_agent.save(iso_save_path)
    print(f"ISO agent saved to {iso_save_path}")
    
    # Close tensorboard writer for this cycle
    iso_agent.writer.close()
    print(f"Tensorboard logging for cycle {cycle + 1} closed")
    
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
    pcs_save_path = f"pcn_trained_pcs_cycle_{cycle + 1}"
    pcs_agent.save(pcs_save_path)
    print(f"PCS agent saved to {pcs_save_path}")
    
    print(f"Cycle {cycle + 1} completed!")

# Final save with standard names
print("\nSaving final models...")
iso_agent.save("pcn_trained_iso.pth")
pcs_agent.save("pcn_trained_pcs")

# Ensure final tensorboard writer is closed
if iso_agent and hasattr(iso_agent, 'writer') and iso_agent.writer:
    iso_agent.writer.close()
    print("Final tensorboard writer closed")

print("Final models saved as 'trained_iso.pth' and 'trained_pcs'")

print("Alternating training completed!")
