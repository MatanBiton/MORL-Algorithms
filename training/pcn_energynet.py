print("Starting imports...")
from morl_algorithms.pcn.pareto_conditioned_networks import PCN
from EnergyNetMoISO.MoISOEnv import MultiObjectiveISOEnv
import numpy as np

print("Entered PCN training script")

# Create environment
env = MultiObjectiveISOEnv()

# Create PCN agent with configuration similar to MOSAC
agent = PCN(
    env=env,
    objectives=2,  # EnergyNet has 2 objectives
    hidden_sizes=(256, 256, 256),  # Same network size as MOSAC
    learning_rate=3e-4,
    gamma=0.99,
    batch_size=256,  # Reduced from 512 to match typical PCN settings
    buffer_capacity=100,  # Number of episodes (not transitions like MOSAC)
    max_steps_per_episode=1000,
    training_frequency=48,  # Train every 48 steps like MOSAC
    noise_std=0.1,  # Exploration noise for continuous actions
    scaling_factor=np.array([0.01, 0.01, 0.001]),  # Scale for [obj1, obj2, horizon]
    writer_filename='pcn_energynet_runs',
    verbose=True
)

print("Starting PCN training...")


agent.train(10)


agent.save("trained_pcn.pth")

print("PCN training completed and model saved!")

print("Evaluating trained PCN agent...")
eval_returns = agent.evaluate(episodes=5, num_eval_points=5)
if len(eval_returns) > 0:
    print(f"Evaluation results:")
    print(f"Mean return per objective: {np.mean(eval_returns, axis=0)}")
    print(f"Std return per objective: {np.std(eval_returns, axis=0)}")
else:
    print("No evaluation results (buffer empty)")
