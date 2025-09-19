import argparse
from typing import Callable

from morl_algorithms.mosac.multi_objective_sac import MOSAC
from morl_algorithms.pcn.pareto_conditioned_networks import PCN
from EnergyNetMoISO.MoISOEnv import MultiObjectiveISOEnv
from stable_baselines3.ppo import PPO
from energy_net.env.pcs_unit_v0 import PCSUnitEnv


def set_up_arg_parser():
    parser = argparse.ArgumentParser(description='Python Script for alternating training an ISO Multi Objective Agent')


    # env settings
    parser.add_argument("--algo", action='store', choices=['mosac', 'pcn'], default='mosac',
                        help='MO Algorithm for ISO Agent: mosac for SAC-D or pcn for Pareto Conditioned Networks')
    
    parser.add_argument("--disp", action='store_true', default=False,
                        help='Enable dispatch action for ISO')


    #model and training params
    parser.add_argument("--hidden", type=int, nargs='*', default=[256,256,256], 
                        help='Hidden dimentions for Actor and Critics')
    
    parser.add_argument("--train_freq", type=int, default=48,
                        help="Sets avter how many steps the agent learns")
    
    parser.add_argument('--batch', type=int, default=512, 
                        help='Batch size for training')
    
    parser.add_argument('--cycles', type=int, default=5, 
                        help='Number of cycles in the alternating training')
    
    parser.add_argument('--iso_episodes', type=int, default=100000,
                        help='Number of episodes for ISO training')
    
    parser.add_argument('--pcs_timesteps', type=int, default=100000,
                        help='Number of timesteps for PCS training')
    

    # Optimization
    parser.add_argument('--cagrad', action='store_true', default=False,
                        help='use cagrad during mosac training')
    
    parser.add_argument('--lr_anneal', action='store', choices=['cosine'],
                        help='use lr annealing optimization')
    
    parser.add_argument('--orth_init', action='store_true', default=False,
                        help='use orthogonal initilazation optimization')
    

    # Other
    # parser.add_argument('--writer', action='store', required=True,
    #                     help='name to write logs under {alg}_run_iter{i} dir')
    
    return parser


def mosac_creation(parser, iso_env, file_name):
    return MOSAC(obs_dim=3, act_dim=3 if parser.disp else 2,
                      act_low=iso_env.action_space.low,
                      act_high=iso_env.action_space.high,
                      objectives=2,
                      hidden_sizes=parser.hidden,
                      training_frequency=parser.train_freq,
                      batch_size=parser.batch,
                      use_cagrad=parser.cagrad,
                      use_orthogonal_init=parser.orth_init,
                      use_lr_annealing=True if parser.lr_anneal else False,
                      lr_anneal_type=parser.lr_anneal,
                      writer_filename=file_name)

def pcn_creation(parser, iso_env, file_name):
    return PCN(obs_dim=3, act_dim=3 if parser.disp else 2,
                    act_low=iso_env.action_space.low,
                    act_high=iso_env.action_space.high,
                    objectives=2,
                    hidden_sizes=parser.hidden,
                    training_frequency=parser.train_freq,
                    batch_size=parser.batch,
                    learning_rate=3e-4,
                    gamma=0.99,
                    buffer_capacity=100,
                    max_steps_per_episode=1000,
                    noise_std=0.1,
                    writer_filename=file_name,
                    verbose=False)



def train(parser, agent_creation:Callable):
    iso_env = MultiObjectiveISOEnv(use_dispatch_action=parser.disp)

    # Initialize ISO agent (will be recreated for each cycle with unique tensorboard logging)
    #print("ISO agent will be initialized for each training cycle with unique tensorboard logging...")
    iso_agent = None

    # Initialize PCS agent (will be trained later)
    pcs_agent = None

    print("Starting alternating training...")

    for cycle in range(parser.cycles):
        print(f"\n=== Training Cycle {cycle + 1}/{parser.cycles} ===")
        
        # Create new ISO agent for this cycle with unique tensorboard logging
        print(f"Initializing ISO agent for cycle {cycle + 1} with tensorboard logging...")
        tensorboard_filename = f"{parser.algo}_run_iter_{cycle + 1}"
        
        if iso_agent is not None:
            # Close previous tensorboard writer before creating new one
            iso_agent.writer.close()
        
        iso_agent = agent_creation(parser, iso_env, tensorboard_filename)
        print(f"ISO agent initialized with tensorboard logging to: {tensorboard_filename}")
        
        # Phase 1: Train ISO agent
        print(f"Phase 1: Training ISO agent for {parser.iso_episodes} episodes...")
        
        if pcs_agent is not None:
            # Use trained PCS agent in ISO environment
            print("Using trained PCS agent in ISO environment...")
            iso_env_with_pcs = MultiObjectiveISOEnv(use_dispatch_action=parser.disp, trained_pcs_model=pcs_agent)
            iso_agent.train(iso_env_with_pcs, parser.iso_episodes)
        else:
            # First cycle: train ISO without PCS agent
            print("Training ISO agent without PCS agent (first cycle)...")
            iso_agent.train(iso_env, parser.iso_episodes)
        
        # Save ISO agent after training
        iso_save_path = f"{parser.algo}_trained_iso_cycle_{cycle + 1}.pth"
        iso_agent.save(iso_save_path)
        print(f"ISO agent saved to {iso_save_path}")
        
        # Close tensorboard writer for this cycle
        iso_agent.writer.close()
        print(f"Tensorboard logging for cycle {cycle + 1} closed")
        
        # Phase 2: Train PCS agent using current ISO agent
        print(f"Phase 2: Training PCS agent for {parser.pcs_timesteps} timesteps...")
        
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
        pcs_agent.learn(total_timesteps=parser.pcs_timesteps)
        
        # Save PCS agent after training
        pcs_save_path = f"{parser.algo}_trained_pcs_cycle_{cycle + 1}"
        pcs_agent.save(pcs_save_path)
        print(f"PCS agent saved to {pcs_save_path}")
        
        print(f"Cycle {cycle + 1} completed!")
    
    print("\nSaving final models...")
    iso_agent.save(f"{parser.algo}_trained_iso.pth")
    pcs_agent.save(f"{parser.algo}_trained_pcs")

    # Ensure final tensorboard writer is closed
    if iso_agent and hasattr(iso_agent, 'writer') and iso_agent.writer:
        iso_agent.writer.close()
        print("Final tensorboard writer closed")

    print("Final models saved as 'trained_iso.pth' and 'trained_pcs'")

    print("Alternating training completed!")

def main():
    parser = set_up_arg_parser()
    args = parser.parse_args()
    creator = mosac_creation if args.algo == 'mosac' else pcn_creation
    trained_agent = train(args, creator)
    
    


if __name__ == "__main__":
    main()