from morl_algorithms.mosac.multi_objective_sac import MOSAC
from morl_algorithms.pcn.pareto_conditioned_networks import PCN
from EnergyNetMoISO.MoISOEnv import MultiObjectiveISOEnv
from stable_baselines3.ppo.ppo import PPO

from evaluating.produce_results import *


def main(iso_path, pcs_path, algo='mosac'):

    pcs_agent = PPO.load(pcs_path)

    iso_env = MultiObjectiveISOEnv(use_dispatch_action=True, trained_pcs_model=pcs_agent)
    if algo == 'mosac':
        agent = MOSAC(obs_dim=3, act_dim=3 ,
                      act_low=iso_env.action_space.low,
                      act_high=iso_env.action_space.high,
                      objectives=2,
                      hidden_sizes=[256,256,256],
                      training_frequency=48,
                      batch_size=512,
                      use_cagrad=True,
                      use_orthogonal_init=True,
                      use_lr_annealing=True,
                      lr_anneal_type='cosine')
    else:
        agent = PCN(obs_dim=3, act_dim=3,
                    act_low=iso_env.action_space.low,
                    act_high=iso_env.action_space.high,
                    objectives=2,
                    hidden_sizes=[256,256,256],
                    training_frequency=48,
                    batch_size=512,
                    learning_rate=3e-4,
                    gamma=0.99,
                    buffer_capacity=100,
                    max_steps_per_episode=1000,
                    noise_std=0.1,
                    verbose=False)
        
    agent.load(iso_path)

    produce_eval_data(agent, pcs_agent)
    generate_graph()


if __name__ == '__main__':
    main(iso_path=r'C:\Coding\MORL-Algorithms\final_models\third_iter\mosac_trained_iso.pth',
         pcs_path=r'C:\Coding\MORL-Algorithms\final_models\third_iter\mosac_trained_pcs.zip')
        
    