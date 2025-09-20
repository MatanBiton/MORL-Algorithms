from typing import Union
import pathlib

from morl_algorithms.mosac.multi_objective_sac import MOSAC
from morl_algorithms.pcn.pareto_conditioned_networks import PCN
from EnergyNetMoISO.MoISOEnv import MultiObjectiveISOEnv
from stable_baselines3.ppo.ppo import PPO

import pandas as pd
import matplotlib.pyplot as plt


def produce_eval_data(trained_iso: Union[MOSAC, PCN], trained_pcs: PPO, episodes:int = 1, dir_name:str = 'evaluation_logs') ->  None:
    env = MultiObjectiveISOEnv(
        use_dispatch_action= trained_iso.act_dim == 3,
        trained_pcs_model=trained_pcs
    )

    trained_iso.evaluate(env, episodes=episodes, log_info=True, log_dir=dir_name)


def generate_graph(log_path:str = 'evaluation_logs') -> None:
    if not pathlib.Path(log_path).is_file():
        raise FileNotFoundError(f"could not find {log_path}")
    
    df = pd.read_csv(log_path)

    required_cols = ['dispatch', 'predicted_demand', 'realized_demand', 'net_demand']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in {log_path} loaded")
        
    fig, ax = plt.subplots(figsize=(12,6))

    time_index = range(len(df))

    bars = ax.bar(time_index, df['dispatch'], alpha=0.6,
                  color='lightblue', label='Dispatch', width=0.8)
    
    ax.plot(time_index, df['predicted_demand'], 'b-o', linewidth=2,
            markersize=4, label='Predicted Demand')
    
    ax.plot(time_index, df['realized_demand'], 'r-o', linewidth=2,
            markersize=4, label='Realized Demand')
    
    ax.plot(time_index, df['net_demand'], 'g-o', linewidth=2,
            markersize=4, label='Net Demand')
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Energy (MW)')
    ax.set_title('ISO eval Dispatch & Demand')

    ax.legend(loc='upper right')

    ax.grid(True, alpha=0.3)

    ax.set_xticks(range(0, len(df), 4))
    ax.set_xticklabels(range(0, len(df), 4))

    plt.tight_layout()
    

    output_path = log_path.replace('.csv', '_graph.png')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    