print("starting imports")
from morl_algorithms.mosac.multi_objective_sac import MOSAC
from EnergyNetMoISO.MoISOEnv import MultiObjectiveISOEnv
print("entered script")
env = MultiObjectiveISOEnv()
agent = MOSAC(obs_dim=3, act_dim=2,
              act_low=env.action_space.low,
              act_high=env.action_space.high,
              objectives=2,hidden_sizes=(256,256,256),training_frequency=48,batch_size=512,use_cagrad=True, use_orthogonal_init=True, use_lr_annealing=True, lr_anneal_type="cosine")
print("starting training...")
agent.train(env,10)
agent.save("trained_iso.pth")
