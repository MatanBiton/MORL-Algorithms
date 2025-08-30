# MORL-Algorithms

A comprehensive implementation of Multi-Objective Reinforcement Learning (MORL) algorithms, featuring state-of-the-art methods for training agents to optimize multiple conflicting objectives simultaneously.

## 🎯 Overview

This repository provides implementations of cutting-edge MORL algorithms designed to handle complex multi-objective optimization problems in reinforcement learning. The algorithms are tested on energy system optimization tasks but are general enough to be applied to various multi-objective domains.

## 🚀 Implemented Algorithms

### 1. Multi-Objective Soft Actor-Critic (MOSAC)
- **File**: `morl_algorithms/mosac/multi_objective_sac.py`
- **Description**: SAC-D implementation adapted for multi-objective settings, based on value function decomposition
- **Reference**: MacGlashan, J., et al. (2022). "Value Function Decomposition for Iterative Design of Reinforcement Learning Agents." arXiv:2206.13901
- **Key Features**:
  - Twin-critics with composite minimum selection for improved stability
  - Entropy as an additional objective head
  - CAGrad (Conflict-Averse Gradient) optimization support
  - Squashed Gaussian policy with proper log-probability computation
  - Value decomposition for interpretable multi-objective learning
  - Orthogonal initialization and learning rate annealing options

### 2. Pareto Conditioned Networks (PCN)
- **File**: `morl_algorithms/pcn/pareto_conditioned_networks.py`
- **Description**: Single neural network approach to learn all Pareto-efficient policies by conditioning on desired returns
- **Reference**: Reymond, M., Bargiacchi, E., & Nowé, A. (2022). "Pareto Conditioned Networks." arXiv:2204.05036
- **Key Features**:
  - Single network encompasses all non-dominated policies
  - Supervised learning approach for stability (no moving targets)
  - Transforms optimization into classification problem
  - Continuous action space support
  - Experience replay with crowding distance-based prioritization
  - Command selection based on non-dominated solutions
  - Diversity preservation through Pareto front maintenance

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- PyTorch
- Gymnasium
- NumPy

### Setup

1. Clone the repository:
```bash
git clone https://github.com/MatanBiton/MORL-Algorithms.git
cd MORL-Algorithms
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## 🎮 Usage

### Training MOSAC Agent

```python
from morl_algorithms.mosac.multi_objective_sac import MOSAC
from your_environment import YourMultiObjectiveEnv

# Create environment
env = YourMultiObjectiveEnv()

# Initialize MOSAC agent
agent = MOSAC(
    obs_dim=env.observation_space.shape[0],
    act_dim=env.action_space.shape[0],
    act_low=env.action_space.low,
    act_high=env.action_space.high,
    objectives=2,  # Number of objectives
    hidden_sizes=(256, 256, 256),
    batch_size=512,
    use_cagrad=True,  # Enable CAGrad optimization
    use_orthogonal_init=True,
    use_lr_annealing=True
)

# Train the agent
agent.train(env, episodes=1000)

# Save trained model
agent.save("trained_model.pth")
```

### Training PCN Agent

```python
from morl_algorithms.pcn.pareto_conditioned_networks import PCN
import numpy as np

# Create PCN agent
agent = PCN(
    env=env,
    objectives=2,
    hidden_sizes=(256, 256, 256),
    learning_rate=3e-4,
    batch_size=256,
    buffer_capacity=100,
    scaling_factor=np.array([0.01, 0.01, 0.001])
)

# Train the agent
agent.train(episodes=1000)
```

## 📊 Training Scripts

The repository includes ready-to-use training scripts in the `training/` directory:

- `mosac_energynet.py`: MOSAC training on EnergyNet environment
- `pcn_energynet.py`: PCN training on EnergyNet environment  
- `mosac_alternating_training.py`: Advanced MOSAC training with alternating strategies

## 🔧 Key Features

### MOSAC Algorithm Features
- **Twin-Critic Architecture**: Improved Q-value estimation with dual critics
- **Composite Minimum Selection**: Enhanced stability through intelligent target selection
- **Multi-Head Critics**: Separate heads for each objective plus entropy
- **CAGrad Integration**: Conflict-aware gradient optimization
- **Squashed Gaussian Policy**: Proper handling of bounded action spaces

### PCN Algorithm Features
- **Command-Based Learning**: Direct conditioning on desired performance
- **Pareto Front Maintenance**: Automatic discovery and preservation of trade-offs
- **Crowding Distance**: Diversity preservation in the solution set
- **Experience Prioritization**: Intelligent replay buffer management

## 📈 Monitoring

Training progress is automatically logged using TensorBoard:

```bash
# View MOSAC training logs
tensorboard --logdir mosac_runs/

# View PCN training logs  
tensorboard --logdir pcn_runs/
```

## 🔬 Research Background

This implementation is based on several key research papers:

1. **MOSAC/SAC-D**: MacGlashan, J., Archer, E., Devlic, A., Seno, T., Sherstan, C., Wurman, P. R., & Stone, P. (2022). "Value Function Decomposition for Iterative Design of Reinforcement Learning Agents." *arXiv preprint arXiv:2206.13901*. [Link](https://arxiv.org/abs/2206.13901)

2. **PCN**: Reymond, M., Bargiacchi, E., & Nowé, A. (2022). "Pareto Conditioned Networks." *arXiv preprint arXiv:2204.05036*. [Link](https://arxiv.org/abs/2204.05036)

## 🏗️ Project Structure

```
MORL-Algorithms/
├── morl_algorithms/           # Main algorithm implementations
│   ├── mosac/                # Multi-Objective SAC
│   └── pcn/                  # Pareto Conditioned Networks
├── training/                 # Training scripts and examples
├── requirements.txt         # Dependencies
└── pyproject.toml          # Package configuration
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 👨‍💻 Authors

**Matan Biton**  
Email: matan.biton@campus.technion.ac.il  
GitHub: [@MatanBiton](https://github.com/MatanBiton)

**Erel Hadad**
Email: erel.hadad@campus.technion.ac.il  
GitHub: [@erelhadad](https://github.com/erelhadad)

## 🙏 Acknowledgments

- SAC implementation references from the original SAC paper
- PCN implementation adapted from the original AAMAS 2022 paper
- EnergyNet environment for multi-objective energy system optimization adapted from CLAIR LAB at Technion (https://github.com/CLAIR-LAB-TECHNION)