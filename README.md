# MORL-Algorithms

A comprehensive implementation of Multi-Objective Reinforcement Learning (MORL) algorithms, featuring state-of-the-art methods for training agents to optimize multiple objectives simultaneously as ISO Operators in power-grids management.

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

### Training with Alternating Training Script

The primary way to train agents in this repository is through the `alternating_training.py` script, which implements a sophisticated alternating training strategy between ISO (Independent System Operator) and PCS (Power Conversion System) agents.

#### Basic Usage

```bash
# Train MOSAC algorithm with default settings
python training/alternating_training.py --algo mosac

# Train PCN algorithm with default settings  
python training/alternating_training.py --algo pcn

# Train with dispatch actions enabled
python training/alternating_training.py --algo mosac --disp
```

#### Advanced Configuration

```bash
# Train MOSAC with optimization features
python training/alternating_training.py \
    --algo mosac \
    --cycles 10 \
    --iso_episodes 150000 \
    --pcs_timesteps 200000 \
    --hidden 512 512 512 \
    --batch 1024 \
    --cagrad \
    --orth_init \
    --lr_anneal cosine

# Train PCN with custom parameters
python training/alternating_training.py \
    --algo pcn \
    --cycles 8 \
    --iso_episodes 100000 \
    --pcs_timesteps 150000 \
    --hidden 256 256 256 \
    --batch 256 \
    --train_freq 24
```

#### Command Line Arguments

- `--algo`: Choose algorithm (`mosac` or `pcn`)
- `--disp`: Enable dispatch action for ISO agent
- `--cycles`: Number of alternating training cycles (default: 5)
- `--iso_episodes`: Episodes for ISO training per cycle (default: 100,000)
- `--pcs_timesteps`: Timesteps for PCS training per cycle (default: 100,000)
- `--hidden`: Hidden layer dimensions (default: [256, 256, 256])
- `--batch`: Batch size for training (default: 512)
- `--train_freq`: Training frequency in steps (default: 48)

**MOSAC-specific optimizations:**
- `--cagrad`: Enable CAGrad optimization
- `--orth_init`: Use orthogonal initialization
- `--lr_anneal`: Learning rate annealing (`cosine`)

#### Training Process

The alternating training script performs the following steps:

1. **Initialization**: Sets up the multi-objective ISO environment
2. **Alternating Cycles**: For each training cycle:
   - **Phase 1**: Train ISO agent using MOSAC or PCN
   - **Phase 2**: Train PCS agent (PPO) using the current ISO agent
3. **Model Saving**: Saves trained models after each cycle and final models
4. **Evaluation**: Automatically generates evaluation data and graphs

#### Output Files

After training, the script generates:
- `{algo}_trained_iso.pth`: Final trained ISO agent
- `{algo}_trained_pcs.zip`: Final trained PCS agent  
- `{algo}_trained_iso_cycle_{i}.pth`: ISO models from each cycle
- `{algo}_trained_pcs_cycle_{i}.zip`: PCS models from each cycle
- Evaluation logs and performance graphs

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

Training progress is automatically logged using TensorBoard. The alternating training script creates separate logs for each training cycle:

```bash
# View MOSAC training logs for specific cycle
tensorboard --logdir mosac_run_iter_1/
tensorboard --logdir mosac_run_iter_2/

# View PCN training logs for specific cycle  
tensorboard --logdir pcn_run_iter_1/
tensorboard --logdir pcn_run_iter_2/

# View all training runs
tensorboard --logdir ./
```

The script also generates evaluation graphs and CSV files in the `evaluation_logs/` directory after training completion.

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