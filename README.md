# CFL-ADV: Contrastive Federated Learning with Adversarial Defense

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Secure Learning from Tabular Data Silos: Contrastive Federated Learning with Adversarial Defense**
>
> 📄 [Read the Paper](https://papers.ssrn.com/abstract=5799977)



## 🎯 Overview

**CFL-ADV** extends Contrastive Federated Learning (CFL) with robust adversarial defense mechanisms for tabular data. It enables secure collaborative learning across data silos while defending against malicious clients and privacy attacks.

### Why CFL-ADV?

**The Challenge:**
- Organizations have sensitive tabular data (healthcare, finance, telecom)
- Privacy regulations (GDPR, HIPAA) prevent data sharing
- Malicious clients can poison federated learning systems
- Byzantine attacks can degrade model performance

**Our Solution:**
- ✅ Privacy-preserving federated learning
- ✅ Defense against poisoning attacks
- ✅ Byzantine-robust aggregation
- ✅ Self-supervised contrastive learning

---

## ✨ Key Features

### 🔒 Privacy & Security
- **No data sharing** - Only encrypted model updates exchanged
- **Adversarial defense** - Multiple defense strategies against malicious clients
- **Byzantine robustness** - Resilient to corrupted updates

### 🛡️ Defense Mechanisms
- **Krum** - Selects most representative updates
- **Median** - Coordinate-wise median aggregation
- **Trimmed Mean** - Removes outlier updates
- **FoolsGold** - Detects Sybil attacks
- **Norm Clipping** - Bounds update magnitudes

### 🎯 Attack Simulation
- **Label flipping** - Corrupts training labels
- **Gradient poisoning** - Injects malicious gradients
- **Model replacement** - Replaces entire model
- **Data poisoning** - Corrupts training data

### 📊 Flexible Architecture
- **Autoencoder-based** - Learn robust representations
- **Contrastive learning** - Self-supervised from SubTab
- **Multiple aggregation** - Mean, weighted, robust methods

---

## 📦 Installation

### Quick Install

```bash
# Clone repository
git clone https://github.com/mambo06/CFL-ADV.git
cd CFL-ADV

# Create environment
conda create -n cfl-adv python=3.8
conda activate cfl-adv

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 🚀 Quick Start

### Basic Training (No Attacks)

```bash
python main.py \
  --dataset covtype \
  --clients 4 \
  --epochs 50 \
  --gpu
```

### With Adversarial Defense

```bash
python main.py \
  --dataset adult \
  --clients 10 \
  --mal_clients 2 \
  --attack_type label_flip \
  --defense_type krum \
  --epochs 50
```

### Full Adversarial Experiment

```bash
python main.py \
  --dataset higgs_small \
  --clients 10 \
  --mal_clients 3 \
  --attack_type gradient_poison \
  --defense_type trimmed_mean \
  --random_level 0.3 \
  --epochs 100 \
  --gpu
```

---

## ⚙️ Configuration

### Core Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | covtype | Dataset name |
| `--clients` | int | 4 | Number of clients |
| `--epochs` | int | 50 | Training epochs |
| `--batch_size` | int | 256 | Batch size |
| `--gpu` | flag | False | Use GPU |

### Federated Learning

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--fl_cluster` | int | 4 | FL cluster size |
| `--client_drop` | float | 0.0 | Client dropout rate |
| `--data_drop` | float | 0.0 | Data dropout rate |
| `--noniid_clients` | float | 0.0 | Non-IID ratio |

### Adversarial Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mal_clients` | int | 0 | Number of malicious clients |
| `--attack_type` | str | none | Attack type (see below) |
| `--defense_type` | str | none | Defense type (see below) |
| `--random_level` | float | 0.0 | Attack randomness [0-1] |

---

## 🛡️ Adversarial Settings

### Attack Types

```bash
--attack_type scale

--attack_type model_replacement

--attack_type direction

--attack_type gradient_ascent

--attack_type targeted
```

### Defense Types

```bash
--defense_type multi_krum

--defense_type geometric_median

--defense_type foolsgold

--defense_type trimmed_mean

--defense_type momentum

--defense_type random
```

### Example Configurations

**Scenario 1: Strong Attack, Strong Defense**
```bash
python main.py \
  --dataset adult \
  --clients 10 \
  --mal_clients 4 \
  --attack_type scale \
  --defense_type multi_krum \
  --random_level 0.5
```

**Scenario 2: Weak Attack, Easy Defense**
```bash
python main.py \
  --dataset covtype \
  --clients 8 \
  --mal_clients 1 \
  --attack_type label_flip \
  --defense_type random \
  --random_level 0.1
```




## 📁 Project Structure

```
CFL-ADV/
├── src/
│   ├── model.py              # CFL model architecture
│   ├── attacks.py            # Attack implementations
│   └── defenses.py           # Defense mechanisms
├── utils/
│   ├── load_data.py          # Data loading utilities
│   ├── eval_utils.py         # Evaluation functions
│   └── arguments.py          # Argument parsing
├── configs/
│   └── default.yaml          # Default configuration
├── adv_train.py                   # Main training script
├── _eval.py                  # Evaluation script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

---

## 📖 Citation

```bibtex
@article{cfl2025,
  title={Learning from Tabular Data Silos without Data Sharing: 
         A Contrastive Federated Learning Approach},
  author={Achmad Ginanjar and Xue Li and Priyanka Singh and Wen Hua},
  journal={SSRN Electronic Journal},
  year={2025},
  url={https://papers.ssrn.com/abstract=5799977}
}
```

---

## 🙏 Acknowledgments

Built upon:
- **[SubTab](https://github.com/AstraZeneca/SubTab)** - Self-supervised tabular learning
- **SimCLR** - Contrastive learning framework
- **FedAvg** - Federated averaging algorithm

---

## 📞 Contact

- **Email**: mambo06@gmail.com
- **Paper**: [SSRN Link](https://papers.ssrn.com/abstract=5799977)
- **Issues**: [GitHub Issues](https://github.com/mambo06/CFL-ADV/issues)

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🗺️ Roadmap

**Current (v1.0)**
- ✅ Basic CFL implementation
- ✅ Attack simulation (4 types)
- ✅ Defense mechanisms (5 types)
- ✅ Evaluation framework

**Upcoming (v1.1)**
- 🔄 Differential privacy integration
- 🔄 Adaptive defense selection
- 🔄 Real-time attack detection
- 🔄 Multi-modal data support

**Future (v2.0)**
- 📋 Secure aggregation protocols
- 📋 Homomorphic encryption
- 📋 Blockchain integration
- 📋 AutoML for hyperparameters

---

**Made with ❤️ for secure federated learning**