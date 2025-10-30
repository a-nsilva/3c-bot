# Behavioral Dynamics of Creative Cooperation in Human-3C-Bot Communities

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-pending-orange.svg)](https://github.com/a-nsilva/3c-bot)

This repository contains the complete implementation of the agent-based simulation described in our paper:

> **Silva, A.N., Nikghadam-Hojjati, S., Barata, J., & Estrada, L. (2025).**  
> *"Behavioral dynamics of creative cooperation in human-3C-bot communities: an agent-based simulation of trust-mediated innovation."*  
> [Journal Name] [Under Review]

## 🎯 Overview

This computational model simulates trust-mediated creative cooperation in mixed human-robot organizational communities. It integrates four theoretical frameworks:

- **Social Value Orientation** (Van Lange, 1999; Balliet et al., 2009)
- **Asimov's Three Laws of Robotics** (Anderson & Anderson, 2007)
- **Guilford's Creativity Model** (1967)
- **Trust in Automation Theory** (Lee & See, 2004; Hancock et al., 2011)

## 🚀 Quick Start

### System Requirements
- **Python**: 3.11.0 or higher
- **RAM**: 8GB minimum
- **Disk Space**: ~500MB for results
- **OS**: Windows, macOS, or Linux

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/a-nsilva/3c-bot.git
cd 3c-bot
```

2. **Create virtual environment** (recommended)
```bash
# Using venv
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n 3c-bot python=3.11
conda activate 3c-bot
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import numpy, scipy, networkx, matplotlib; print('✅ All dependencies installed successfully')"
```

### Running the Simulation

The simulation is implemented as a self-contained Jupyter notebook for ease of understanding and replication.
```bash
# Launch Jupyter
jupyter notebook

# Open: human_robot_simulation.ipynb
# Execute cells sequentially (Cell → Run All)
```

**Expected runtime**: ~30 minutes for full experiment (50 simulations)

**Expected output**:
- Statistical results (JSON format) in `results/data/`
- Visualization plots (PNG format) in `results/figures/`

## 📊 Reproducing Paper Results

The notebook reproduces all results from the manuscript:

| Configuration | Mean Trust | 95% CI | Symbiosis Rate |
|--------------|------------|--------|----------------|
| Human Majority (83%H/17%R) | 0.621 | [0.612, 0.631] | 0% |
| Human Lean (67%H/33%R) | 0.660 | [0.650, 0.670] | 0% |
| Balanced (50%H/50%R) | 0.670 | [0.658, 0.681] | 0% |
| Robot Lean (33%H/67%R) | 0.674 | [0.668, 0.681] | 0% |
| **Robot Majority (17%H/83%R)** | **0.692** | **[0.680, 0.703]** | **20%** |

**Statistical Analysis**: F(4,45) = 35.00, p < 0.001, η² = 0.757 (large effect size)

### Quick Validation

To verify correct installation, run the **demo mode** (completes in ~5 minutes):

1. Open the notebook
2. Locate the `main()` function
3. Select option `1. 🚀 Quick demo (validation)`

The demo should produce:
- Final trust: ~0.670
- Convergence cycle: ~50
- 12 diagnostic plots

## 📖 Documentation

### Theoretical Framework
See [`docs/theoretical_framework.md`](docs/theoretical_framework.md) for detailed explanation of:
- Social Value Orientation (SVO) theory
- Asimov's Three Laws implementation
- Guilford's four-factor creativity model
- Trust-mediated innovation mechanisms

### Parameter Justification
See [`docs/parameter_justification.md`](docs/parameter_justification.md) for literature-based justification of all simulation parameters:
- Cooperation tendencies (0.80-0.95 for altruistic, etc.)
- Trust thresholds (0.70 for innovation symbiosis)
- Creative capability distributions (Beta distributions)
- Activity weights (0.8 for creative collaboration, etc.)

### Replication Guide
See [`docs/replication_guide.md`](docs/replication_guide.md) for step-by-step instructions, troubleshooting, and expected outputs.

## 🔬 Methodology Highlights

### Agent Types
- **Human Agents**: 3 behavioral profiles (Altruistic, Egoistic, Vindictive) based on SVO theory
- **Robot Agents**: Consistent behavior governed by Asimov's Three Laws

### Population Configurations
- 5 human-robot ratios tested (83%H/17%R to 17%H/83%R)
- 60 agents per simulation
- 10 replications per configuration (N=50 total simulations)

### Key Metrics
- **Trust Evolution**: Human-robot trust dynamics over 1000 cycles (~3 organizational years)
- **ICC Score**: Cooperation-Creativity Index (composite metric)
- **Innovation Symbiosis**: Binary achievement of trust threshold (≥0.70)
- **Network Density**: Collaboration network properties

### Statistical Validation
- One-way ANOVA with effect size (η²)
- 95% Confidence intervals
- Shapiro-Wilk normality tests
- Levene's homogeneity tests
- Behavioral distribution validation (χ² goodness-of-fit)

## 📁 Repository Structure
```
3c-bot/
├── README.md                          # This file
├── LICENSE                            # GPL-3.0 license
├── CITATION.cff                       # Structured citation metadata
├── requirements.txt                   # Python dependencies
├── human_robot_simulation.ipynb       # Complete simulation code
├── docs/
│   ├── theoretical_framework.md       # Detailed theory
│   ├── parameter_justification.md     # Parameter sources
│   └── replication_guide.md           # Step-by-step guide
└── results/                           # Generated outputs (not in Git)
    ├── data/                          # JSON results
    └── figures/                       # PNG plots
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'numpy'`  
**Solution**: Ensure you've activated the virtual environment and run `pip install -r requirements.txt`

**Issue**: Notebook kernel crashes during simulation  
**Solution**: Reduce population size or number of replications (see configuration section in notebook)

**Issue**: Results differ slightly from paper  
**Solution**: Ensure you're using Python 3.11+ and exact dependency versions from `requirements.txt`. Note: floating-point arithmetic may cause minor (<0.001) variations across platforms.

**Issue**: Plots not displaying  
**Solution**: If using Jupyter Lab, install widget extension: `jupyter labextension install @jupyter-widgets/jupyterlab-manager`

For other issues, please open a [GitHub Issue](https://github.com/a-nsilva/3c-bot/issues).

## 📄 Citation

If you use this code in your research, please cite:
```bibtex
@article{silva2025creative,
  title={Behavioral dynamics of creative cooperation in human-3C-bot communities: an agent-based simulation of trust-mediated innovation},
  author={Silva, Alexandre do Nascimento and Nikghadam-Hojjati, Sanaz and Barata, Jos{\'e} and Estrada, Luiz},
  journal={[Journal Name]},
  year={2025},
  note={Under Review}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

## 📜 License

This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE](LICENSE) file for details.

Key points:
- ✅ Free to use, modify, and distribute
- ✅ Must disclose source code
- ✅ Must use same GPL-3.0 license for derivatives
- ✅ No warranty provided

## 👥 Authors & Contact

- **Alexandre do Nascimento Silva** (Corresponding Author)  
  Universidade Estadual de Santa Cruz (UESC)  
  📧 alnsilva@uesc.br

- **Sanaz Nikghadam-Hojjati**  
  Universidade Nova de Lisboa

- **José Barata**  
  Universidade Nova de Lisboa

- **Luiz Estrada**  
  Universidade Nova de Lisboa

## 🙏 Acknowledgments

This research was supported by:
- Universidade Estadual de Santa Cruz (UESC)
- Universidade Nova de Lisboa
- [Add funding agencies if applicable]

## 🔗 Related Resources

- [Paper Preprint](https://arxiv.org/abs/XXXXX) (when available)
- [Supplementary Materials](https://doi.org/XXXXX) (when available)
- [Project Website](https://a-nsilva.github.io/3c-bot) (optional)

---

**Last Updated**: January 2025  
**Repository Status**: Under active development for publication  
**Issues**: Please report bugs via [GitHub Issues](https://github.com/a-nsilva/3c-bot/issues)
