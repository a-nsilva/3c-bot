# 3C-Bot Simulation: Human-Robot Creative Cooperation Dynamics

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-pending-orange.svg)](https://github.com/a-nsilva/3c-bot)

This repository contains the complete implementation of the agent-based simulation described in our paper:

> **Silva, Alexandre do Nascimento, Nikghadam-Hojjati, Sanaz, Barata, José, & Estrada, Luiz (2025).**  
> *"Behavioral dynamics of creative cooperation in human-3C-bot communities: an agent-based simulation of trust-mediated innovation."*  
> [IEEE Access] [Under Review]

## 🎯 Theoretical Foundation

This computational model simulates trust-mediated creative cooperation in mixed human-robot organizational communities. It integrates four theoretical frameworks:

- **Guilford's Creativity Model** (1967)
- **Asimov's Three Laws of Robotics** (Anderson & Anderson, 2007)
- **Social Value Orientation** (Van Lange, 1999; Balliet et al., 2009)
- **Trust in Automation Theory** (Lee & See, 2004; Hancock et al., 2011)

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/a-nsilva/3c-bot.git
cd 3c-bot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Simulation

The simulation is implemented as Command Line Interface.
```bash
python -m src.main
```
## 📊 Features

- **Complete Experiments**: 5 population configurations × N replications
- **Custom Simulations**: Flexible parameters
- **Advanced Analysis**: 
  - Sensitivity analysis
  - Scalability validation
  - Agent state evolution tracking
- **Publication-Ready Visualizations**: 15+ scientific plots

- **Expected output**:
  - Statistical results (JSON/CSV format) in `results/report/`
  - Visualization plots (PNG format) in `results/plot/`
  
## 📁 Repository Structure
```
3c-bot/
├── README.md                       # This file
├── LICENSE                         # Apache 2.0 license
├── requirements.txt                # Python dependencies
├── src/
│   ├── core.py                     # Simulation engine
│   ├── experiments.py              # Experiment runners
│   ├── visualization.py            # Plotting system
│   ├── analysis.py                 # Advanced analyses
│   └── main.py                     # CLI interface
└── results/                           
    ├── report/                     # JSON outputs
    └── plot/                       # PNG figures
```

## 📄 Citation

If you use this code in your research, please cite:
```bibtex
@article{silva2025creative,
  title = {Behavioral dynamics of creative cooperation in human-3C-bot communities: an agent-based simulation of trust-mediated innovation},
  author = {Silva, Alexandre do Nascimento and Nikghadam-Hojjati, Sanaz and Barata, Jos{\'e} and Jimenez, Luiz Estrada},
  journal = {IEEE Access},
  year = {2025},
  note = {Under Review}
}
```

## 📜 License

MIT License - see LICENSE file for details.


## 👥 Authors & Contact

- **Alexandre do Nascimento Silva** (Corresponding Author)  
  Universidade Estadual de Santa Cruz (UESC), Departamento de Engenharias e Computação
  Universidade do Estado da Bahia (UNEB), Programa de Pós-graduação em Modelagem e Simulação em Biossistemas (PPGMSB)
  📧 alnsilva@uesc.br

- **Sanaz Nikghadam-Hojjati**  
  Universidade Nova de Lisboa

- **José Barata**  
  Universidade Nova de Lisboa

- **Luiz Estrada**  
  Universidade Nova de Lisboa

## 🙏 Acknowledgments

This research was supported by:
- Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES)
- Universidade Estadual de Santa Cruz (UESC)
- Universidade do Estado da Bahia (UNEB)
- Universidade Nova de Lisboa

---

**Last Updated**: November 2025  
**Repository Status**: Under active development for publication 
