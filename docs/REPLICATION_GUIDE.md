# Replication Guide

Step-by-step instructions for reproducing all results from the manuscript.

## Prerequisites

✅ Python 3.11.0 or higher  
✅ 8GB RAM minimum  
✅ ~500MB free disk space  
✅ Internet connection (for initial package installation)

---

## Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/a-nsilva/3c-bot.git
cd 3c-bot
```

### Step 2: Create Virtual Environment

**Option A: Using venv (recommended)**
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Option B: Using conda**
```bash
conda create -n 3c-bot python=3.11
conda activate 3c-bot
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed numpy-1.24.x scipy-1.10.x pandas-1.5.x ...
```

### Step 4: Verify Installation
```bash
python -c "import numpy, scipy, networkx, matplotlib; print('✅ All dependencies installed')"
```

---

## Running the Simulation

### Quick Demo (5 minutes)

1. Launch Jupyter:
```bash
jupyter notebook
```

2. Open `human_robot_simulation.ipynb`

3. Execute cells sequentially until you reach `main()`

4. When prompted, select: `1. 🚀 Quick demo (validation)`

**Expected output:**
```
🚀 RESEARCH DEMO - Quick validation
==================================================
🔬 Executing: BALANCED
   Population: 15H + 15R = 30 agents
   Cycles: 500
   Executing 500 cycles..... 8.2s

📊 DEMO RESULTS:
Final trust: 0.668
Symbiosis achieved: ❌ No
Convergence: 50

✅ Research visualizations generated!
💾 Research results saved: results/result_YYYYMMDD_HHMMSS.json
```

### Full Replication (30 minutes)

1. In the notebook, execute `main()` and select: `2. 🔬 Complete research experiment`

2. When prompted for scale, select: `2. MEDIUM (60 agents, 10 replications) [~800s]`

**Expected output:**
```
🧬 COMPLETE RESEARCH EXPERIMENT
Scale: MEDIUM | Replications: 10 | Cycles: 1000
============================================================

▶ Configuration: MAJORITY_HUMAN
   Replication 1/10... 28.3s
   ...
   ✅ Completed: 10 replications                    
   Mean ± SD: 0.621 ± 0.013
   95% CI: [0.612, 0.631]
   Symbiosis rate: 0.0%

▶ Configuration: HUMAN_LEAN
   ...

▶ Configuration: BALANCED
   ...

▶ Configuration: ROBOT_LEAN
   ...

▶ Configuration: MAJORITY_ROBOT
   ...
   Mean ± SD: 0.692 ± 0.017
   95% CI: [0.680, 0.703]
   Symbiosis rate: 20.0%

📊 Statistical analysis...

🏆 SCIENTIFIC RESULTS:
Best configuration: MAJORITY_ROBOT
Trust: 0.692
95% CI: [0.680, 0.703]
ANOVA F: 35.00
p-value: 0.000000
η² = 0.757

📊 Generating research visualizations...
✅ Research visualizations generated!
💾 Research results saved: results/result_YYYYMMDD_HHMMSS.json
```

---

## Expected Results

### Table 1: Trust-Mediated Creative Cooperation

| Configuration | Mean Trust | SD | 95% CI Lower | 95% CI Upper | Symbiosis Rate |
|--------------|------------|------|--------------|--------------|----------------|
| Human Majority (83%H/17%R) | 0.621 | 0.013 | 0.612 | 0.631 | 0% |
| Human Lean (67%H/33%R) | 0.660 | 0.014 | 0.650 | 0.670 | 0% |
| Balanced (50%H/50%R) | 0.670 | 0.016 | 0.658 | 0.681 | 0% |
| Robot Lean (33%H/67%R) | 0.674 | 0.009 | 0.668 | 0.681 | 0% |
| **Robot Majority (17%H/83%R)** | **0.692** | **0.017** | **0.680** | **0.703** | **20%** |

**Statistical Test**: F(4,45) = 35.00, p < 0.001, η² = 0.757 (large effect size)

### Validation Tolerance

Due to floating-point arithmetic variations across systems, expect:

- **Trust values**: ±0.003 variation
- **F-statistic**: ±0.5 variation
- **p-value**: Same order of magnitude (p<0.001)
- **η²**: ±0.01 variation

If your results fall within these tolerances, replication is **successful**.

---

## Generated Outputs

After running the full experiment, check:

### 1. Statistical Results (JSON)
```bash
ls results/result_*.json
```

**File structure:**
```json
{
  "results_by_config": {
    "MAJORITY_HUMAN": {
      "mean_trust": 0.621,
      "std_trust": 0.013,
      "ci_95_lower": 0.612,
      "ci_95_upper": 0.631,
      ...
    },
    ...
  },
  "anova_results": {
    "f_statistic": 35.00,
    "p_value": 2.77e-13,
    "eta_squared": 0.757
  },
  ...
}
```

### 2. Visualization Plots (PNG)
```bash
ls results/figures/plot_complete_*.png
```

**Expected files:**
- `plot_complete_dashboard.png` (3×4 panel overview)
- `plot_complete_trust_evolution.png`
- `plot_complete_behavioral_profiles.png`
- `plot_complete_organizational_phases.png`
- `plot_complete_icc_evolution.png`
- `plot_complete_convergence_analysis.png`
- `plot_complete_behavioral_validation.png`

---

## Troubleshooting

### Issue: Different numerical results

**Possible causes:**
1. Different Python version (ensure 3.11+)
2. Different NumPy version (ensure 1.24+)
3. Different random seed initialization

**Solution:**
```bash
# Verify versions
python --version  # Should be 3.11.x
pip show numpy    # Should be 1.24.x or 1.25.x

# Reset random seeds (already in code)
np.random.seed(42)
random.seed(42)
```

### Issue: Memory error during simulation

**Solution:**
```python
# In notebook, reduce scale:
# Change from MEDIUM (60 agents) to SMALL (30 agents)
scale = ExperimentScale.SMALL
```

### Issue: Plots not rendering

**Solution for Jupyter Lab:**
```bash
# Install widget extension
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Or use classic notebook
jupyter notebook  # Instead of jupyter lab
```

### Issue: Slow execution

**Expected runtimes** (on 2.5 GHz quad-core processor):

| Scale | Agents | Replications | Cycles | Time |
|-------|--------|--------------|--------|------|
| SMALL | 30 | 5 | 500 | ~3 min |
| MEDIUM | 60 | 10 | 1000 | ~30 min |
| LARGE | 90 | 15 | 1000 | ~90 min |

If significantly slower, check:
1. Other programs consuming CPU/RAM
2. Running on integrated graphics (not GPU-related but system load)
3. Python interpreter (CPython is faster than PyPy for NumPy)

---

## Extending the Simulation

### Change Population Ratio
```python
# In CoreEngine.__init__(), modify:
custom_config = (0.40, 0.60)  # 40% human, 60% robot
num_humans = round(60 * 0.40)  # = 24 humans
num_robots = 60 - num_humans   # = 36 robots
```

### Change Simulation Length
```python
# In main(), modify:
cycles = 2000  # Instead of 1000 (doubles runtime)
```

### Add New Behavioral Profile
```python
# In BehaviorProfile enum:
class BehaviorProfile(Enum):
    ALTRUISTIC = "altruistic"
    EGOISTIC = "egoistic"
    VINDICTIVE = "vindictive"
    CAUTIOUS = "cautious"  # NEW

# In TheoreticalParameters:
CAUTIOUS_COOPERATION: Tuple[float, float] = (0.50, 0.65)
```

---

## Validation Checklist

After running full replication:

- [ ] All 5 configurations executed without errors
- [ ] Statistical results saved to JSON file
- [ ] 12+ PNG plots generated
- [ ] Mean trust values within ±0.003 of paper
- [ ] ANOVA F-statistic ~35 (±0.5)
- [ ] p-value < 0.001
- [ ] η² ~0.76 (±0.01)
- [ ] Robot Majority shows highest trust
- [ ] Robot Majority shows 20% symbiosis rate

If all checked: **Replication successful!** ✅

---

## Citation

If you use or modify this simulation, please cite:
```bibtex
@article{silva2025creative,
  title={Behavioral dynamics of creative cooperation in human-3C-bot communities},
  author={Silva, Alexandre do Nascimento and Nikghadam-Hojjati, Sanaz and Barata, José and Estrada, Luiz},
  journal={[Journal Name]},
  year={2025}
}
```

---

## Support

For issues not covered here:

1. Check [GitHub Issues](https://github.com/a-nsilva/3c-bot/issues)
2. Email: alnsilva@uesc.br
3. Include:
   - Python version (`python --version`)
   - Error message (full traceback)
   - Operating system
   - Steps to reproduce
```

---

## 9. results/.gitkeep
```
# This file ensures the results/ directory is tracked by Git
# even when it's empty. Actual result files are ignored by .gitignore.
