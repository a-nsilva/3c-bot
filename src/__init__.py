"""
3C-BOT Research Simulator
Computational Modeling of Behavioral Dynamics in Human-Robot Organizational Communities
"""

__version__ = "1.0.0"
__author__ = "Alexandre do Nascimento Silva"
__email__ = "alnsilva@uesc.br"

# Core components
from .core import (
    Agent,
    HumanAgent,
    RobotAgent,
    DataCollector,
    CoreEngine,
    BehaviorProfile,
    ActivityType,
    ConfigurationType,
    ExperimentScale,
    TheoreticalParameters
)

# Experiments
from .experiments import ResearchExperiment

# Visualization
from .visualization import ResearchVisualizer

# Advanced Analysis
from .analysis import AdvancedAnalysis

__all__ = [
  # Core
  'Agent', 'HumanAgent', 'RobotAgent',
  'DataCollector', 'CoreEngine',
  'BehaviorProfile', 'ActivityType',
  'ConfigurationType', 'ExperimentScale',
  'TheoreticalParameters',
  # High-level
  'ResearchExperiment',
  'ResearchVisualizer',
  'AdvancedAnalysis'
]
