"""
3C-BOT Research Simulator
Computational Modeling of Behavioral Dynamics in Human-Robot Organizational Communities
"""

__version__ = "1.0.0"
__author__ = "Alexandre do Nascimento Silva"
__email__ = "alnsilva@uesc.br"

from .core import (
    BehaviorProfile,
    ActivityType, 
    ExperimentScale,
    ConfigurationType,
    ResearchExperiment,
    ResearchVisualizer
)

__all__ = [
    'BehaviorProfile',
    'ActivityType',
    'ExperimentScale', 
    'ConfigurationType',
    'ResearchExperiment',
    'ResearchVisualizer'
]
