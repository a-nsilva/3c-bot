"""
3C-BOT Research Simulator
Human-3C-Bot Creative Cooperation Dynamics

Agent-based simulation investigating trust-mediated creative cooperation
in human-robot organizational communities.

Usage:
    # Run interactive interface
    python -m src.main
    
    # Or use programmatically
    from src import ResearchExperiment
    runner = ResearchExperiment()
    result = runner.run_demo_experiment()
"""

__version__ = "1.0.0"
__author__ = "Alexandre do Nascimento Silva et al. (2025)"
__email__ = "alnsilva@uesc.br"
__license__ = "APACHE 2.0"

# CORE COMPONENTS
from .core import (
    # Enums
    BehaviorProfile,
    ActivityType,
    ExperimentScale,
    ConfigurationType,
    
    # Dataclasses
    TheoreticalParameters,
    CreativeCapabilities,
    Activity,
    
    # Agent Classes
    Agent,
    HumanAgent,
    RobotAgent,
    
    # Engine
    DataCollector,
    CoreEngine,
)

# HIGH-LEVEL INTERFACES
from .experiments import ResearchExperiment
from .visualization import ResearchVisualizer
from .analysis import AdvancedAnalysis

# EXPORTS
__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',
    '__license__',
    
    # Enums
    'BehaviorProfile',
    'ActivityType',
    'ExperimentScale',
    'ConfigurationType',
    
    # Dataclasses
    'TheoreticalParameters',
    'CreativeCapabilities',
    'Activity',
    
    # Agent Classes
    'Agent',
    'HumanAgent',
    'RobotAgent',
    
    # Core Engine
    'DataCollector',
    'CoreEngine',
    
    # High-level interfaces
    'ResearchExperiment',
    'ResearchVisualizer',
    'AdvancedAnalysis',
]
