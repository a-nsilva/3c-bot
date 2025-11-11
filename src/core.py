"""
COMPUTATIONAL MODELING OF BEHAVIORAL DYNAMICS IN HUMAN-ROBOT ORGANIZATIONAL COMMUNITIES
=======================================================================================
Author: 
  Alexandre do Nascimento Silva
Affiliation: 
  Universidade Estadual de Santa Cruz (UESC), Departamento de Engenharias e Computação
  Universidade do Estado da Bahia (UNEB), Programa de Pós-graduação em Modelagem e Simulação de Biossistemas (PPGMSB)
Contact:
  alnsilva@uesc.br

Theoretical Foundation:
- Social Value Orientation (Van Lange, 1999; Balliet et al., 2009)
- Asimov's Three Laws of Robotics (1950; Anderson & Anderson, 2007) 
- Guilford's Structure of Intellect Model (1967)
- Trust in Automation Theory (Lee & See, 2004; Hancock et al., 2011)
"""

# =====================================================================
# 1. IMPORTS
# =====================================================================
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import random
import seaborn as sns
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from scipy import stats
from typing import Any, Dict, List, Optional, Tuple, Union

# Configuration
warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)

# =====================================================================
# 2. ENUMS & CONFIGURATIONS
# =====================================================================
class BehaviorProfile(Enum):
    """Social Value Orientation profiles (Van Lange, 1999; Balliet et al., 2009)"""
    ALTRUISTIC = "altruistic"
    EGOISTIC = "egoistic" 
    VINDICTIVE = "vindictive"

class ActivityType(Enum):
    """Organizational activity types with creativity weights"""
    CREATIVE_COLLABORATION = "creative_collaboration"
    KNOWLEDGE_EXCHANGE = "knowledge_exchange"
    ADAPTIVE_RESOLUTION = "adaptive_resolution"

class ExperimentScale(Enum):
    """Experimental scales for cross-validation"""
    SMALL = (30, 5)
    MEDIUM = (60, 10) 
    LARGE = (90, 15)

class ConfigurationType(Enum):
    """Human-robot population ratios"""
    MAJORITY_HUMAN = (0.83, 0.17)
    HUMAN_LEAN = (0.67, 0.33)
    BALANCED = (0.50, 0.50)
    ROBOT_LEAN = (0.33, 0.67)
    MAJORITY_ROBOT = (0.17, 0.83)

@dataclass
class TheoreticalParameters:
    """Empirically-grounded parameters from literature"""
    ALTRUISTIC_COOPERATION = (0.80, 0.95)
    EGOISTIC_COOPERATION = (0.15, 0.35)
    VINDICTIVE_COOPERATION = (0.45, 0.70)
    TRUST_THRESHOLD_SYMBIOSIS = 0.70
    INITIAL_ROBOT_TRUST = 0.50
    TRUST_LEARNING_RATE = 0.10
    ROBOT_ETHICAL_COMPLIANCE = 0.97
    HARM_THRESHOLD = 0.20

@dataclass
class CreativeCapabilities:
    """Guilford's Structure of Intellect Model (1967)"""
    fluency: float = 0.5
    flexibility: float = 0.5
    originality: float = 0.5
    elaboration: float = 0.5

    def overall_score(self) -> float:
        return (self.fluency + self.flexibility + self.originality + self.elaboration) / 4

@dataclass 
class Activity:
    """Organizational activity with scientific metrics"""
    id: str
    type: ActivityType
    duration: int
    ideal_team_size: int
    creativity_weight: float
    participants: List[str] = field(default_factory=list)
    progress: float = 0.0
    quality: float = 0.0
    icc_score: float = 0.0
    completed: bool = False
    start_cycle: int = 0

# =====================================================================
# 3. CORE CLASSES (Agent, HumanAgent, RobotAgent)
# =====================================================================
class Agent:
    """Base agent class with theoretical grounding"""
    def __init__(self, agent_id: str, agent_type: str):
        self.id: str = agent_id
        self.type: str = agent_type
        self.energy = 1.0
        self.stress = 0.0
        self.satisfaction = 0.5
        self.trust_network: Dict[str, float] = {}
        self.performance_history: List[float] = []
        self.creative_capabilities = CreativeCapabilities()
        self._initialize_creativity()

    def _initialize_creativity(self):
        """Initialize creativity based on agent type (Guilford, 1967)"""
        if self.type == "human":
            self.creative_capabilities.fluency = np.random.beta(2, 2)
            self.creative_capabilities.flexibility = np.random.beta(2, 2)
            self.creative_capabilities.originality = np.random.beta(2, 3)
            self.creative_capabilities.elaboration = np.random.beta(3, 2)
        else:
            self.creative_capabilities.fluency = np.random.beta(3, 2)
            self.creative_capabilities.flexibility = np.random.beta(2, 2)
            self.creative_capabilities.originality = np.random.beta(1.5, 3)
            self.creative_capabilities.elaboration = np.random.beta(4, 2)

    def update_trust(self, partner_id: str, outcome: float):
        """Update trust based on Lee & See (2004)"""
        current_trust = self.trust_network.get(partner_id, 0.5)
        learning_rate = TheoreticalParameters.TRUST_LEARNING_RATE
        new_trust = current_trust + learning_rate * (outcome - current_trust)
        self.trust_network[partner_id] = np.clip(new_trust, 0, 1)

    def update_state(self, workload: float, social_support: float):
        """Update psychological state (Job Demands-Resources Model)"""
        energy_change = -workload * 0.1 + social_support * 0.05
        self.energy = np.clip(self.energy + energy_change, 0, 1)

        stress_change = workload * 0.08 - social_support * 0.04
        self.stress = np.clip(self.stress + stress_change, 0, 1)

        self.satisfaction = (self.energy + (1 - self.stress) + social_support) / 3

class HumanAgent(Agent):
    """Human agent with SVO-based behavior"""
    def __init__(self, agent_id: str, behavior_profile: BehaviorProfile):
        super().__init__(agent_id, "human")
        self.behavior_profile = behavior_profile
        self.robot_trust = TheoreticalParameters.INITIAL_ROBOT_TRUST
        self.fatigue_factor = 1.0
        self._initialize_cooperation_tendency()

    def _initialize_cooperation_tendency(self):
        """Initialize cooperation based on SVO (Van Lange, 1999)"""
        if self.behavior_profile == BehaviorProfile.ALTRUISTIC:
            range_min, range_max = TheoreticalParameters.ALTRUISTIC_COOPERATION
        elif self.behavior_profile == BehaviorProfile.EGOISTIC:
            range_min, range_max = TheoreticalParameters.EGOISTIC_COOPERATION
        else:
            range_min, range_max = TheoreticalParameters.VINDICTIVE_COOPERATION

        self.cooperation_tendency = np.random.uniform(range_min, range_max)

    def decide_cooperation(self, partner: Agent, activity_type: ActivityType) -> float:
        """Cooperation decision based on SVO theory"""
        base_cooperation = self.cooperation_tendency
        trust_modifier = self.trust_network.get(partner.id, 0.5) - 0.5

        if partner.type == "robot":
            cooperation = base_cooperation * self.robot_trust + trust_modifier * 0.2
        else:
            if self.behavior_profile == BehaviorProfile.VINDICTIVE:
                reciprocity_score = self.trust_network.get(partner.id, 0.5)
                cooperation = base_cooperation * (0.5 + reciprocity_score * 0.5)
            else:
                cooperation = base_cooperation + trust_modifier * 0.3

        return np.clip(cooperation, 0, 1)

    def contribute_to_activity(self, cooperation_level: float) -> float:
        """Activity contribution with human factors"""
        base_contribution = (self.creative_capabilities.overall_score() *
                            self.energy * (1 - self.stress * 0.4) *
                            self.fatigue_factor * cooperation_level)

        human_variance = np.random.uniform(0.85, 1.15)
        contribution = base_contribution * human_variance

        self.fatigue_factor = max(0.3, self.fatigue_factor - 0.02)

        return np.clip(contribution, 0, 1)

    def update_robot_trust(self, robot_id: str, outcome: float):
        """Robot-specific trust update (Hancock et al., 2011)"""
        conservative_rate = TheoreticalParameters.TRUST_LEARNING_RATE * 0.7
        self.robot_trust += conservative_rate * (outcome - self.robot_trust)
        self.robot_trust = np.clip(self.robot_trust, 0, 1)
        self.update_trust(robot_id, outcome)

class RobotAgent(Agent):
    """Robot agent following Asimov's Three Laws"""
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "robot")
        self.processing_power = np.random.uniform(0.7, 1.0)
        self.consistency_factor = np.random.uniform(0.90, 0.99)
        self.ethical_compliance = TheoreticalParameters.ROBOT_ETHICAL_COMPLIANCE
        self.asimov_weights = [1.0, 0.8, 0.6]

    def decide_cooperation(self, partner: Agent, activity_type: ActivityType) -> float:
        """Cooperation based on Asimov's Laws"""
        base_cooperation = self.processing_power

        if partner.type == "human" and partner.stress > TheoreticalParameters.HARM_THRESHOLD:
            base_cooperation *= 1.3

        if partner.type == "human":
            base_cooperation *= 1.1

        if self.stress > 0.8:
            base_cooperation *= 0.8

        return np.clip(base_cooperation, 0, 1)

    def contribute_to_activity(self, cooperation_level: float) -> float:
        """Robot contribution with consistency and precision"""
        base_contribution = (self.creative_capabilities.overall_score() *
                            self.processing_power * self.consistency_factor *
                            self.energy * (1 - self.stress * 0.3) * cooperation_level)

        if np.random.random() < (1 - self.ethical_compliance):
            contribution = base_contribution * np.random.uniform(0.6, 0.9)
        else:
            contribution = base_contribution * np.random.uniform(0.95, 1.05)

        return np.clip(contribution, 0, 1)

# =====================================================================
# 4. DATA COLLECTOR (com sensitivity, scalability embutidas)
# =====================================================================
class DataCollector:
    """Scientific data collector with validation metrics"""
    def __init__(self):
        self.metrics_history: List[Dict] = []
        self.raw_data: Dict[str, List] = defaultdict(list)
        self.agent_states_history: Dict[str, Dict[str, Any]] = {}

    # ... (mantendo todos os métodos originais: calculate_icc, validate_behavioral_distribution, etc.)

    def run_sensitivity_analysis(self):
        """Análise de sensibilidade dos parâmetros principais"""
        print("\n" + "="*70)
        print("🔬 SENSITIVITY ANALYSIS - Parameter Robustness Testing")
        print("="*70)

        original_learning_rate = TheoreticalParameters.TRUST_LEARNING_RATE
        original_threshold = TheoreticalParameters.TRUST_THRESHOLD_SYMBIOSIS

        sensitivity_results = {}

        # Testar Trust Learning Rate
        print("\n📊 Testing Trust Learning Rate (α) sensitivity...")
        alpha_values = [0.05, 0.10, 0.15]
        alpha_results = []

        for alpha in alpha_values:
            print(f"   Testing α = {alpha}...", end='')
            TheoreticalParameters.TRUST_LEARNING_RATE = alpha
            
            # Executar simulação rápida
            from .core import ResearchExperiment
            result = ResearchExperiment().run_single_experiment(
                ConfigurationType.MAJORITY_ROBOT,
                ExperimentScale.SMALL,
                cycles=500,
                seed=42
            )

            alpha_results.append({
                'alpha': alpha,
                'final_trust': result['final_trust'],
                'achieved_symbiosis': result['achieved_symbiosis']
            })
            print(f" Trust: {result['final_trust']:.3f}")

        TheoreticalParameters.TRUST_LEARNING_RATE = original_learning_rate
        # ... (continuar com análise completa)

        return sensitivity_results

    def validate_population_scalability(self):
        """Validação de escalabilidade populacional"""
        print("\n" + "="*70)
        print("📊 POPULATION SCALABILITY VALIDATION")
        print("="*70)

        scales_to_test = [ExperimentScale.SMALL, ExperimentScale.MEDIUM, ExperimentScale.LARGE]
        results_by_scale = {}

        for scale in scales_to_test:
            pop_size = scale.value[0]
            print(f"\n🔬 Testing population size: {pop_size} agents")

            from .core import ResearchExperiment
            result = ResearchExperiment().run_single_experiment(
                ConfigurationType.MAJORITY_ROBOT,
                scale,
                cycles=1000,
                seed=42
            )

            results_by_scale[pop_size] = {
                'final_trust': result['final_trust'],
                'achieved_symbiosis': result['achieved_symbiosis'],
                'convergence_cycle': result.get('convergence_cycle')
            }

        # Análise estatística de escalabilidade
        # ... (implementar análise completa)

        return results_by_scale

# =====================================================================
# 5. VISUALIZER (inclui Figure 4)
# =====================================================================
class ResearchVisualizer:
    def __init__(self):
        self.colors = {
            'altruistic': '#2E86AB',
            'egoistic': '#A23B72', 
            'vindictive': '#F18F01',
        }

    # ... (mantendo todos os métodos de visualização originais)

    def generate_figure4(self, data_collector: DataCollector, agents: List[Agent], save_path: str = None):
        """Generate Figure 4 - Agent States Heatmap"""
        print("\n🎨 Generating Figure 4 - Agent States Heatmap")
        return self.create_agent_states_heatmap(data_collector, agents, save_path)

    def create_agent_states_heatmap(self, data_collector: DataCollector, agents: List[Agent], save_path: str = None):
        """Create the agent states heatmap visualization"""
        # ... (implementação completa do heatmap)

# =====================================================================
# 6. SIMULATION ENGINE
# =====================================================================
class CoreEngine:
    """Core simulation engine with clean architecture"""
    # ... (mantendo toda a implementação original)

# =====================================================================
# 7. EXPERIMENT RUNNERS (Complete & Custom)
# =====================================================================
class ResearchExperiment:
    """Comprehensive experiment runner for research purposes"""
    
    def __init__(self):
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        self.plots_dir = Path("plots") 
        self.plots_dir.mkdir(exist_ok=True)
        self.visualizer = ResearchVisualizer()

    def run_complete_experiment(self, scale: ExperimentScale = ExperimentScale.MEDIUM, cycles: int = 1000):
        """Execute complete experiment with all configurations"""
        print("🧬 COMPLETE RESEARCH EXPERIMENT")
        print(f"Scale: {scale.name} | Cycles: {cycles}")
        print("="*60)

        # Implementação completa do experimento
        # ... (mantendo toda a lógica original)

    def run_custom_experiment(self, config_type: ConfigurationType = None, 
                            scale: ExperimentScale = None, cycles: int = 1000):
        """Execute custom experiment with user-defined parameters"""
        print("🎯 CUSTOM EXPERIMENT")
        
        # Se parâmetros não forem fornecidos, pedir ao usuário
        if config_type is None:
            config_type = self._select_configuration_interactive()
        if scale is None:
            scale = self._select_scale_interactive()
        
        print(f"Configuration: {config_type.name}")
        print(f"Scale: {scale.name}")
        print(f"Cycles: {cycles}")

        return self.run_single_experiment(config_type, scale, cycles, seed=42)

    def _select_configuration_interactive(self):
        """Selecionar configuração interativamente"""
        print("\nSelect population configuration:")
        configs = list(ConfigurationType)
        for i, config in enumerate(configs, 1):
            humans, robots = config.value
            print(f"{i}. {config.name} ({int(humans*100)}%H/{int(robots*100)}%R)")
        
        choice = int(input("Choose (1-5): ")) - 1
        return configs[choice]

    def _select_scale_interactive(self):
        """Selecionar escala interativamente"""
        print("\nSelect experimental scale:")
        scales = list(ExperimentScale)
        for i, scale in enumerate(scales, 1):
            print(f"{i}. {scale.name} ({scale.value[0]} agents)")
        
        choice = int(input("Choose (1-3): ")) - 1
        return scales[choice]

    # ... (mantendo todos os outros métodos: run_single_experiment, run_sensitivity_analysis, etc.)
