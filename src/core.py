"""
COMPUTATIONAL MODELING OF BEHAVIORAL DYNAMICS IN HUMAN-ROBOT ORGANIZATIONAL COMMUNITIES
=======================================================================================
Author: 
  Alexandre do Nascimento Silva (1,2)
  Sanaz Nikghadam-Hojjati (3)
  José Barata (3)
  Luis A. Estrada Jimenez (3)

Affiliation: 
  (1) Universidade Estadual de Santa Cruz (UESC), Departamento de Engenharias e Computação
  (2) Universidade do Estado da Bahia (UNEB), Programa de Pós-graduação em Modelagem e Simulação de Biossistemas (PPGMSB)
  (3) UNINOVA—Center of Technology and Systems (CTS)

Contact:
  alnsilva@uesc.br

Theoretical Foundation:
  - Social Value Orientation (Van Lange, 1999; Balliet et al., 2009)
  - Asimov's Three Laws of Robotics (1950; Anderson & Anderson, 2007) 
  - Guilford's Structure of Intellect Model (1967)
  - Trust in Automation Theory (Lee & See, 2004; Hancock et al., 2011)
"""

# ========== IMPORTS ==========
# === Standard library imports ===
import json
import random
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# === Third-party imports ===
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ========== CONFIGURATIONS ==========
warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)

# ========== ENUMS ==========
class BehaviorProfile(Enum):
  """Social Value Orientation profiles (Van Lange, 1999; Balliet et al., 2009)"""
  ALTRUISTIC = "altruistic"   # Prosocial: ~60% population
  EGOISTIC = "egoistic"       # Individualistic: ~25% population
  VINDICTIVE = "vindictive"   # Competitive: ~15% population

class ActivityType(Enum):
  """Organizational activity types with creativity weights"""
  CREATIVE_COLLABORATION = "creative_collaboration"  # Weight: 0.8
  KNOWLEDGE_EXCHANGE = "knowledge_exchange"          # Weight: 0.3
  ADAPTIVE_RESOLUTION = "adaptive_resolution"        # Weight: 0.5

class ExperimentScale(Enum):
  """Experimental scales for cross-validation"""
  SMALL = (30, 5)     # (population, replications)
  MEDIUM = (60, 10)   # Recommended for papers
  LARGE = (90, 15)    # For robust analyses

class ConfigurationType(Enum):
  """Human-robot population ratios"""
  MAJORITY_HUMAN = (0.83, 0.17)    # 83%H / 17%R
  HUMAN_LEAN = (0.67, 0.33)        # 67%H / 33%R
  BALANCED = (0.50, 0.50)          # 50%H / 50%R
  ROBOT_LEAN = (0.33, 0.67)        # 33%H / 67%R
  MAJORITY_ROBOT = (0.17, 0.83)    # 17%H / 83%R

# ========== DATACLASSES ==========
@dataclass
class CreativeCapabilities:
  """Guilford's Structure of Intellect Model (1967)"""
  fluency: float = 0.5      # Idea generation rate
  flexibility: float = 0.5   # Conceptual shifting ability
  originality: float = 0.5   # Novel solution generation
  elaboration: float = 0.5   # Idea development depth

  def overall_score(self) -> float:
    return (self.fluency + self.flexibility + self.originality + self.elaboration) / 4

@dataclass
class TheoreticalParameters:
  """Empirically-grounded parameters from literature"""
  # Social Value Orientation (Van Lange, 1999; Fehr & Fischbacher, 2003)
  ALTRUISTIC_COOPERATION = (0.80, 0.95)
  EGOISTIC_COOPERATION = (0.15, 0.35)
  VINDICTIVE_COOPERATION = (0.45, 0.70)

  # Asimov's Laws compliance (Anderson & Anderson, 2007)
  ROBOT_ETHICAL_COMPLIANCE = 0.97
  HARM_THRESHOLD = 0.20

  # Trust in automation (Mayer et al., 1995; Schaefer et al., 2016)
  TRUST_THRESHOLD_SYMBIOSIS = 0.70
  INITIAL_ROBOT_TRUST = 0.50
  TRUST_LEARNING_RATE = 0.10

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
  cci_score: float = 0.0  # Cooperation-Creativity Index
  completed: bool = False
  start_cycle: int = 0

# ========== AGENT CLASSES ==========
class Agent:
  """Base agent class with theoretical grounding"""
  def __init__(self, agent_id: str, agent_type: str):
    self.id: str = agent_id #reviewer
    self.type: str = agent_type #reviewer
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
      # Higher variability for humans
      self.creative_capabilities.fluency = np.random.beta(2, 2)
      self.creative_capabilities.flexibility = np.random.beta(2, 2)
      self.creative_capabilities.originality = np.random.beta(2, 3)
      self.creative_capabilities.elaboration = np.random.beta(3, 2)
    else:  # robot
      # Higher consistency, lower originality
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
    else:  # VINDICTIVE
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

    # Human variability
    human_variance = np.random.uniform(0.85, 1.15)
    contribution = base_contribution * human_variance

    # Cumulative fatigue
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

    # Three Laws implementation
    self.asimov_weights = [1.0, 0.8, 0.6]  # Law 1 > Law 2 > Law 3

  def decide_cooperation(self, partner: Agent, activity_type: ActivityType) -> float:
    """Cooperation based on Asimov's Laws"""
    base_cooperation = self.processing_power

    # Law 1: Do no harm - detect high human stress
    if partner.type == "human" and partner.stress > TheoreticalParameters.HARM_THRESHOLD: 
      base_cooperation *= 1.3  # Increase support

    # Law 2: Obey humans - prioritize human partners
    if partner.type == "human":
      base_cooperation *= 1.1

    # Law 3: Self-preservation - avoid overload
    if self.stress > 0.8:
      base_cooperation *= 0.8

    return np.clip(base_cooperation, 0, 1)

  def contribute_to_activity(self, cooperation_level: float) -> float:
    """Robot contribution with consistency and precision"""
    base_contribution = (self.creative_capabilities.overall_score() *
                        self.processing_power * self.consistency_factor *
                        self.energy * (1 - self.stress * 0.3) * cooperation_level)

    # Small variability due to technical failures
    if np.random.random() < (1 - self.ethical_compliance):
      contribution = base_contribution * np.random.uniform(0.6, 0.9)
    else:
      contribution = base_contribution * np.random.uniform(0.95, 1.05)

    return np.clip(contribution, 0, 1)

# ========== DATA COLLECTOR ==========
class DataCollector:
  """Scientific data collector with validation metrics"""
  
  def __init__(self):
    self.metrics_history: List[Dict] = []
    ##self.theoretical_params = TheoreticalParameters() #reviewer
    self.raw_data: Dict[str, List] = defaultdict(list)
    self.agent_states_history: Dict[str, Dict[str, Any]] = {} #reviewer

  def calculate_cci(self, cooperation_level: float, creativity_level: float,
                    stability_level: float) -> float:
    """
    CCI (Cooperation-Creativity Index) - Our theoretical contribution
    CCI = 0.4 × cooperation + 0.4 × creativity + 0.2 × stability
    """
    cci = (0.4 * cooperation_level + 0.4 * creativity_level + 0.2 * stability_level)
    return np.clip(cci, 0, 1)

  @staticmethod
  def calculate_descriptive_stats(values: List[float]) -> Dict:
    """
    Calculate mean, std, CI95 for a list of values
    
    Returns:
            Dict with keys: mean, std, ci_95_lower, ci_95_upper
    """
    if not values:
      return {
        'mean': 0.0, 'std': 0.0,
        'ci_95_lower': 0.0, 'ci_95_upper': 0.0
      }
    
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    
    if len(values) > 1:
      ci_95 = stats.t.interval(
        0.95, len(values)-1,
        loc = mean, scale = stats.sem(values)
      )
    else:
      ci_95 = (mean, mean)
    
    return {
      'mean': float(mean),
      'std': float(std),
      'ci_95_lower': float(ci_95[0]),
      'ci_95_upper': float(ci_95[1])
    }
      
  def validate_behavioral_distribution(self, humans: List[HumanAgent]) -> Dict:
    """Validate against empirical distribution (Balliet et al., 2009)"""
    if not humans:
      return {}

    # Expected distribution from literature
    expected_distribution = {
      BehaviorProfile.ALTRUISTIC: 0.60,
      BehaviorProfile.EGOISTIC: 0.25,
      BehaviorProfile.VINDICTIVE: 0.15
    }

    # Observed distribution
    total = len(humans)
    observed_distribution = {}
    for profile in BehaviorProfile:
      count = sum(1 for h in humans if h.behavior_profile == profile)
      observed_distribution[profile] = count / total

    # Chi-square goodness of fit test
    expected_counts = [expected_distribution[p] * total for p in BehaviorProfile]
    observed_counts = [observed_distribution[p] * total for p in BehaviorProfile]

    chi2_stat, p_value = stats.chisquare(observed_counts, expected_counts)

    return {
      'chi2_statistic': chi2_stat,
      'p_value': p_value,
      'expected_distribution': {p.value: v for p, v in expected_distribution.items()},
      'observed_distribution': {p.value: v for p, v in observed_distribution.items()},
      'validation_passed': p_value > 0.05,
      'realism_score': max(0, 1 - chi2_stat / 10)
    }

  def detect_organizational_phase(self, trust_history: List[float]) -> str:
    """Detect organizational phase based on trust dynamics"""
    if len(trust_history) < 50:
      return "INTEGRATION"

    recent_trust = np.mean(trust_history[-50:])
    trust_trend = np.polyfit(range(len(trust_history[-100:])), trust_history[-100:], 1)[0]
    trust_volatility = np.std(trust_history[-50:])

    threshold = TheoreticalParameters.TRUST_THRESHOLD_SYMBIOSIS #reviewer

    if recent_trust > threshold and trust_trend >= 0 and trust_volatility < 0.1:
      return "SYMBIOSIS"
    elif trust_trend < -0.005 or trust_volatility > 0.15:
      return "RESISTANCE"
    elif recent_trust > 0.4 and trust_trend > 0:
      return "NORMALIZATION"
    else:
      return "INTEGRATION"

  def collect_metrics(self, cycle: int, agents: List[Agent], activities: List[Activity],
                      completed_activities: List[Activity], network: nx.Graph) -> Dict:
    """Collect comprehensive metrics from real simulation data"""
    humans = [a for a in agents if isinstance(a, HumanAgent)]
    robots = [a for a in agents if isinstance(a, RobotAgent)]

    # Core metrics
    avg_human_robot_trust = np.mean([h.robot_trust for h in humans]) if humans else 0
    avg_satisfaction = np.mean([a.satisfaction for a in agents])
    avg_stress = np.mean([a.stress for a in agents])

    # Profile-specific metrics
    profile_metrics = {}
    cooperation_by_profile = {}

    for profile in BehaviorProfile:
      profile_agents = [h for h in humans if h.behavior_profile == profile]
      if profile_agents:
        profile_metrics[profile.value] = {
          'count': len(profile_agents),
          'avg_robot_trust': np.mean([h.robot_trust for h in profile_agents]),
          'avg_satisfaction': np.mean([h.satisfaction for h in profile_agents]),
          'avg_cooperation': np.mean([h.cooperation_tendency for h in profile_agents])
        }

        # Real cooperation with robots
        if robots:
          robot_cooperations = []
          for human in profile_agents:
            robot_partner = random.choice(robots)
            coop_level = human.decide_cooperation(robot_partner, ActivityType.CREATIVE_COLLABORATION)
            robot_cooperations.append(coop_level)
          cooperation_by_profile[profile.value] = np.mean(robot_cooperations)
        else:
          cooperation_by_profile[profile.value] = 0.5

    # Activity metrics
    recent_activities = [a for a in completed_activities if cycle - a.start_cycle <= 50]

    if recent_activities:
      avg_quality = np.mean([a.quality for a in recent_activities])
      avg_cci = np.mean([a.cci_score for a in recent_activities])
      success_rate = len([a for a in recent_activities if a.completed]) / len(recent_activities)
    else:
      avg_quality = avg_cci = success_rate = 0

    # Organizational phase detection
    trust_history = self.raw_data['trust'] + [avg_human_robot_trust]
    current_phase = self.detect_organizational_phase(trust_history)

    # Behavioral validation
    humans = [a for a in agents if isinstance(a, HumanAgent)]##reviewer
    behavioral_validation = self.validate_behavioral_distribution(humans)

    # Network metrics
    network_density = nx.density(network) if network.number_of_nodes() > 0 else 0

    # Construct comprehensive metrics
    metrics = {
      'cycle': cycle,
      'trust': avg_human_robot_trust,
      'satisfaction': avg_satisfaction,
      'stress': avg_stress,
      'quality': avg_quality,
      'cci_score': avg_cci,
      'success_rate': success_rate,
      'network_density': network_density,
      'organizational_phase': current_phase,
      'profile_metrics': profile_metrics,
      'cooperation_by_profile': cooperation_by_profile,
      'behavioral_validation': behavioral_validation,
      'population': {
        'humans': len(humans),
        'robots': len(robots),
        'total': len(agents)
      },
      'ethical_compliance': np.mean([r.ethical_compliance for r in robots]) if robots else 0.97
    }

    # Store in history and raw data
    self.metrics_history.append(metrics)

    # Update raw data for visualizations
    self.raw_data['trust'].append(avg_human_robot_trust)
    self.raw_data['satisfaction'].append(avg_satisfaction)
    self.raw_data['stress'].append(avg_stress)
    self.raw_data['quality'].append(avg_quality)
    self.raw_data['cci_score'].append(avg_cci)
    self.raw_data['network_density'].append(network_density)

    #reviewer
    # Collect individual agent states for temporal analysis
    for agent in agents:
      # Initialize structure if this is the first time viewing this agent
      if agent.id not in self.agent_states_history:
        self.agent_states_history[agent.id] = {
          'type': agent.type,  # 'human' ou 'robot'
          'energy': [],
          'stress': [],
          'satisfaction': [],
          'trust_avg': []
        }

      # Add current states
      self.agent_states_history[agent.id]['energy'].append(agent.energy)
      self.agent_states_history[agent.id]['stress'].append(agent.stress)
      self.agent_states_history[agent.id]['satisfaction'].append(agent.satisfaction)

      # Calculate average trust with all other agents
      if agent.trust_network:
        avg_trust = np.mean(list(agent.trust_network.values()))
      else:
        avg_trust = 0.5  # Value neutral if there is no trust network
      self.agent_states_history[agent.id]['trust_avg'].append(avg_trust)

    return metrics

  def get_summary(self) -> Dict:
    """Generate final summary with convergence analysis"""
    if not self.metrics_history:
      return {}

    final = self.metrics_history[-1]
    trust_data = np.array(self.raw_data['trust'])

    # Convergence analysis
    convergence_cycle = None
    window_size = 50

    if len(trust_data) > window_size:
      for i in range(window_size, len(trust_data)):
        window = trust_data[i-window_size:i]
        if len(window) > 0 and np.mean(window) > 0:
          cv = np.std(window) / np.mean(window)
          if cv < 0.03:
            convergence_cycle = i
            break

    return {
      'final_metrics': final,
      'convergence_cycle': convergence_cycle,
      'achieved_symbiosis': final['trust'] > TheoreticalParameters.TRUST_THRESHOLD_SYMBIOSIS,
      'total_cycles': len(self.metrics_history),
      'trust_statistics': {
        'mean': float(np.mean(trust_data)),
        'std': float(np.std(trust_data, ddof=1)) if len(trust_data) > 1 else 0.0,
        'final': float(trust_data[-1]) if len(trust_data) > 0 else 0.0
      }
    }

# ========== SIMULATION ENGINE ==========
class CoreEngine:
  """Core simulation engine with clean architecture"""
  def __init__(self, num_humans: int, num_robots: int, config_type: ConfigurationType):
    self.agents: List[Agent] = []
    self.activities: List[Activity] = []
    self.completed_activities: List[Activity] = []
    self.current_cycle = 0
    self.activity_counter = 0
    self.config_type = config_type

    # Modular components
    self.data_collector = DataCollector()
    self.network = nx.Graph()

    self._initialize_agents(num_humans, num_robots)
    self._initialize_network()

  def _initialize_agents(self, num_humans: int, num_robots: int):
    """Initialize agents with balanced behavioral distribution"""
    behavior_profiles = list(BehaviorProfile)

    # Create humans with balanced distribution
    for i in range(num_humans):
      profile = behavior_profiles[i % len(behavior_profiles)]
      human = HumanAgent(f"H_{i:03d}", profile)
      self.agents.append(human)

    # Create robots
    for i in range(num_robots):
      robot = RobotAgent(f"R_{i:03d}")
      self.agents.append(robot)

  def _initialize_network(self):
    """Initialize collaboration network"""
    self.network.add_nodes_from([agent.id for agent in self.agents])

  def _create_activity(self, activity_id: str, current_cycle: int) -> Activity:
    """Create new organizational activity"""
    activity_type = random.choice(list(ActivityType))

    # Activity parameters by type
    type_params = {
      ActivityType.CREATIVE_COLLABORATION: (3, 5, 0.8),
      ActivityType.KNOWLEDGE_EXCHANGE: (2, 3, 0.3),
      ActivityType.ADAPTIVE_RESOLUTION: (2, 4, 0.5)
    }

    min_team, max_team, creativity_weight = type_params[activity_type]

    return Activity(
      id = activity_id,
      type = activity_type,
      duration = random.randint(1, 3),
      ideal_team_size = random.randint(min_team, max_team),
      creativity_weight = creativity_weight,
      start_cycle = current_cycle
    )

  def step(self) -> Dict:
    """Execute one simulation cycle"""
    self.current_cycle += 1

    # 1. Create new activity
    if random.random() < 0.3:
      self.activity_counter += 1
      activity = self._create_activity(f"A_{self.activity_counter:04d}", self.current_cycle)
      team = self._assign_team(activity)

      if team:
        success = self._execute_activity(activity, team)
        self.activities.append(activity)

        if success:
          self.completed_activities.append(activity)

    # 2. Update ongoing activities
    self._update_activities()

    # 3. Social interactions
    self._execute_social_interactions()

    # 4. Update agent states
    self._update_agent_states()

    # 5. Collect metrics
    metrics = self.data_collector.collect_metrics(
        self.current_cycle, self.agents, self.activities,
        self.completed_activities, self.network
    )

    return metrics

  def _assign_team(self, activity: Activity) -> List[Agent]:
    """Assign team based on capabilities and availability"""
    available_agents = [a for a in self.agents if a.energy > 0.2]

    if len(available_agents) < 2:
      return []

    # Score agents based on creativity and availability
    agent_scores = []
    for agent in available_agents:
      creativity_score = agent.creative_capabilities.overall_score()
      availability_score = agent.energy * (1 - agent.stress)
      total_score = creativity_score * 0.6 + availability_score * 0.4
      agent_scores.append((agent, total_score))

    # Select best agents
    agent_scores.sort(key=lambda x: x[1], reverse=True)
    team_size = min(len(agent_scores), activity.ideal_team_size)
    team = [agent for agent, _ in agent_scores[:team_size]]

    activity.participants = [agent.id for agent in team]
    return team

  def _execute_activity(self, activity: Activity, team: List[Agent]) -> bool:
    """Execute activity with CCI metrics"""
    if len(team) < 2:
      return False

    # Calculate pairwise cooperation
    cooperation_sum = 0
    cooperation_count = 0

    for agent_a in team:
      for agent_b in team:
        if agent_a != agent_b:
          cooperation = agent_a.decide_cooperation(agent_b, activity.type)
          cooperation_sum += cooperation
          cooperation_count += 1

          # Update collaboration network
          if not self.network.has_edge(agent_a.id, agent_b.id):
            self.network.add_edge(agent_a.id, agent_b.id, weight=cooperation)
          else:
            current_weight = self.network[agent_a.id][agent_b.id]['weight']
            new_weight = (current_weight + cooperation) / 2
            self.network[agent_a.id][agent_b.id]['weight'] = new_weight

    avg_cooperation = cooperation_sum / max(1, cooperation_count)

    # Calculate team creativity
    team_creativity = np.mean([a.creative_capabilities.overall_score() for a in team])

    # Individual contributions
    contributions = []
    for agent in team:
      contribution = agent.contribute_to_activity(avg_cooperation)
      contributions.append(contribution)

    # Calculate team stability
    stability = 1 - np.std(contributions) if len(contributions) > 1 else 1

    # Calculate CCI (our theoretical contribution)
    cci_score = self.data_collector.calculate_cci(avg_cooperation, team_creativity, stability)

    # Final result based on synergy
    base_performance = np.mean(contributions)
    creativity_bonus = team_creativity * activity.creativity_weight * 0.3
    cooperation_bonus = avg_cooperation * 0.2

    final_result = base_performance + creativity_bonus + cooperation_bonus
    final_result = np.clip(final_result, 0, 1)

    # Update activity
    activity.progress = final_result
    activity.quality = final_result * (1 + np.random.uniform(-0.05, 0.05))
    activity.cci_score = cci_score
    activity.completed = final_result >= 0.6

    # Update agent trust
    for agent in team:
      agent.performance_history.append(final_result)
      for other in team:
        if agent != other:
          agent.update_trust(other.id, final_result)
          if isinstance(agent, HumanAgent) and isinstance(other, RobotAgent):
            agent.update_robot_trust(other.id, final_result)

      return activity.completed

  def _update_activities(self):
    """Update ongoing activities"""
    for activity in self.activities:
      if not activity.completed:
        cycles_elapsed = self.current_cycle - activity.start_cycle
        if cycles_elapsed >= activity.duration:
          activity.completed = True
          if activity.progress >= 0.6:
            self.completed_activities.append(activity)

  def _execute_social_interactions(self, num_interactions: int = 5):
    """Execute social interactions based on proximity"""
    for _ in range(num_interactions):
      if len(self.agents) < 2:
        break

      agent_a, agent_b = random.sample(self.agents, 2)

      # Determine interaction type
      interaction_type = random.choice(list(ActivityType))

      # Mutual cooperation
      coop_a = agent_a.decide_cooperation(agent_b, interaction_type)
      coop_b = agent_b.decide_cooperation(agent_a, interaction_type)

      # Interaction outcome with noise
      interaction_outcome = (coop_a + coop_b) / 2
      noise = np.random.normal(0, 0.02)
      final_outcome = np.clip(interaction_outcome + noise, 0, 1)

      # Update trust
      agent_a.update_trust(agent_b.id, final_outcome)
      agent_b.update_trust(agent_a.id, final_outcome)

      # Update H-R specific trust
      if isinstance(agent_a, HumanAgent) and isinstance(agent_b, RobotAgent):
        agent_a.update_robot_trust(agent_b.id, final_outcome)
      elif isinstance(agent_b, HumanAgent) and isinstance(agent_a, RobotAgent):
        agent_b.update_robot_trust(agent_a.id, final_outcome)

  def _update_agent_states(self):
    """Update agent psychological states"""
    for agent in self.agents:
      # Calculate workload based on active activities
      active_activities = [a for a in self.activities
                          if agent.id in a.participants and not a.completed]
      workload = min(1.0, len(active_activities) / 3)

      # Calculate social support based on trust
      trust_values = list(agent.trust_network.values())
      social_support = np.mean(trust_values) if trust_values else 0.5

      agent.update_state(workload, social_support)

      # Fatigue recovery for humans
      if isinstance(agent, HumanAgent) and workload < 0.3:
        agent.fatigue_factor = min(1.0, agent.fatigue_factor + 0.01)

  def run_simulation(self, total_cycles: int = 1000) -> Dict:
    """Execute complete simulation"""
    print(f"   Executing {total_cycles} cycles...", end = "")

    start_time = time.time()

    for cycle in range(total_cycles):
      self.step()

      # Progress indicator
      if cycle % 100 == 0 and cycle > 0:
        print(".", end = "")

    execution_time = time.time() - start_time
    print(f" {execution_time:.1f}s")

    return {
      'data_collector': self.data_collector,
      'agents': self.agents,
      'completed_activities': self.completed_activities,
      'execution_time': execution_time,
      'final_metrics': self.data_collector.metrics_history[-1] if self.data_collector.metrics_history else {}
    }
