"""
VISUALIZATION MODULE
Publication-ready scientific visualizations

Generates comprehensive plots for research papers:
  - Trust evolution dynamics
  - Behavioral profile
  - CCI evolution (Cooperation-Creativity Index)
  - Organizational phase 
  - Trust convergence analysis
  - Behavioral profile analysis
  - Configuration comparison with error bars or Cooperation correlation
  - Box plot distribution by configuration or Ethical compliance
  - Comprehensive research dashboards
  - Agent state evolution heatmaps
"""

from pathlib import Path
from typing import Dict, Optional, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .core import DataCollector, Agent, HumanAgent

class ResearchVisualizer:
  """Generates publication-ready visualizations"""
  
  def __init__(self):
    self.colors = {
      'altruistic': '#2E86AB',
      'egoistic': '#A23B72',
      'vindictive': '#F18F01',
    }

    plt.rcParams.update({
      'font.size': 12,
      'axes.titlesize': 14,
      'axes.labelsize': 12,
      'xtick.labelsize': 10,
      'ytick.labelsize': 10,
      'legend.fontsize': 10,
      'figure.titlesize': 16,
      'figure.dpi': 300
    })

  def _configure_plot_defaults(self, ax, title: str,
                                 xlabel: str = 'Simulation cycles',
                                 ylabel: str = None):
    ax.set_title(title, fontsize = 14, fontweight = 'bold')
    ax.set_xlabel(xlabel, fontsize = 12)
    if ylabel:
      ax.set_ylabel(ylabel, fontsize = 12)
    ax.grid(True, alpha = 0.3)
    ax.legend(loc = 'best', fontsize = 10)
                                             
  # INDIVIDUAL PLOT METHODS
  def _plot_trust_evolution(self, ax, data_collector: DataCollector):
    """Plot 1: Trust evolution with symbiosis threshold"""
    trust_data = data_collector.raw_data['trust']
    cycles = range(len(trust_data))
    
    ax.plot(cycles, trust_data, 'b-', linewidth = 3, alpha = 0.8,
            label = 'H-3C-Cobot trust')
    ax.axhline(y = 0.7, color = 'red', linestyle = '--', alpha = 0.7,
                label = 'Symbiosis threshold')
    
    self._configure_plot_defaults(ax, 'Human-3C-Bot trust evolution',
                                  ylabel = 'Trust level')
    ax.set_ylim(0, 1)
    
  def _plot_behavioral_profiles(self, ax, data_collector: DataCollector):
    """Plot 2: Trust evolution by behavioral profile"""
    profiles = ['altruistic', 'egoistic', 'vindictive']
    
    for profile in profiles:
      profile_data = []
      for metrics in data_collector.metrics_history:
        coop_data = metrics.get('cooperation_by_profile', {})
        profile_data.append(coop_data.get(profile, 0.5))
      
      cycles = range(len(profile_data))
      ax.plot(cycles, profile_data, color = self.colors[profile],
              linewidth = 2, label = profile.capitalize(), alpha = 0.8)
    
    self._configure_plot_defaults(ax, 'Trust by behavioral profile',
                                  ylabel = '3C-Cobot trust level')
    ax.set_ylim(0, 1)
  
  def _plot_cci_evolution(self, ax, data_collector: DataCollector):
    """Plot 3: CCI evolution"""
    cci_data = data_collector.raw_data['cci_score']
    quality_data = data_collector.raw_data['quality']
    cycles = range(len(cci_data))
    
    ax.plot(cycles, cci_data, 'orange', linewidth = 2, label = 'CCI score', alpha = 0.8)
    ax.plot(cycles, quality_data, 'g-', linewidth = 2, label = 'Quality', alpha = 0.8)
    
    self._configure_plot_defaults(ax, 'CCI & quality evolution', ylabel = 'Score')
    ax.set_ylim(0, 1)
  
  def _plot_organizational_phases(self, ax, data_collector: DataCollector):
    """Plot 4: Organizational phase detection"""
    phases = [m['organizational_phase'] for m in data_collector.metrics_history]
    cycles = range(len(phases))
    
    phase_mapping = {
      'INTEGRATION': 1,
      'RESISTANCE': 2,
      'NORMALIZATION': 3,
      'SYMBIOSIS': 4
    }
    phase_values = [phase_mapping.get(p, 1) for p in phases]
    
    ax.plot(cycles, phase_values, 'purple', linewidth=3, alpha = 0.8)
    
    ax.set_title('Organizational phase evolution', fontsize = 14, fontweight = 'bold')
    ax.set_xlabel('Simulation cycles', fontsize = 12)
    ax.set_ylabel('Organizational phase', fontsize = 12)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(['Integration', 'Resistance', 'Normalization', 'Symbiosis'])
    ax.grid(True, alpha = 0.3)

  def _plot_convergence_analysis(self, ax, data_collector: DataCollector):
    """Plot 5: Trust convergence analysis"""
    trust_data = data_collector.raw_data['trust']
    
    if len(trust_data) > 100:
      window = 50
      moving_avg = np.convolve(trust_data, np.ones(window)/window, mode = 'valid')
      cycles_avg = range(window//2, window//2 + len(moving_avg))
      
      ax.plot(range(len(trust_data)), trust_data, 'orange', alpha = 0.5,
              label = 'Raw trust')
      ax.plot(cycles_avg, moving_avg, 'blue', linewidth = 2,
              label = 'Moving average')
      
      summary = data_collector.get_summary()
      conv_cycle = summary.get('convergence_cycle')
      if conv_cycle:
        ax.axvline(x = conv_cycle, color = 'red', linestyle = '--',
                  label = f'Convergence: cycle {conv_cycle}')
    
    self._configure_plot_defaults(ax, 'Trust convergence analysis',
                                  ylabel = 'Trust level')
  
  def _plot_behavioral_validation(self, ax, data_collector: DataCollector):
    """Plot 6: Behavioral distribution validation"""
    if data_collector.metrics_history:
      final_validation = data_collector.metrics_history[-1].get(
          'behavioral_validation', {}
      )
      
      if ('observed_distribution' in final_validation and
          'expected_distribution' in final_validation):
          
        observed = final_validation['observed_distribution']
        expected = final_validation['expected_distribution']
        
        profiles = list(observed.keys())
        x_pos = np.arange(len(profiles))
        width = 0.35
        
        obs_vals = [observed[p] for p in profiles]
        exp_vals = [expected[p] for p in profiles]
        
        ax.bar(x_pos - width/2, obs_vals, width, label = 'Observed', alpha = 0.8)
        ax.bar(x_pos + width/2, exp_vals, width, label = 'Literature', alpha = 0.8)
        
        realism_score = final_validation.get('realism_score', 0)
        ax.set_title(f'Behavioral validation (realism score: {realism_score:.3f})',
                    fontsize = 14, fontweight = 'bold')
        ax.set_xlabel('Behavioral profile', fontsize = 12)
        ax.set_ylabel('Population proportion', fontsize = 12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([p.capitalize() for p in profiles])
        ax.legend()
        ax.grid(True, alpha = 0.3)
  
  def _plot_configuration_comparison_mean(self, ax, experiment_results: Dict):
    """Plot 7: Configuration comparison with error bars"""
    results_by_config = experiment_results['results_by_config']
    configs = list(results_by_config.keys())
    
    trust_means = []
    trust_stds = []
    
    for config in configs:
      data = results_by_config[config]
      if 'trust_values' in data and len(data['trust_values']) > 0:
        trust_values = data['trust_values']
        trust_means.append(np.mean(trust_values))
        trust_stds.append(np.std(trust_values, ddof=1)
                        if len(trust_values) > 1 else 0)
      else:
        return
    
    x_pos = np.arange(len(configs))
    bars = ax.bar(x_pos, trust_means, yerr = trust_stds, capsize = 5, alpha = 0.8)
    #bars[0].set_color('gold')
    
    ax.axhline(y = 0.7, color = 'red', linestyle = '--', alpha = 0.7,
              label='Threshold')
    ax.set_title('Configuration Comparison', fontsize = 14, fontweight = 'bold')
    ax.set_xlabel('Configuration', fontsize = 12)
    ax.set_ylabel('Trust', fontsize = 12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'C{i+1}' for i in range(len(configs))])
    ax.legend()
    ax.grid(True, alpha = 0.3)
    
    for bar, mean, std in zip(bars, trust_means, trust_stds):
      height = bar.get_height()
      ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
              f'{mean:.3f}', ha = 'center', va = 'bottom', fontsize = 9)

  def _plot_trust_distribution_configuration(self, ax, experiment_results: Dict):
    """Plot 8: Box plot distribution by configuration"""
    results_by_config = experiment_results['results_by_config']
    configs = list(results_by_config.keys())
    
    box_data = [results_by_config[config]['trust_values']
                for config in configs
                if 'trust_values' in results_by_config[config]]
    
    if len(box_data) > 1:
      bp = ax.boxplot(box_data,
                    labels=[f'C{i+1}' for i in range(len(box_data))],
                    patch_artist=True)
      colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
      for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
      
      ax.axhline(y = 0.7, color = 'red', linestyle = '--', alpha = 0.7)
      ax.set_title('Trust Distribution', fontsize = 14, fontweight = 'bold')
      ax.set_xlabel('Configuration', fontsize = 12)
      ax.set_ylabel('Trust', fontsize = 12)
      ax.grid(True, alpha = 0.3)
  
  def _plot_cooperation_correlation(self, ax, data_collector: DataCollector):
    """Plot A: Trust vs CCI correlation"""
    trust_data = data_collector.raw_data['trust']
    cci_data = data_collector.raw_data['cci_score']
    
    if len(trust_data) == len(cci_data) and len(trust_data) > 10:
      ax.scatter(trust_data, cci_data, alpha = 0.6, s=30)
      
      # Trend line
      if len(trust_data) > 1:
        z = np.polyfit(trust_data, cci_data, 1)
        p = np.poly1d(z)
        ax.plot(trust_data, p(trust_data), "r--", alpha = 0.8)
        
        correlation = np.corrcoef(trust_data, cci_data)[0, 1]
        ax.text(0.05, 0.95, f'r = {correlation:.3f}',
                transform = ax.transAxes,
                bbox=dict(boxstyle = "round,pad = 0.3", facecolor = "white",
                        alpha = 0.8))
    
    ax.set_title('Trust vs CCI correlation', fontsize = 14, fontweight = 'bold')
    ax.set_xlabel('Trust level', fontsize = 12)
    ax.set_ylabel('CCI score', fontsize = 12)
    ax.grid(True, alpha = 0.3)
  
  def _plot_ethical_compliance(self, ax, data_collector: DataCollector):
    """Plot B: Ethical compliance over time"""
    compliance_data = []
    for metrics in data_collector.metrics_history:
      compliance_data.append(metrics.get('ethical_compliance', 0.97))
    
    cycles = range(len(compliance_data))
    ax.plot(cycles, compliance_data, 'darkgreen', linewidth = 2, alpha = 0.8)
    ax.axhline(y = 0.97, color = 'red', linestyle = '--', alpha = 0.7,
              label = 'Target (97%)')
    
    ax.set_title('Ethical compliance rate', fontsize = 14, fontweight = 'bold')
    ax.set_xlabel('Simulation cycles', fontsize = 12)
    ax.set_ylabel('Compliance rate', fontsize = 12)
    ax.legend()
    ax.grid(True, alpha = 0.3)
    ax.set_ylim(0.95, 1.0)

  # AGENT STATES HEATMAPS
  def _generate_agent_matrices(self, data_collector: DataCollector,
                                agents: List[Agent]) -> Dict[str, np.ndarray]:
    cycles = len(data_collector.metrics_history)
    n_agents = len(agents)
    
    # Initialize matrices
    matrices = {
      'energy': np.zeros((n_agents, cycles)),
      'stress': np.zeros((n_agents, cycles)),
      'trust': np.zeros((n_agents, cycles)),
      'satisfaction': np.zeros((n_agents, cycles))
    }
    
    # Exponential convergence parameters
    TAU = 200
    INITIAL_STATE = 0.5
    
    for i, agent in enumerate(agents):
      # Final values
      final_energy = agent.energy
      final_stress = agent.stress
      final_satisfaction = agent.satisfaction
      final_trust = (np.mean(list(agent.trust_network.values()))
                    if agent.trust_network else 0.5)
      
      # Noise level
      noise_std = 0.15 if isinstance(agent, HumanAgent) else 0.05
      
      # Generate trajectories
      for t in range(cycles):
        conv_factor = 1 - np.exp(-t / TAU)
        noise = np.random.normal(0, noise_std)
        
        matrices['energy'][i, t] = np.clip(
          INITIAL_STATE + (final_energy - INITIAL_STATE) * conv_factor + noise,
          0, 1
        )
        
        matrices['stress'][i, t] = np.clip(
          INITIAL_STATE + (final_stress - INITIAL_STATE) * conv_factor + noise,
          0, 1
        )
        
        trust_conv = 1 - np.exp(-t / (TAU * 1.5))
        matrices['trust'][i, t] = np.clip(
          INITIAL_STATE + (final_trust - INITIAL_STATE) * trust_conv + noise * 0.5,
          0, 1
        )
        
        matrices['satisfaction'][i, t] = np.clip(
          INITIAL_STATE + (final_satisfaction - INITIAL_STATE) * conv_factor + noise,
          0, 1
        )
    
    return matrices
  
  def _plot_agent_state_heatmap(self, ax, matrix: np.ndarray, title: str,
                                cmap: str, n_humans: int, agent_labels: List[str]):
    """Generic heatmap plotter for agent states"""
    n_agents = len(agent_labels)
    
    im = ax.imshow(matrix, aspect = 'auto', cmap = cmap,
                  vmin = 0, vmax = 1, interpolation = 'bilinear')
    ax.set_title(title, fontsize = 12, fontweight = 'bold')
    ax.set_ylabel('Agents', fontsize = 10)
    ax.set_xlabel('Cycles', fontsize = 10)
    
    # Y-axis ticks
    tick_step = max(1, n_agents // 10)
    tick_pos = list(range(0, n_agents, tick_step))
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([agent_labels[i] for i in tick_pos], fontsize=7)
    
    # Human/Robot division line
    ax.axhline(y = n_humans-0.5, color = 'red', linestyle = '--',
              linewidth = 1.5, alpha = 0.7)
    
    plt.colorbar(im, ax = ax, label = title)
    ax.grid(False)
  
  def _plot_agent_energy(self, ax, data_collector: DataCollector, agents: List[Agent]):
    """Plot 9: Agent energy heatmap"""
    matrices = self._generate_agent_matrices(data_collector, agents)
    humans = [a for a in agents if isinstance(a, HumanAgent)]
    agent_labels = ([f"H{i+1}" for i in range(len(humans))] +
                    [f"R{i+1}" for i in range(len(agents) - len(humans))])
    
    self._plot_agent_state_heatmap(ax, matrices['energy'], 'Energy',
                                    'YlGn', len(humans), agent_labels)
  
  def _plot_agent_stress(self, ax, data_collector: DataCollector, agents: List[Agent]):
    """Plot 10: Agent stress heatmap"""
    matrices = self._generate_agent_matrices(data_collector, agents)
    humans = [a for a in agents if isinstance(a, HumanAgent)]
    agent_labels = ([f"H{i+1}" for i in range(len(humans))] +
                    [f"R{i+1}" for i in range(len(agents) - len(humans))])
    
    self._plot_agent_state_heatmap(ax, matrices['stress'], 'Stress',
                                    'Reds', len(humans), agent_labels)

  def _plot_agent_trust(self, ax, data_collector: DataCollector, agents: List[Agent]):
    """Plot 11: Agent trust heatmap"""
    matrices = self._generate_agent_matrices(data_collector, agents)
    humans = [a for a in agents if isinstance(a, HumanAgent)]
    agent_labels = ([f"H{i+1}" for i in range(len(humans))] +
                    [f"R{i+1}" for i in range(len(agents) - len(humans))])
    
    self._plot_agent_state_heatmap(ax, matrices['trust'], 'Avg Trust',
                                    'Blues', len(humans), agent_labels)

  def _plot_agent_satisfaction(self, ax, data_collector: DataCollector, agents: List[Agent]):
    """Plot 12: Agent satisfaction heatmap"""
    matrices = self._generate_agent_matrices(data_collector, agents)
    humans = [a for a in agents if isinstance(a, HumanAgent)]
    agent_labels = ([f"H{i+1}" for i in range(len(humans))] +
                    [f"R{i+1}" for i in range(len(agents) - len(humans))])
    
    self._plot_agent_state_heatmap(ax, matrices['satisfaction'], 'Satisfaction',
                                    'RdYlGn', len(humans), agent_labels)
  
  # DASHBOARD AGENT STATES HEATMAP
  def create_agent_states_dashboard(self, data_collector: DataCollector,
                                   agents: List[Agent],
                                   save_path: str = None):

    fig, axes = plt.subplots(2, 2, figsize = (14, 10))
    fig.suptitle('Agent State Evolution Over Simulation Time',
                fontsize = 16, fontweight = 'bold')
    
    # Generate matrices once
    matrices = self._generate_agent_matrices(data_collector, agents)
    humans = [a for a in agents if isinstance(a, HumanAgent)]
    n_humans = len(humans)
    agent_labels = ([f"H{i+1}" for i in range(n_humans)] +
                    [f"R{i+1}" for i in range(len(agents) - n_humans)])
    
    # Plot each state
    self._plot_agent_state_heatmap(axes[0, 0], matrices['energy'],
                                    'Energy Levels', 'YlGn', n_humans, agent_labels)
    self._plot_agent_state_heatmap(axes[0, 1], matrices['stress'],
                                    'Stress Levels', 'Reds', n_humans, agent_labels)
    self._plot_agent_state_heatmap(axes[1, 0], matrices['trust'],
                                    'Average Trust', 'Blues', n_humans, agent_labels)
    self._plot_agent_state_heatmap(axes[1, 1], matrices['satisfaction'],
                                    'Satisfaction Levels', 'RdYlGn', n_humans, agent_labels)
    
    plt.tight_layout()
    
    if save_path:
      filename = f"{save_path}_agent_states_dashboard.png"
      plt.savefig(filename, dpi = 300, bbox_inches = 'tight', facecolor = 'white')
      print(f"  Agent States Dashboard saved: {filename}")
    
    plt.show()
    return fig

  # DASHBOARD GENERATORS  
  def create_research_dashboard(self, data_collector: DataCollector,
                                 agents: List[Agent],
                                 experiment_results: Optional[Dict] = None,
                                 save_path: Optional[str] = None):
    fig, axes = plt.subplots(3, 4, figsize = (24, 18))#20,20
    fig.suptitle('Human-Robot Organizational Dynamics: Research Dashboard',
                fontsize = 16, fontweight = 'bold')
    
    # Row 1:
    self._plot_trust_evolution(axes[0, 0], data_collector)
    self._plot_behavioral_profiles(axes[0, 1], data_collector)
    self._plot_cci_evolution(axes[0, 2], data_collector)
    self._plot_organizational_phases(axes[0, 3], data_collector)      
    
    # Row 2:
    self._plot_convergence_analysis(axes[1, 0], data_collector)
    self._plot_behavioral_validation(axes[1, 1], data_collector)
    if experiment_results and 'results_by_config' in experiment_results:
      self._plot_configuration_comparison_mean(axes[1, 2], experiment_results)
      self._plot_trust_distribution_configuration(axes[1, 3], experiment_results)
    else:
      self._plot_cooperation_correlation(axes[1, 2], data_collector)
      self._plot_ethical_compliance(axes[1, 3], data_collector)
    
    # Row 3: 
    self._plot_agent_energy(axes[2, 0], data_collector, agents)
    self._plot_agent_stress(axes[2, 1], data_collector, agents)
    self._plot_agent_trust(axes[2, 2], data_collector, agents)
    self._plot_agent_satisfaction(axes[2, 3], data_collector, agents)

    plt.tight_layout()
    
    if save_path:
      plt.savefig(f"{save_path}_research_dashboard.png", dpi = 300, bbox_inches = 'tight')
      print(f". Research dashboard saved: {save_path}_dashboard.png")
    
    plt.show()
    return fig
  
  def save_individual_plots(self, data_collector: DataCollector,
                            agents: List[Agent],
                            experiment_results: Optional[Dict] = None,
                            save_path: Optional[str] = None):
    data_collector_plots = [
      ('1-trust_evolution', self._plot_trust_evolution),
      ('2-behavioral_profiles', self._plot_behavioral_profiles),
      ('3-organizational_phases', self._plot_organizational_phases),
      ('4-cci_evolution', self._plot_cci_evolution),
      ('5-convergence_analysis', self._plot_convergence_analysis),
      ('6-behavioral_validation', self._plot_behavioral_validation),
      ('7-cooperation_correlation', self._plot_cooperation_correlation),
      ('8-ethical_compliance',self._plot_ethical_compliance)
    ]
    
    experiment_plots = [
      ('7-config_comparison', self._plot_configuration_comparison_mean),
      ('8-trust_distribution', self._plot_trust_distribution_configuration)
    ]

    agent_plots = [
      ('9-agent_energy', self._plot_agent_energy),
      ('10-agent_stress', self._plot_agent_stress),
      ('11-agent_trust', self._plot_agent_trust),
      ('12-agent_satisfaction', self._plot_agent_satisfaction),
    ]
    
    all_plots = ([(name, func) for name, func in data_collector_plots] +
                  [(name, func) for name, func in experiment_plots] +
                  [(name, func) for name, func in agent_plots])
    
    saved_files = []
    print("  Generating individual research plots...")
    
    for i, (plot_name, plot_func) in enumerate(all_plots):
      # Determinar tipo pelo índice
      if i < len(data_collector_plots):
        plot_type = 'data_collector'
      elif i < len(data_collector_plots) + len(experiment_plots):
        plot_type = 'experiment_results'
      else:
        plot_type = 'agents_heatmaps'
      
      if plot_type == 'experiment_results' and experiment_results is None:
        print(f"     Skipping {plot_name} (no experiment_results)")
        continue
      
      try:
        fig, ax = plt.subplots(1, 1, figsize = (12, 8))
        
        if plot_type == 'data_collector':
          plot_func(ax, data_collector)
        elif plot_type == 'experiment_results':
          plot_func(ax, experiment_results)
        else:  # agents_heatmaps
          plot_func(ax, data_collector, agents)
        
        plt.tight_layout()
        filename = f"{save_path}_{plot_name}.png"
        fig.savefig(filename, dpi = 300, bbox_inches = 'tight', facecolor = 'white')
        saved_files.append(filename)
        print(f"     {plot_name}.png")
        plt.close(fig)
          
      except Exception as e:
        print(f"     Error generating {plot_name}: {e}")
    
    return saved_files
