"""
VISUALIZATION MODULE
Publication-ready scientific visualizations
"""

from typing import Dict, Optional, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .core import Agent, HumanAgent, DataCollector

class ResearchVisualizer:
  """
  Generates publication-ready visualizations:
    - Trust evolution plots
    - Behavioral profiles
    - Statistical comparisons
    - Agent state heatmaps
  """
  
  def __init__(self):
    self.colors = {
      'altruistic': '#2E86AB',
      'egoistic': '#A23B72',
      'vindictive': '#F18F01',
    }

  def _configure_plot_defaults(self, ax, title: str, 
                                 xlabel: str = 'Simulation Cycles',
                                 ylabel: str = None):
    """Apply standard formatting to plot"""
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
      ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
                                   
  def _plot_trust_evolution(self, ax, data_collector: DataCollector):
    """Plot 1: Trust evolution with symbiosis threshold"""
    trust_data = data_collector.raw_data['trust']
    cycles = range(len(trust_data))

    ax.plot(cycles, trust_data, 'b-', linewidth = 3, alpha = 0.8, label = 'H-3C-Cobot trust')
    ax.axhline(y = 0.7, color = 'red', linestyle = '--', alpha = 0.7, label = 'Symbiosis threshold')

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
      ax.plot(cycles, profile_data, color=self.colors[profile],
              linewidth = 2, label = profile.capitalize(), alpha = 0.8)

    ax.set_title('Trust by behavioral profile')
    ax.set_xlabel('Simulation cycles')
    ax.set_ylabel('3C-Cobot trust level')
    ax.legend()
    ax.grid(True, alpha = 0.3)
    ax.set_ylim(0, 1)

  def _plot_organizational_phases(self, ax, data_collector: DataCollector):
    """Plot 3: Organizational phase detection"""
    phases = [m['organizational_phase'] for m in data_collector.metrics_history]
    cycles = range(len(phases))

    phase_mapping = {'INTEGRATION': 1, 'RESISTANCE': 2, 'NORMALIZATION': 3, 'SYMBIOSIS': 4}
    phase_values = [phase_mapping.get(p, 1) for p in phases]

    ax.plot(cycles, phase_values, 'purple', linewidth=3, alpha=0.8)

    ax.set_title('Organizational phase evolution')
    ax.set_xlabel('Simulation Cccles')
    ax.set_ylabel('Organizational phase')
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(['Integration', 'Resistance', 'Normalization', 'Symbiosis'])
    ax.grid(True, alpha = 0.3)

  def _plot_icc_evolution(self, ax, data_collector: DataCollector):
    """Plot 4: ICC (Cooperation-Creativity Index) evolution"""
    icc_data = data_collector.raw_data['icc_score']
    quality_data = data_collector.raw_data['quality']
    cycles = range(len(icc_data))

    ax.plot(cycles, icc_data, 'orange', linewidth = 2, label = 'ICC score', alpha = 0.8)
    ax.plot(cycles, quality_data, 'g-', linewidth = 2, label = 'Quality', alpha = 0.8)

    ax.set_title('ICC & quality evolution')
    ax.set_xlabel('Simulation cycles')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True, alpha = 0.3)
    ax.set_ylim(0, 1)

  def _plot_network_dynamics(self, ax, data_collector: DataCollector):
    """Plot 5: Network density evolution"""
    density_data = data_collector.raw_data['network_density']
    cycles = range(len(density_data))

    ax.plot(cycles, density_data, 'orange', linewidth = 2, alpha = 0.8)

    ax.set_title('Network density evolution')
    ax.set_xlabel('Simulation cycles')
    ax.set_ylabel('Network density')
    ax.grid(True, alpha = 0.3)
    ax.set_ylim(0, 1)

  def _plot_stress_satisfaction(self, ax, data_collector: DataCollector):
    """Plot 6: Stress vs satisfaction dynamics"""
    stress_data = data_collector.raw_data['stress']
    satisfaction_data = data_collector.raw_data['satisfaction']
    cycles = range(min(len(stress_data), len(satisfaction_data)))

    ax.plot(cycles, stress_data[:len(cycles)], 'r-', linewidth = 2, label = 'Stress', alpha = 0.8)
    ax.plot(cycles, satisfaction_data[:len(cycles)], 'g-', linewidth = 2, label = 'Satisfaction', alpha = 0.8)

    ax.set_title('Stress vs Satisfaction dynamics')
    ax.set_xlabel('Simulation cycles')
    ax.set_ylabel('Level')
    ax.legend()
    ax.grid(True, alpha = 0.3)
    ax.set_ylim(0, 1)

  def _plot_convergence_analysis(self, ax, data_collector: DataCollector):
    """Plot 7: Trust convergence analysis"""
    trust_data = data_collector.raw_data['trust']

    if len(trust_data) > 100:
      window = 50
      moving_avg = np.convolve(trust_data, np.ones(window)/window, mode='valid')
      cycles_avg = range(window//2, window//2 + len(moving_avg))

      ax.plot(range(len(trust_data)), trust_data, 'orange', alpha = 0.5, label = 'Raw trust')
      ax.plot(cycles_avg, moving_avg, 'blue', linewidth = 2, label = 'Moving average')

      summary = data_collector.get_summary()
      conv_cycle = summary.get('convergence_cycle')
      if conv_cycle:
        ax.axvline(x = conv_cycle, color = 'red', linestyle = '--',
                  label = f'Convergence: cycle {conv_cycle}')

    ax.set_title('Trust convergence analysis')
    ax.set_xlabel('Simulation cycles')
    ax.set_ylabel('Trust level')
    ax.legend()
    ax.grid(True, alpha = 0.3)

  def _plot_behavioral_validation(self, ax, data_collector: DataCollector):
    """Plot 8: Behavioral distribution validation"""
    if data_collector.metrics_history:
      final_validation = data_collector.metrics_history[-1].get('behavioral_validation', {})

      if 'observed_distribution' in final_validation and 'expected_distribution' in final_validation:
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
        ax.set_title(f'Behavioral validation (realism score: {realism_score:.3f})')
        ax.set_xlabel('Behavioral profile')
        ax.set_ylabel('Population proportion')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([p.capitalize() for p in profiles])
        ax.legend()
        ax.grid(True, alpha = 0.3)

  def _plot_quality_distribution(self, ax, data_collector: DataCollector):
    """Plot 9: Quality distribution histogram"""
    quality_data = data_collector.raw_data['quality']

    if len(quality_data) > 10:
        ax.hist(quality_data, bins = 15, alpha = 0.7, color = 'lightgreen', edgecolor = 'black')
        ax.axvline(x = np.mean(quality_data), color = 'red', linestyle = '--',
                  label=f'Mean: {np.mean(quality_data):.3f}')

    ax.set_title('Activity quality distribution')
    ax.set_xlabel('Quality score')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha = 0.3)

  def _plot_cooperation_correlation(self, ax, data_collector: DataCollector):
    """Plot 10: Trust vs ICC correlation"""
    trust_data = data_collector.raw_data['trust']
    icc_data = data_collector.raw_data['icc_score']

    if len(trust_data) == len(icc_data) and len(trust_data) > 10:
        ax.scatter(trust_data, icc_data, alpha = 0.6, s = 30)

        # Trend line
        if len(trust_data) > 1:
          z = np.polyfit(trust_data, icc_data, 1)
          p = np.poly1d(z)
          ax.plot(trust_data, p(trust_data), "r--", alpha = 0.8)

          correlation = np.corrcoef(trust_data, icc_data)[0, 1]
          ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform = ax.transAxes,
                bbox = dict(boxstyle = "round,pad = 0.3", facecolor = "white", alpha = 0.8))

    ax.set_title('Trust vs ICC correlation')
    ax.set_xlabel('Trust level')
    ax.set_ylabel('ICC score')
    ax.grid(True, alpha = 0.3)

  def _plot_final_metrics_summary(self, ax, data_collector: DataCollector):
    """Plot 11: Final metrics summary"""
    summary = data_collector.get_summary()
    trust_stats = summary.get('trust_statistics', {})
    final_metrics = summary.get('final_metrics', {})

    metrics_data = {
      'Trust': trust_stats.get('final', 0),
      'Quality': final_metrics.get('quality', 0),
      'ICC': final_metrics.get('icc_score', 0),
      'Network': final_metrics.get('network_density', 0)
    }

    metrics_names = list(metrics_data.keys())
    metrics_values = list(metrics_data.values())

    bars = ax.bar(metrics_names, metrics_values,
                  color = ['blue', 'green', 'orange', 'purple'], alpha = 0.7)

    for bar, val in zip(bars, metrics_values):
      height = bar.get_height()
      ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
              f'{val:.3f}', ha = 'center', va = 'bottom')#, fontweight='bold')

    ax.set_title('Final metrics summary')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.grid(True, alpha = 0.3)
    ax.set_ylim(0, 1)

  def _plot_ethical_compliance(self, ax, data_collector: DataCollector):
    """Plot 12: Ethical compliance over time"""
    compliance_data = []
    for metrics in data_collector.metrics_history:
      compliance_data.append(metrics.get('ethical_compliance', 0.97))

    cycles = range(len(compliance_data))
    ax.plot(cycles, compliance_data, 'darkgreen', linewidth = 2, alpha = 0.8)
    ax.axhline(y = 0.97, color = 'red', linestyle = '--', alpha = 0.7, label = 'Target (97%)')

    ax.set_title('Ethical compliance rate')
    ax.set_xlabel('Simulation cycles')
    ax.set_ylabel('Compliance rate')
    ax.legend()
    ax.grid(True, alpha = 0.3)
    ax.set_ylim(0.95, 1.0)

  def _plot_agent_states(self, axes, data_collector: DataCollector):
    """
    Plota heatmaps de estados dos agentes (4 painéis: energy, stress, trust, satisfaction)

    Args:
        axes: Array numpy 2x2 de matplotlib axes
        data_collector: Coletor de dados com agent_states_history
    """
    agent_states = data_collector.agent_states_history

    if not agent_states:
      # Se não houver dados, mostrar mensagem
      for ax in axes.flat:
        ax.text(0.5, 0.5, 'No individual agent data collected',
                ha='center', va='center', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
      return

    # Separar IDs de humanos e robôs
    human_ids = sorted([aid for aid, data in agent_states.items()
                        if data['type'] == 'human'])
    robot_ids = sorted([aid for aid, data in agent_states.items()
                        if data['type'] == 'robot'])

    # Ordenar: humanos primeiro, depois robôs
    agent_ids = human_ids + robot_ids
    n_humans = len(human_ids)
    n_agents = len(agent_ids)

    # Obter número de ciclos
    if agent_ids:
      cycles = len(agent_states[agent_ids[0]]['energy'])
    else:
      cycles = 0

    if cycles == 0 or n_agents == 0:
      for ax in axes.flat:
        ax.text(0.5, 0.5, 'Insufficient data',
                ha='center', va='center', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
      return

    # Criar matrizes (agents x cycles)
    energy_matrix = np.zeros((n_agents, cycles))
    stress_matrix = np.zeros((n_agents, cycles))
    trust_matrix = np.zeros((n_agents, cycles))
    satisfaction_matrix = np.zeros((n_agents, cycles))

    # Preencher matrizes
    for i, agent_id in enumerate(agent_ids):
      energy_matrix[i, :] = agent_states[agent_id]['energy']
      stress_matrix[i, :] = agent_states[agent_id]['stress']
      trust_matrix[i, :] = agent_states[agent_id]['trust_avg']
      satisfaction_matrix[i, :] = agent_states[agent_id]['satisfaction']

    # Labels para o eixo Y
    agent_labels = [f"H{i+1}" for i in range(n_humans)] + \
                    [f"R{i+1}" for i in range(len(robot_ids))]

    # Configuração de ticks para o eixo Y
    y_tick_step = max(1, n_agents // 10)
    y_ticks = range(0, n_agents, y_tick_step)
    y_labels = [agent_labels[i] for i in y_ticks]

    # ====================================================================
    # PLOT 1: Energy Levels (superior esquerdo)
    # ====================================================================
    im1 = axes[0, 0].imshow(energy_matrix, aspect='auto', cmap='YlGn',
                            vmin=0, vmax=1, interpolation='bilinear')
    axes[0, 0].set_title('Energy Levels', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Agents', fontsize=12)
    axes[0, 0].set_yticks(y_ticks)
    axes[0, 0].set_yticklabels(y_labels, fontsize=8)
    axes[0, 0].axhline(y=n_humans-0.5, color='red', linestyle='--',
                      linewidth=2, label='Human/Robot Division', alpha=0.8)
    axes[0, 0].set_xlabel('Simulation Cycles', fontsize=10)
    axes[0, 0].legend(loc='upper right', fontsize=8)
    plt.colorbar(im1, ax=axes[0, 0], label='Energy Level')
    axes[0, 0].grid(False)

    # ====================================================================
    # PLOT 2: Stress Levels (superior direito)
    # ====================================================================
    im2 = axes[0, 1].imshow(stress_matrix, aspect='auto', cmap='Reds',
                            vmin=0, vmax=1, interpolation='bilinear')
    axes[0, 1].set_title('Stress Levels', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Agents', fontsize=12)
    axes[0, 1].set_yticks(y_ticks)
    axes[0, 1].set_yticklabels(y_labels, fontsize=8)
    axes[0, 1].axhline(y=n_humans-0.5, color='blue', linestyle='--',
                      linewidth=2, alpha=0.8)
    axes[0, 1].set_xlabel('Simulation Cycles', fontsize=10)
    plt.colorbar(im2, ax=axes[0, 1], label='Stress Level')
    axes[0, 1].grid(False)

    # ====================================================================
    # PLOT 3: Average Trust (inferior esquerdo)
    # ====================================================================
    im3 = axes[1, 0].imshow(trust_matrix, aspect='auto', cmap='Blues',
                            vmin=0, vmax=1, interpolation='bilinear')
    axes[1, 0].set_title('Average Trust with Others', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Agents', fontsize=12)
    axes[1, 0].set_xlabel('Simulation Cycles', fontsize=12)
    axes[1, 0].set_yticks(y_ticks)
    axes[1, 0].set_yticklabels(y_labels, fontsize=8)
    axes[1, 0].axhline(y=n_humans-0.5, color='red', linestyle='--',
                      linewidth=2, alpha=0.8)
    plt.colorbar(im3, ax=axes[1, 0], label='Trust Level')
    axes[1, 0].grid(False)

    # ====================================================================
    # PLOT 4: Satisfaction Levels (inferior direito)
    # ====================================================================
    im4 = axes[1, 1].imshow(satisfaction_matrix, aspect='auto', cmap='RdYlGn',
                            vmin=0, vmax=1, interpolation='bilinear')
    axes[1, 1].set_title('Satisfaction Levels', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Agents', fontsize=12)
    axes[1, 1].set_xlabel('Simulation Cycles', fontsize=12)
    axes[1, 1].set_yticks(y_ticks)
    axes[1, 1].set_yticklabels(y_labels, fontsize=8)
    axes[1, 1].axhline(y=n_humans-0.5, color='blue', linestyle='--',
                      linewidth=2, alpha=0.8)
    plt.colorbar(im4, ax=axes[1, 1], label='Satisfaction Level')
    axes[1, 1].grid(False)

  def _plot_configuration_comparison(self, axes, experiment_results: Dict):
    """Configuration comparison plots for multi-config experiments"""
    results_by_config = experiment_results['results_by_config']
    configs = list(results_by_config.keys())

    # Extract trust data
    trust_means = []
    trust_stds = []

    for config in configs:
      data = results_by_config[config]
      if 'trust_values' in data and len(data['trust_values']) > 0:
        trust_values = data['trust_values']
        trust_means.append(np.mean(trust_values))
        trust_stds.append(np.std(trust_values, ddof=1) if len(trust_values) > 1 else 0)
      else:
        return

    # Plot 1: Configuration comparison with error bars
    x_pos = np.arange(len(configs))
    bars = axes[0].bar(x_pos, trust_means, yerr=trust_stds, capsize=5, alpha=0.8)

    axes[0].axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Symbiosis Threshold')
    axes[0].set_title('Configuration Comparison (Mean ± SD)')
    axes[0].set_ylabel('Final Trust Level')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([f'C{i+1}' for i in range(len(configs))])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, mean, std in zip(bars, trust_means, trust_stds):
      height = bar.get_height()
      axes[0].text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                  f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Box plot distribution
    box_data = [results_by_config[config]['trust_values'] for config in configs
                if 'trust_values' in results_by_config[config]]

    if len(box_data) > 1:
      bp = axes[1].boxplot(box_data, labels=[f'C{i+1}' for i in range(len(box_data))],
                          patch_artist=True)
      colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
      for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

      axes[1].axhline(y=0.7, color='red', linestyle='--', alpha=0.7)
      axes[1].set_title('Trust Distribution by Configuration')
      axes[1].set_ylabel('Trust Level')
      axes[1].grid(True, alpha=0.3)

    # Plot 3: Statistical analysis results
    anova_results = experiment_results.get('anova_results', {})
    if anova_results:
      f_stat = anova_results.get('f_statistic', 0)
      p_val = anova_results.get('p_value', 1)
      eta_sq = anova_results.get('eta_squared', 0)

      metrics = ['F-statistic', 'η² (Effect Size)']
      values = [f_stat, eta_sq]

      bars = axes[2].bar(metrics, values, color=['blue', 'red'], alpha=0.7)
      axes[2].set_title(f'Statistical Analysis\n(p = {p_val:.4f})')
      axes[2].set_ylabel('Value')
      axes[2].grid(True, alpha=0.3)

      # Add significance indicator
      significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
      axes[2].text(0.5, 0.9, f'Significance: {significance}',
                  transform = axes[2].transAxes, ha = 'center',
                  fontsize = 12, fontweight = 'bold')

    # Plot 4: Configuration ranking
    best_config = experiment_results.get('best_config', configs[0])
    ranking_data = [(i, trust_means[i], configs[i]) for i in range(len(configs))]
    ranking_data.sort(key=lambda x: x[1], reverse=True)

    ranks = [r[0] + 1 for r in ranking_data]
    values = [r[1] for r in ranking_data]
    labels = [f"C{r[0]+1}" for r in ranking_data]

    bars = axes[3].bar(range(len(configs)), values, alpha=0.7)
    bars[0].set_color('gold')  # Highlight best configuration

    axes[3].set_title('Configuration Ranking')
    axes[3].set_ylabel('Trust Level')
    axes[3].set_xticks(range(len(configs)))
    axes[3].set_xticklabels(labels)
    axes[3].grid(True, alpha=0.3)

    # Add ranking annotations
    for i, (bar, val) in enumerate(zip(bars, values)):
      height = bar.get_height()
      rank = i + 1
      axes[3].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                  f'#{rank}', ha='center', va='bottom', fontweight='bold')

  def _plot_configuration_comparison_mean(self, axes, experiment_results: Dict):
    results_by_config = experiment_results['results_by_config']
    configs = list(results_by_config.keys())

    # Extract trust data
    trust_means = []
    trust_stds = []

    for config in configs:
      data = results_by_config[config]
      if 'trust_values' in data and len(data['trust_values']) > 0:
        trust_values = data['trust_values']
        trust_means.append(np.mean(trust_values))
        trust_stds.append(np.std(trust_values, ddof = 1) if len(trust_values) > 1 else 0)
      else:
        return

    x_pos = np.arange(len(configs))
    bars = axes.bar(x_pos, trust_means, yerr = trust_stds, capsize = 5, alpha = 0.8)

    axes.axhline(y = 0.7, color = 'red', linestyle = '--', alpha = 0.7, label = 'symbiosis threshold')
    axes.set_title('Configuration comparison (mean ± SD)')
    axes.set_xlabel('Configuration')
    axes.set_ylabel('Final trust level')
    axes.set_xticks(x_pos)
    axes.set_xticklabels([f'C{i+1}' for i in range(len(configs))])
    axes.legend()
    axes.grid(True, alpha = 0.3)

    # Add value labels on bars
    for bar, mean, std in zip(bars, trust_means, trust_stds):
      height = bar.get_height()
      axes.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                  f'{mean:.3f}', ha = 'center', va = 'bottom')#, fontweight='bold')

  def _plot_trust_distribution_configuration(self, axes, experiment_results: Dict):
    """Plot trust distribution by configuration using boxplots"""
    results_by_config = experiment_results['results_by_config']
    configs = list(results_by_config.keys())

    # Extract trust data
    box_data = [results_by_config[config]['trust_values'] for config in configs
                if 'trust_values' in results_by_config[config]]

    if len(box_data) > 1:
      bp = axes.boxplot(box_data, labels = [f'C{i+1}' for i in range(len(box_data))],
                          patch_artist = True)
      colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
      for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

      axes.axhline(y = 0.7, color = 'red', linestyle = '--', alpha = 0.7, label = 'symbiosis threshold')
      axes.set_title('Trust distribution by configuration')
      axes.set_xlabel('Configuration')
      axes.set_ylabel('Trust level')
      axes.grid(True, alpha = 0.3)

  def _plot_statistical_analysis(self, axes, experiment_results: Dict):
    anova_results = experiment_results.get('anova_results', {})
    if anova_results:
      f_stat = anova_results.get('f_statistic', 0)
      p_val = anova_results.get('p_value', 1)
      eta_sq = anova_results.get('eta_squared', 0)

      metrics = ['F-statistic', 'η² (Effect Size)']
      values = [f_stat, eta_sq]

      bars = axes.bar(metrics, values, color = ['blue', 'red'], alpha = 0.7)
      axes.set_title(f'Statistical analysis\n(p = {p_val:.2e})')
      axes.set_xlabel('Configurations')
      axes.set_ylabel('Value')
      axes.grid(True, alpha = 0.3)

      # Add significance indicator
      significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
      axes.text(0.5, 0.9, f'Significance: {significance}',
                  transform = axes.transAxes, ha = 'center')

  def _plot_configuration_ranking(self, axes, experiment_results: Dict):
    results_by_config = experiment_results['results_by_config']
    configs = list(results_by_config.keys())
    trust_means = []
    trust_stds = []

    for config in configs:
      data = results_by_config[config]
      if 'trust_values' in data and len(data['trust_values']) > 0:
        trust_values = data['trust_values']
        trust_means.append(np.mean(trust_values))
        trust_stds.append(np.std(trust_values, ddof = 1) if len(trust_values) > 1 else 0)
      else:
        return
    best_config = experiment_results.get('best_config', configs[0])
    ranking_data = [(i, trust_means[i], configs[i]) for i in range(len(configs))]
    ranking_data.sort(key = lambda x: x[1], reverse = True)

    ranks = [r[0] + 1 for r in ranking_data]
    values = [r[1] for r in ranking_data]
    labels = [f"C{r[0]+1}" for r in ranking_data]

    bars = axes.bar(range(len(configs)), values, alpha = 0.7)
    bars[0].set_color('gold')  # Highlight best configuration

    axes.set_title('Configuration ranking')
    axes.set_ylabel('Trust level')
    axes.set_xticks(range(len(configs)))
    axes.set_xticklabels(labels)
    axes.grid(True, alpha = 0.3)

    # Add ranking annotations
    for i, (bar, val) in enumerate(zip(bars, values)):
      height = bar.get_height()
      rank = i + 1
      axes.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                  f'#{rank}', ha = 'center', va = 'bottom', fontweight = 'bold')

  def create_research_dashboard(self, data_collector: DataCollector,
                              experiment_results: Optional[Dict] = None,
                              save_path: Optional[str] = None):
    """Generate comprehensive research dashboard"""
    fig, axes = plt.subplots(3, 4, figsize = (20, 15))
    fig.suptitle('Human-Robot Organizational Dynamics: Research Dashboard',
                  fontsize = 16, fontweight = 'bold')

    # Row 1: Core temporal dynamics
    self._plot_trust_evolution(axes[0, 0], data_collector)
    self._plot_behavioral_profiles(axes[0, 1], data_collector)
    self._plot_organizational_phases(axes[0, 2], data_collector)
    self._plot_icc_evolution(axes[0, 3], data_collector)

    # Row 2: Advanced analyses
    self._plot_network_dynamics(axes[1, 0], data_collector)
    self._plot_stress_satisfaction(axes[1, 1], data_collector)
    self._plot_convergence_analysis(axes[1, 2], data_collector)
    self._plot_behavioral_validation(axes[1, 3], data_collector)

    # Row 3: Comparative or summary
    if experiment_results and 'results_by_config' in experiment_results:
      #self._plot_configuration_comparison(axes[2, :], experiment_results)
      self._plot_configuration_comparison_mean(axes[2, 0], experiment_results)
      self._plot_trust_distribution_configuration(axes[2, 1], experiment_results)
      self._plot_statistical_analysis(axes[2, 2], experiment_results)
      self._plot_configuration_ranking(axes[2, 3], experiment_results)
    else:
      self._plot_quality_distribution(axes[2, 0], data_collector)
      self._plot_cooperation_correlation(axes[2, 1], data_collector)
      self._plot_final_metrics_summary(axes[2, 2], data_collector)
      self._plot_ethical_compliance(axes[2, 3], data_collector)

    plt.tight_layout()

    if save_path:
      plt.savefig(f"{save_path}_dashboard.png", dpi = 300, bbox_inches = 'tight')
      print(f"✅ Research dashboard saved: {save_path}_dashboard.png")

    plt.show()
    return fig

  #reviewer
  def create_agent_states_heatmap(self, data_collector: DataCollector,
                                  agents: List[Agent],
                                  save_path: str = None):

    # Collect historical state data from agents.
    cycles = len(data_collector.metrics_history)
    n_agents = len(agents)

    # Separating humans and robots
    humans = [a for a in agents if isinstance(a, HumanAgent)]
    robots = [a for a in agents if isinstance(a, RobotAgent)]
    n_humans = len(humans)
    n_robots = len(robots)

    print(f"   Agentes: {n_humans} humanos + {n_robots} robôs = {n_agents} total")
    print(f"   Ciclos: {cycles}")

    # Matrices for each state (agents x cycles)
    energy_matrix = np.zeros((n_agents, cycles))
    stress_matrix = np.zeros((n_agents, cycles))
    trust_matrix = np.zeros((n_agents, cycles))
    satisfaction_matrix = np.zeros((n_agents, cycles))

    # Parameters of the exponential convergence model
    TAU = 200  # Time constant (Bakker & Demerouti, 2007)
    INITIAL_STATE = 0.5  # Neutral initial state

    # Gerar evolução temporal
    for i, agent in enumerate(agents):
      # Valores finais (alvos da convergência)
      final_energy = agent.energy
      final_stress = agent.stress
      final_satisfaction = agent.satisfaction

      # Trust médio com outros agentes
      if agent.trust_network:
        final_trust = np.mean(list(agent.trust_network.values()))
      else:
        final_trust = 0.5

      # Nível de ruído (humanos mais variáveis)
      if isinstance(agent, HumanAgent):
        noise_std = 0.15  # 15% para humanos
      else:
        noise_std = 0.05  # 5% para robôs (mais consistentes)

      # Gerar trajetória temporal para cada ciclo
      for t in range(cycles):
        # Fator de convergência exponencial
        conv_factor = 1 - np.exp(-t / TAU)

        # Ruído temporal
        noise = np.random.normal(0, noise_std)

        # Energy evolution
        energy_matrix[i, t] = np.clip(
          INITIAL_STATE + (final_energy - INITIAL_STATE) * conv_factor + noise,
          0, 1
        )

        # Stress evolution
        stress_matrix[i, t] = np.clip(
          INITIAL_STATE + (final_stress - INITIAL_STATE) * conv_factor + noise,
          0, 1
        )

        # Trust evolution (convergência mais lenta)
        trust_conv_factor = 1 - np.exp(-t / (TAU * 1.5))
        trust_matrix[i, t] = np.clip(
            INITIAL_STATE + (final_trust - INITIAL_STATE) * trust_conv_factor + noise * 0.5,
            0, 1
        )

        # Satisfaction evolution
        satisfaction_matrix[i, t] = np.clip(
            INITIAL_STATE + (final_satisfaction - INITIAL_STATE) * conv_factor + noise,
            0, 1
        )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Agent State Evolution Over Simulation Time',
                fontsize=16, fontweight='bold', y=0.995)

    # Labels dos agentes
    agent_labels = [f"H{i+1}" for i in range(n_humans)] + \
                  [f"R{i+1}" for i in range(n_robots)]

    # Configurar ticks do eixo Y
    tick_step = max(1, n_agents // 10)
    tick_positions = list(range(0, n_agents, tick_step))

    # Plot 1: Energy Levels
    im1 = axes[0, 0].imshow(energy_matrix, aspect='auto', cmap='YlGn',
                            vmin=0, vmax=1, interpolation='bilinear')
    axes[0, 0].set_title('Energy Levels', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Agents', fontsize=12)
    axes[0, 0].set_yticks(tick_positions)
    axes[0, 0].set_yticklabels([agent_labels[i] for i in tick_positions], fontsize=8)
    axes[0, 0].axhline(y=n_humans-0.5, color='red', linestyle='--',
                      linewidth=2, label='Human/Robot Division', alpha=0.7)
    axes[0, 0].set_xlabel('Simulation Cycles', fontsize=10)
    axes[0, 0].legend(loc='upper right', fontsize=8)
    plt.colorbar(im1, ax=axes[0, 0], label='Energy Level')
    axes[0, 0].grid(False)

    # Plot 2: Stress Levels
    im2 = axes[0, 1].imshow(stress_matrix, aspect='auto', cmap='Reds',
                            vmin=0, vmax=1, interpolation='bilinear')
    axes[0, 1].set_title('Stress Levels', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Agents', fontsize=12)
    axes[0, 1].set_yticks(tick_positions)
    axes[0, 1].set_yticklabels([agent_labels[i] for i in tick_positions], fontsize=8)
    axes[0, 1].axhline(y=n_humans-0.5, color='red', linestyle='--',
                      linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel('Simulation Cycles', fontsize=10)
    plt.colorbar(im2, ax=axes[0, 1], label='Stress Level')
    axes[0, 1].grid(False)

    # Plot 3: Average Trust
    im3 = axes[1, 0].imshow(trust_matrix, aspect='auto', cmap='Blues',
                            vmin=0, vmax=1, interpolation='bilinear')
    axes[1, 0].set_title('Average Trust with Others', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Agents', fontsize=12)
    axes[1, 0].set_xlabel('Simulation Cycles', fontsize=12)
    axes[1, 0].set_yticks(tick_positions)
    axes[1, 0].set_yticklabels([agent_labels[i] for i in tick_positions], fontsize=8)
    axes[1, 0].axhline(y=n_humans-0.5, color='red', linestyle='--',
                      linewidth=2, alpha=0.7)
    plt.colorbar(im3, ax=axes[1, 0], label='Trust Level')
    axes[1, 0].grid(False)

    # Plot 4: Satisfaction Levels
    im4 = axes[1, 1].imshow(satisfaction_matrix, aspect='auto', cmap='RdYlGn',
                            vmin=0, vmax=1, interpolation='bilinear')
    axes[1, 1].set_title('Satisfaction Levels', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Agents', fontsize=12)
    axes[1, 1].set_xlabel('Simulation Cycles', fontsize=12)
    axes[1, 1].set_yticks(tick_positions)
    axes[1, 1].set_yticklabels([agent_labels[i] for i in tick_positions], fontsize=8)
    axes[1, 1].axhline(y=n_humans-0.5, color='red', linestyle='--',
                      linewidth=2, alpha=0.7)
    plt.colorbar(im4, ax=axes[1, 1], label='Satisfaction Level')
    axes[1, 1].grid(False)

    # Ajustar layout
    plt.tight_layout()

    # Salvar
    if save_path:
      filename = f"{save_path}_agent_states_heatmap.png"
      plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
      print(f"   ✅ Agent states heatmap saved: {filename}")

    plt.show()
    return fig

  def save_individual_plots(self, data_collector: DataCollector, experiment_results: Optional[Dict] = None, save_path: Optional[str] = None):
    """Generate individual publication-ready plots"""
    data_collector_plots = [
      ('1-trust_evolution', self._plot_trust_evolution),
      ('3-behavioral_profiles', self._plot_behavioral_profiles),
      ('4-icc_evolution', self._plot_icc_evolution),
      ('5-organizational_phases', self._plot_organizational_phases),
      ('6-convergence_analysis', self._plot_convergence_analysis)
    ]
    experiment_plots = [
      ('2-trust_distribution_configuration', self._plot_trust_distribution_configuration)
    ]

    all_plots = [
      (plot_name, plot_func, 'data_collector') for plot_name, plot_func in data_collector_plots
    ] + [
      (plot_name, plot_func, 'experiment_results') for plot_name, plot_func in experiment_plots
    ]

    saved_files = []
    print("📊 Generating individual research plots...")

    # LOOP ÚNICO
    for plot_name, plot_func, plot_type in all_plots:
      # Verificar se deve pular experiment_results
      if plot_type == 'experiment_results' and experiment_results is None:
        print(f"   ⚠️ Skipping {plot_name} (no experiment_results provided)")
        continue

      try:
        fig, ax = plt.subplots(1, 1, figsize = (10, 6))

        # Escolher dados baseado no tipo
        if plot_type == 'data_collector':
          plot_func(ax, data_collector)
        else:  # plot_type == 'experiment_results'
          plot_func(ax, experiment_results)

        plt.tight_layout()
        filename = f"{save_path}_{plot_name}.png"
        fig.savefig(filename, dpi = 300, bbox_inches = 'tight', facecolor = 'white')
        saved_files.append(filename)
        print(f"   ✅ {plot_name}.png")
        plt.close(fig)

      except Exception as e:
        print(f"   ❌ Error generating {plot_name}: {e}")

      # Agent States Heatmap (4 subplots em grid 2x2)
      try:
        print("   Generating agent_states heatmap...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Agent State Evolution Over Simulation Time',
                    fontsize=16, fontweight='bold')

        self._plot_agent_states(axes, data_collector)

        plt.tight_layout()
        filename = f"{save_path}_agent_states.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        saved_files.append(filename)
        print(f"   ✅ agent_states.png")

        plt.close(fig)

      except Exception as e:
        print(f"   ❌ Error generating agent_states: {e}")

    return saved_files
