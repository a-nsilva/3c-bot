"""
ADVANCED ANALYSIS MODULE
Sensitivity analysis and scalability validation

This module contains computationally intensive analyses that test
the robustness and generalizability of simulation results.
"""

from typing import Dict

import numpy as np
from scipy import stats

from .core import (
  CoreEngine, 
  ConfigurationType, 
  ExperimentScale,
  TheoreticalParameters
)

class AdvancedAnalysis:
  """ Performs advanced statistical analyses for research validation"""
  
  def __init__(self, results_dir: Path):
    self.results_dir = results_dir
    self.results_dir.mkdir(parents = True, exist_ok = True)
    
  def _run_single_simulation(self, config_type: ConfigurationType,
                            scale: ExperimentScale,
                            cycles: int = 1000,
                            seed: int = 42) -> Dict:
    """
    Helper method to run a single simulation
    
    Args:
      config_type: Population configuration
      scale: Experimental scale
      cycles: Number of simulation cycles
      seed: Random seed for reproducibility
        
    Returns:
      Dict with simulation results
    """
    # Set reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    # Calculate population
    population, _ = scale.value
    humans_ratio, robots_ratio = config_type.value
    num_humans = round(population * humans_ratio)
    num_robots = population - num_humans
    
    # Run simulation
    simulator = CoreEngine(num_humans, num_robots, config_type)
    result = simulator.run_simulation(cycles)
    
    # Extract summary
    data_collector = result['data_collector']
    summary = data_collector.get_summary()
    
    return {
      'final_trust': summary['trust_statistics']['final'],
      'achieved_symbiosis': summary['achieved_symbiosis'],
      'convergence_cycle': summary.get('convergence_cycle'),
      'execution_time': result['execution_time']
    }
  
  def run_sensitivity_analysis(self) -> Dict:
    """
    Sensitivity analysis of main model parameters
    
    Tests robustness of results with parameter variations:
    - Trust learning rate (α): [0.05, 0.10, 0.15]
    - Symbiosis threshold: [0.63, 0.70, 0.77] (±10%)
    
    Returns:
        Dict with sensitivity analysis results
    """    
    print("\n" + "="*70)
    print("🔬 SENSITIVITY ANALYSIS - Parameter Robustness Testing")
    print("="*70)

    original_learning_rate = TheoreticalParameters.TRUST_LEARNING_RATE
    original_threshold = TheoreticalParameters.TRUST_THRESHOLD_SYMBIOSIS

    sensitivity_results = {}

    # 1. TEST TRUST LEARNING RATE (α)
    print("\n📊 Testing Trust Learning Rate (α) sensitivity...")
    alpha_values = [0.05, 0.10, 0.15]
    alpha_results = []

    for alpha in alpha_values:
      print(f"   Testing α = {alpha}...", end = '')

      TheoreticalParameters.TRUST_LEARNING_RATE = alpha

      # Executar simulação pequena
      result = self.run_single_experiment(
        ConfigurationType.MAJORITY_ROBOT,
        ExperimentScale.SMALL,
        cycles = 1000,
        seed = 42
      )

      alpha_results.append({
        'alpha': alpha,
        'final_trust': result['final_trust'],
        'achieved_symbiosis': result['achieved_symbiosis']
      })

      print(f" Trust: {result['final_trust']:.3f}")

    TheoreticalParameters.TRUST_LEARNING_RATE = original_learning_rate

    # Calculate Coefficient of Variation
    trust_values = [r['final_trust'] for r in alpha_results]
    mean_trust = np.mean(trust_values)
    std_trust = np.std(trust_values, ddof = 1) if len(trust_values) > 1 else 0
    cv_alpha = std_trust / mean_trust if mean_trust > 0 else 0

    sensitivity_results['alpha'] = {
      'results': alpha_results,
      'cv': cv_alpha,
      'mean': mean_trust,
      'std': std_trust,
      'robust': cv_alpha < 0.12 
    }

    print(f"   ✅ CV = {cv_alpha:.4f} {'(ROBUST)' if cv_alpha < 0.12 else '(REVIEW)'}")
    
    # 2. TEST SYMBIOSIS THRESHOLD
    print("\n📊 Testing Symbiosis Threshold sensitivity...")
    threshold_values = [0.63, 0.70, 0.77]  # (±10%)
    threshold_results = []

    for threshold in threshold_values:
      print(f"   Testing threshold = {threshold}...", end = '')

      # Modify parameter
      TheoreticalParameters.TRUST_THRESHOLD_SYMBIOSIS = threshold

      # Run simulation
      result = self.run_single_experiment(
        ConfigurationType.MAJORITY_ROBOT,
        ExperimentScale.SMALL,
        cycles = 1000,
        seed = 42
      )

      # Check if achieves new threshold
      achieved = 1 if result['final_trust'] >= threshold else 0

      threshold_results.append({
        'threshold': threshold,
        'final_trust': result['final_trust'],
        'symbiosis_achieved': achieved
      })

      print(f" Trust: {result['final_trust']:.3f}, Symbiosis: {'✅' if achieved else '❌'}")

    # Restore original
    TheoreticalParameters.TRUST_THRESHOLD_SYMBIOSIS = original_threshold

    # Calculate variability
    symbiosis_rates = [r['symbiosis_achieved'] for r in threshold_results]
    mean_rate = np.mean(symbiosis_rates)
    std_rate = np.std(symbiosis_rates, ddof = 1) if len(symbiosis_rates) > 1 else 0

    sensitivity_results['threshold'] = {
      'results': threshold_results,
      'mean_rate': mean_rate,
      'std_rate': std_rate,
      'robust': std_rate < 0.35,
      'interpretation': 'Threshold effects expected for binary outcomes'
    }

    print(f"   ✅ Success rate: {mean_rate:.2f} ± {std_rate:.2f} {'(STABLE)' if std_rate < 0.35 else '(VARIABLE)'}")

    # 3. SUMMARY
    print("\n" + "="*70)
    print("📈 SENSITIVITY ANALYSIS SUMMARY")
    print("="*70)

    print(f"\n1. Trust Learning Rate (α):")
    print(f"   - Coefficient of Variation: {cv_alpha:.4f}")
    print(f"   - Range: [{min(trust_values):.3f}, {max(trust_values):.3f}]")
    print(f"   - Mean ± SD: {mean_trust:.3f} ± {std_trust:.3f}")
    print(f"   - Robustness: {'✅ PASS (CV < 0.12)' if cv_alpha < 0.12 else '⚠️ REVIEW (CV >= 0.12)'}")

    print(f"\n2. Symbiosis Threshold:")
    print(f"   - Success Rate: {mean_rate:.2f} ± {std_rate:.2f}")
    print(f"   - Robustness: {'✅ PASS (SD < 0.35)' if std_rate < 0.35 else '⚠️ REVIEW (SD >= 0.35)'}")
    print(f"   - Note: Binary outcomes (pass/fail) naturally show higher variance")

    print(f"\n3. Overall Assessment:")
    all_robust = sensitivity_results['alpha']['robust'] and sensitivity_results['threshold']['robust']
    print(f"   {'✅ MODEL IS ROBUST' if all_robust else '⚠️ REVIEW REQUIRED'}")
    print(f"   Results demonstrate {'stable patterns' if all_robust else 'sensitivity'} in human-robot creative cooperation dynamics.")

    # Metadata
    sensitivity_results['metadata'] = {
      'original_learning_rate': original_learning_rate,
      'original_threshold': original_threshold,
      'test_configuration': 'MAJORITY_ROBOT',
      'test_scale': 'SMALL (30 agents)',
      'test_cycles': 1000,
      'test_seed': 42
    }

    return sensitivity_results
  
  def validate_population_scalability(self) -> Dict:
    """
    Validate result consistency across population scales
        
    Tests [30, 60, 90] agents with multiple replications to verify
    that findings generalize beyond specific population sizes.
    
    Returns:
      Dict with scalability validation results
    """
    
    print("\n" + "="*70)
    print("📊 POPULATION SCALABILITY VALIDATION")
    print("="*70)

    scales_to_test = [
      ExperimentScale.SMALL,   # 30 agents
      ExperimentScale.MEDIUM,  # 60 agents
      ExperimentScale.LARGE    # 90 agents
    ]

    results_by_scale = {}
    n_replications = 3 
    cycles = 1000  

    # PHASE 1: Test each population scale 
    for scale in scales_to_test:
      pop_size = scale.value[0]
      print(f"\n🔬 Testing population size: {pop_size} agents ({n_replications} replications, {cycles} cycles)")

      trust_values = []
      symbiosis_count = 0
      convergence_cycles = []
      execution_times = []

      for rep in range(n_replications):
        seed = 42 + rep
        print(f"   Rep {rep+1}/{n_replications} (seed={seed})...", end = '', flush = True)

        # Execute simulation
        result = self.run_single_experiment(
          ConfigurationType.MAJORITY_ROBOT,
          scale,
          cycles = cycles,
          seed = seed
        )

        # Collect metrics
        trust_values.append(result['final_trust'])
        execution_times.append(result['execution_time'])

        if result['achieved_symbiosis']:
          symbiosis_count += 1

        if result.get('convergence_cycle'):
          convergence_cycles.append(result['convergence_cycle'])

        print(f" Trust = {result['final_trust']:.3f}, "
              f"Time = {result['execution_time']:.1f}s")

      # Calculate statistics using DataCollector method
      stats_dict = DataCollector.calculate_descriptive_stats(trust_values)

      symbiosis_rate = symbiosis_count / n_replications
      avg_convergence = (np.mean(convergence_cycles) 
                        if convergence_cycles else None)
      avg_exec_time = np.mean(execution_times)
      
      results_by_scale[pop_size] = {
        'trust_values': trust_values,
        'mean_trust': stats_dict['mean'],
        'std_trust': stats_dict['std'],
        'ci_95_lower': stats_dict['ci_95_lower'],
        'ci_95_upper': stats_dict['ci_95_upper'],
        'symbiosis_rate': symbiosis_rate,
        'symbiosis_count': symbiosis_count,
        'avg_convergence': avg_convergence,
        'convergence_cycles': convergence_cycles,
        'avg_execution_time': avg_exec_time,
        'n_replications': n_replications
      }
      
      print(f"   📊 Summary: Trust = {stats_dict['mean']:.3f} ± "
            f"{stats_dict['std']:.3f}")
      print(f"      95% CI: [{stats_dict['ci_95_lower']:.3f}, "
            f"{stats_dict['ci_95_upper']:.3f}]")
      print(f"      Symbiosis: {symbiosis_rate:.0%} "
            f"({symbiosis_count}/{n_replications})")
      if avg_convergence:
        print(f"      Avg convergence: cycle {avg_convergence:.0f}")

    # PHASE 2: Statistical Analysis
    print(f"\n📈 SCALABILITY ANALYSIS:")

    pop_sizes = list(results_by_scale.keys())
    mean_trusts = [results_by_scale[p]['mean_trust'] for p in pop_sizes]
    
    # Overall statistics
    overall_mean = np.mean(mean_trusts)
    overall_std = np.std(mean_trusts, ddof=1) if len(mean_trusts) > 1 else 0
    
    # Coefficient of Variation
    cv = overall_std / overall_mean if overall_mean > 0 else 0

    # Correlation
    if len(mean_trusts) >= 3:
      # Pearson correlation
      correlation_pearson, p_value_pearson = stats.pearsonr(pop_sizes, mean_trusts)
      correlation_spearman, p_value_spearman = stats.spearmanr(pop_sizes, mean_trusts)
    else:
      correlation_pearson = correlation_spearman = None
      p_value_pearson = p_value_spearman = None

    print(f"   Trust range: [{min(mean_trusts):.3f}, {max(mean_trusts):.3f}]")
    print(f"   Mean trust: {overall_mean:.3f} ± {overall_std:.3f}")
    print(f"   CV: {cv:.3f} {'(STABLE)' if cv < 0.10 else '(VARIABLE)'}")
    
    if correlation_spearman is not None:
      print(f"   Spearman correlation: ρ = {correlation_spearman:.3f}, "
            f"p = {p_value_spearman:.3f}")
    
    stable = cv < 0.10
    print(f"   ✅ Results {'are stable' if stable else 'show variability'} "
          f"across population scales")

    # PHASE 3: Configuration Ranking Consistency
    print(f"\n🔍 Validating configuration ranking consistency...")
    print(f"   Testing 3 configurations at MEDIUM scale (60 agents, {n_replications} reps each)")

    configs_to_compare = [
      ConfigurationType.MAJORITY_HUMAN,
      ConfigurationType.BALANCED,
      ConfigurationType.MAJORITY_ROBOT
    ]

    ranking_results = {}
        
    for config in configs_to_compare:
      print(f"\n   Configuration: {config.name}")
      
      config_trust_values = []
      
      for rep in range(n_replications):
        seed = 42 + rep
        print(f"      Rep {rep+1}/{n_replications} (seed={seed})...", 
              end='', flush=True)
          
        result = self._run_single_simulation(
          config,
          ExperimentScale.MEDIUM,
          cycles = cycles,
          seed = seed
        )
        
        config_trust_values.append(result['final_trust'])
        print(f" Trust = {result['final_trust']:.3f}")
      
      # Calculate statistics
      config_stats = DataCollector.calculate_descriptive_stats(
          config_trust_values
      )
      
      ranking_results[config.name] = {
        'trust_values': config_trust_values,
        'mean_trust': config_stats['mean'],
        'std_trust': config_stats['std']
      }
      
      print(f"      Mean: {config_stats['mean']:.3f} ± "
            f"{config_stats['std']:.3f}")
    
    # Check if MAJORITY_ROBOT is still superior
    robot_majority_trust = ranking_results['MAJORITY_ROBOT']['mean_trust']
    is_superior = all(
      robot_majority_trust > data['mean_trust']
      for name, data in ranking_results.items()
      if name != 'MAJORITY_ROBOT'
    )
    
    print(f"\n   Robot-majority superiority maintained: "
          f"{'✅ YES' if is_superior else '❌ NO'}")
    
    # Print ranking
    sorted_configs = sorted(
      ranking_results.items(),
      key = lambda x: x[1]['mean_trust'],
      reverse = True
    )
    
    print(f"\n   📊 Configuration Ranking (60 agents):")
    for rank, (config_name, data) in enumerate(sorted_configs, 1):
      print(f"      #{rank}: {config_name}: "
            f"{data['mean_trust']:.3f} ± {data['std_trust']:.3f}")

    # PHASE 4: Compile results
    scalability_summary = {
      'results_by_scale': results_by_scale,
      'overall_mean': overall_mean,
      'overall_std': overall_std,
      'cv': cv,
      'correlation_pearson': correlation_pearson,
      'correlation_spearman': correlation_spearman,
      'p_value_pearson': p_value_pearson,
      'p_value_spearman': p_value_spearman,
      'stable': stable,
      'ranking_maintained': is_superior,
      'ranking_results': ranking_results,
      'methodology': {
        'scales_tested': pop_sizes,
        'replications_per_scale': n_replications,
        'cycles_per_simulation': cycles,
        'seeds_used': list(range(42, 42 + n_replications)),
        'total_simulations': len(scales_to_test) * n_replications + len(configs_to_compare) * n_replications
      }
    }

    # PHASE 5: Final Summary
    print("\n" + "="*70)
    print("📊 SCALABILITY VALIDATION SUMMARY")
    print("="*70)
    
    print(f"\n✅ Tested {len(scales_to_test)} population scales with "
          f"{n_replications} replications each")
    print(f"   Total simulations: "
          f"{scalability_summary['methodology']['total_simulations']}")
    
    print(f"\n📈 Key Findings:")
    print(f"   • Mean trust across scales: {overall_mean:.3f} ± {overall_std:.3f}")
    print(f"   • Coefficient of variation: {cv:.3f}")
    print(f"   • Stability: {'✅ STABLE' if stable else '⚠️ VARIABLE'}")
    
    if correlation_spearman is not None:
      print(f"   • Spearman correlation: ρ = {correlation_spearman:.3f}")
    
    print(f"\n🏆 Configuration Ranking:")
    print(f"   • Robot-majority superiority: "
          f"{'✅ MAINTAINED' if is_superior else '❌ NOT MAINTAINED'}")
    
    print(f"\n💡 Interpretation:")
    if stable and is_superior:
      print(f"   ✅ Results demonstrate robust scalability")
      print(f"   ✅ Robot-majority configuration remains optimal")
    elif stable and not is_superior:
      print(f"   ⚠️ Results stable but ranking changed")
    elif not stable and is_superior:
      print(f"   ⚠️ Robot-majority superior but trust varies")
    else:
      print(f"   ⚠️ Results show instability and ranking changes")
    
    return scalability_summary
