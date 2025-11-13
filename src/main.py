"""
COMPUTATIONAL MODELING OF BEHAVIORAL DYNAMICS IN HUMAN-ROBOT ORGANIZATIONAL COMMUNITIES
MAIN INTERFACE
==============
Command-line interface for research experiments

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
"""

# IMPORTS
from .core import ConfigurationType, ExperimentScale
from .experiments import ResearchExperiment

# INTERFACE FUNCTIONS
def print_header():
  """Print application header"""
  print("\n" + "="*70)
  print("  HUMAN-3C-BOT ORGANIZATIONAL DYNAMICS SIMULATOR v1.0")
  print("="*70)
  print("Theoretical: SVO, Asimov Laws, Guilford, Trust Theory")
  print("="*70)


def main_menu():
  print("\n  RESEARCH OPTIONS:")
  print("1.   Quick Demo (validation)")
  print("2.   Complete Experiment (5 configs × N reps)")
  print("3.   Custom Experiment")
  print("4.   Advanced Analysis")
  print("5.   Exit")
  
  return input("\nSelect option (1-5): ").strip()


def advanced_analysis_menu():   
  print("\n  ADVANCED ANALYSIS:")
  print("4.1  Sensitivity Analysis")
  print("4.2  Scalability Validation")
  print("4.3  Generate Figure 4 (Agent States)")
  print("4.4  Back to Main Menu")
  
  return input("\nSelect option (4.1-4.4): ").strip()

# MENU HANDLERS
def handle_demo(runner: ResearchExperiment):
  print("\n" + "="*70)
  print("  QUICK DEMO")
  print("="*70)
  result = runner.run_demo_experiment()
  
  print("\n  Demo completed successfully!")
  print(f"   Results saved to: results/reports/")
  print(f"   Plots saved to: results/plots/")

def handle_complete_experiment(runner: ResearchExperiment):
  print("\n" + "="*70)
  print("  COMPLETE EXPERIMENT")
  print("="*70)
  
  # Scale selection
  print("\n  Select experimental scale:")
  scales = list(ExperimentScale)
  for i, s in enumerate(scales, 1):
    pop, reps = s.value
    est_time = pop * reps * 0.8  # Rough estimate
    print(f"{i}. {s.name:8} ({pop:2} agents, {reps:2} reps) "
          f"[~{est_time:.0f}s]")
  
  try:
    scale_idx = int(input("\nChoice (1-3): ")) - 1
    if 0 <= scale_idx < len(scales):
      scale = scales[scale_idx]
      
      print(f"\n▶ Running {scale.name} experiment...")
      print(f"  This will take approximately "
            f"{scale.value[0] * scale.value[1] * 0.8:.0f} seconds")
      
      confirm = input("\nProceed? (y/n): ").strip().lower()
      if confirm == 'y':
        results = runner.run_complete_experiment(scale=scale)
        runner.generate_research_visualizations(
            experiment_results=results,
            output_prefix="complete"
        )
        runner.save_research_results(results)
        
        print("\n  Complete experiment finished!")
        print(f"   Results saved to: results/reports/")
        print(f"   Plots saved to: results/plots/")
      else:
        print("   Cancelled.")
    else:
      print("     Invalid choice")
  except (ValueError, IndexError):
    print("     Invalid input")


def handle_custom_experiment(runner: ResearchExperiment):
  print("\n" + "="*70)
  print("  CUSTOM EXPERIMENT")
  print("="*70)
  
  try:
    # Scale selection
    print("\n  Select scale:")
    scales = list(ExperimentScale)
    for i, s in enumerate(scales, 1):
      print(f"{i}. {s.name} ({s.value[0]} agents)")
    
    scale_idx = int(input("Choice (1-3): ")) - 1
    if not (0 <= scale_idx < len(scales)):
      print("     Invalid choice")
      return
    scale = scales[scale_idx]
    
    # Configuration selection
    print("\n  Select population configuration:")
    configs = list(ConfigurationType)
    for i, c in enumerate(configs, 1):
      h, r = c.value
      print(f"{i}. {int(h*100):2}%H / {int(r*100):2}%R - {c.name}")
    
    config_idx = int(input("Choice (1-5): ")) - 1
    if not (0 <= config_idx < len(configs)):
      print("     Invalid choice")
      return
    config = configs[config_idx]
    
    # Cycles selection
    cycles_input = input("\n   Cycles (default 1000, min 500): ").strip()
    cycles = int(cycles_input) if cycles_input else 1000
    cycles = max(500, cycles)
    
    # Summary
    print(f"\n  Configuration Summary:")
    print(f"   Scale: {scale.name} ({scale.value[0]} agents)")
    print(f"   Population: {config.name}")
    print(f"   Cycles: {cycles}")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm == 'y':
      print(f"\n▶ Running custom experiment...")
      result = runner.run_single_experiment(config, scale, cycles)
      runner.generate_research_visualizations(
          single_result = result,
          output_prefix = "custom"
      )
      
      print("\n  Custom experiment finished!")
      print(f"   Results saved to: results/plots/")
    else:
      print("   Cancelled.")
          
  except (ValueError, IndexError) as e:
      print(f"     Invalid input: {e}")

def handle_advanced_analysis(runner: ResearchExperiment):
  while True:
    sub_choice = advanced_analysis_menu()
    
    if sub_choice == "4.1":
      # Sensitivity Analysis
      print("\n" + "="*70)
      print("  SENSITIVITY ANALYSIS")
      print("="*70)
      print("This will test parameter robustness (α and threshold).")
      
      confirm = input("\nProceed? (y/n): ").strip().lower()
      if confirm == 'y':
          results = runner.advanced_analysis.run_sensitivity_analysis()
          runner.save_research_results(
              results,
              filename = "sensitivity_analysis_results.json"
          )
          print("\n  Sensitivity analysis complete!")
          print(f"   Results: results/reports/sensitivity_analysis_results.json")
      else:
          print("   Cancelled.")
    
    elif sub_choice == "4.2":
      # Scalability Validation
      print("\n" + "="*70)
      print("  SCALABILITY VALIDATION")
      print("="*70)
      print("This will test 3 population scales (30, 60, 90 agents).")        
      
      confirm = input("\nProceed? (y/n): ").strip().lower()
      if confirm == 'y':
        results = runner.advanced_analysis.validate_population_scalability()
        runner.save_research_results(
            results,
            filename = "scalability_validation_results.json"
        )
        print("\n  Scalability validation complete!")
        print(f"   Results: results/reports/scalability_validation_results.json")
      else:
        print("   Cancelled.")
    
    elif sub_choice == "4.3":
      # Generate agent states heatmap
      print("\n" + "="*70)
      print("  GENERATE FIGURE")
      print("="*70)
      print("This will generate agent states heatmap (2×2 grid).")
      
      confirm = input("\nProceed? (y/n): ").strip().lower()
      if confirm == 'y':
        print("\n▶ Running baseline simulation...")
        result = runner.run_single_experiment(
            ConfigurationType.MAJORITY_ROBOT,
            ExperimentScale.MEDIUM,
            cycles = 1000,
            seed = 42
        )
        
        print("▶ Creating Figure 4...")
        runner.visualizer.create_agent_states_figure(
            result['data_collector'],
            result['agents'],
            "results/plots/figure4"
        )
        
        print("\n✅ Figure 4 generated!")
        print(f"   Saved: results/plots/figure4_agent_states_2x2.png")
      else:
        print("   Cancelled.")
    
    elif sub_choice == "4.4":
      # Back to main menu
      break
    
    else:
        print("     Invalid option")

# MAIN FUNCTION
def main():
  """Main application entry point"""
  print_header()
  runner = ResearchExperiment()
  
  while True:
    try:
      choice = main_menu()
      
      if choice == "1":
        handle_demo(runner)
      
      elif choice == "2":
        handle_complete_experiment(runner)
      
      elif choice == "3":
        handle_custom_experiment(runner)
      
      elif choice == "4":
        handle_advanced_analysis(runner)
      
      elif choice == "5":
        print("\n" + "="*70)
        print("  Thank you for using the simulator!")
        print("="*70)
        print("\n  For academic citation:")
        print("Silva, A.N. et al. (2025). Behavioral dynamics of creative")
        print("cooperation in human-3C-bot communities. [Journal Name].")
        print("="*70)
        break
        
      else:
        print("     Invalid option. Please choose 1-5.")
      
    except KeyboardInterrupt:
      print("\n\n   Operation interrupted by user")
      print("Exiting gracefully...")
      break
    
    except Exception as e:
      print(f"\n  Unexpected error: {e}")
      print("Please report this issue.")
      import traceback
      traceback.print_exc()

if __name__ == "__main__":
    main()
