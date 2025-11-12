"""
MAIN INTERFACE
Command-line interface for research experiments
"""

from .core import ConfigurationType, ExperimentScale
from .experiments import ResearchExperiment

def print_header():
  print("\n" + "="*70)
  print("🧬 HUMAN-3C-BOT ORGANIZATIONAL DYNAMICS SIMULATOR v1.0")
  print("="*70)
  print("Theoretical: SVO, Asimov Laws, Guilford, Trust Theory")
  print("="*70)

def main_menu():
  print("\n🎯 RESEARCH OPTIONS:")
  print("1. 🚀 Quick Demo (validation)")
  print("2. 🔬 Complete Experiment (5 configs × N reps)")
  print("3. 🎯 Custom Experiment")
  print("4. 🔍 Advanced Analysis")
  print("5. ❌ Exit")

  return input("\nSelect option (1-5): ").strip()

def advanced_analysis_menu():
  print("\n🔍 ADVANCED ANALYSIS:")
  print("4.1 📊 Sensitivity Analysis")
  print("4.2 📈 Scalability Validation")
  print("4.3 🎨 Generate Figure 4 (Agent States)")
  print("4.4 ← Back to Main Menu")
    
  return input("\nSelect option (4.1-4.4): ").strip()

def main():
  print_header()
  runner = ResearchExperiment()
    
  while True:
    try:
      choice = main_menu()
        
        if choice == "1":
          # Quick Demo
          result = runner.run_demo_experiment()
            
        elif choice == "2":
          # Complete Experiment
          print("\n📊 Select scale:")
          scales = list(ExperimentScale)
          for i, s in enumerate(scales, 1):
            pop, reps = s.value
            print(f"{i}. {s.name} ({pop} agents, {reps} reps)")
            
          scale_idx = int(input("Choice (1-3): ")) - 1
          if 0 <= scale_idx < len(scales):
            scale = scales[scale_idx]
            results = runner.run_complete_experiment(scale=scale)
            runner.generate_research_visualizations(experiment_results=results)
            runner.save_research_results(results)
                
        elif choice == "3":
          # Custom Experiment
          print("\n🎯 CUSTOM CONFIGURATION:")
            
          # Scale
          print("\nScale:")
          scales = list(ExperimentScale)
          for i, s in enumerate(scales, 1):
            print(f"{i}. {s.name} ({s.value[0]} agents)")
          scale = scales[int(input("Choice (1-3): ")) - 1]
            
          # Configuration
          print("\nPopulation:")
          configs = list(ConfigurationType)
          for i, c in enumerate(configs, 1):
            h, r = c.value
            print(f"{i}. {int(h*100)}%H/{int(r*100)}%R")
          config = configs[int(input("Choice (1-5): ")) - 1]
            
          # Cycles
          cycles = int(input("\nCycles (default 1000): ") or "1000")
            
          print(f"\n🔬 Running: {config.name}, {scale.name}, {cycles} cycles")
          result = runner.run_single_experiment(config, scale, cycles)
          runner.generate_research_visualizations(single_result=result)
            
        elif choice == "4":
          # Advanced Analysis submenu
          sub_choice = advanced_analysis_menu()
            
          if sub_choice == "4.1":
            print("\n📊 Running Sensitivity Analysis...")
            results = runner.advanced_analysis.run_sensitivity_analysis()
            print("✅ Complete! Saved to sensitivity_analysis_results.json")
                
          elif sub_choice == "4.2":
            print("\n📈 Running Scalability Validation...")
            results = runner.advanced_analysis.validate_population_scalability()
            print("✅ Complete! Saved to scalability_validation_results.json")
                
          elif sub_choice == "4.3":
            print("\n🎨 Generating Figure 4...")
            result = runner.run_single_experiment(
              ConfigurationType.MAJORITY_ROBOT,
              ExperimentScale.MEDIUM,
              cycles = 1000,
              seed = 42
            )
            runner.visualizer.create_agent_states_heatmap(
              result['data_collector'],
              result['agents'],
              "figure4"
            )
            print("✅ Saved: figure4_agent_states_heatmap.png")
                
          elif sub_choice == "4.4":
            continue  # Volta ao menu principal
                
        elif choice == "5":
          print("\n👋 Thank you!")
          print("Citation: Silva, A.N. et al. (2025)")
          break
            
        else:
          print("❌ Invalid option")
            
    except KeyboardInterrupt:
      print("\n\n⏹️ Interrupted by user")
      break
    except Exception as e:
      print(f"\n❌ Error: {e}")
      import traceback
      traceback.print_exc()

if __name__ == "__main__":
  main()
