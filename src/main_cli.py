#!/usr/bin/env python3
"""
3C-BOT Research Simulator - CLI Interface
Command line interface for the human-robot organizational dynamics simulator
"""

import sys
import os

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import ConfigurationType, ExperimentScale, ResearchExperiment


def display_menu():
    """Display the main menu"""
    print("\n" + "="*70)
    print("🧬 3C-BOT RESEARCH SIMULATOR v1.0")
    print("Theoretical Foundation: SVO, Asimov Laws, Guilford Model, Trust Theory")
    print("="*70)
    print("\n1. 🚀 Quick demo")
    print("2. 🔬 Complete experiment (all configurations)")
    print("3. 🎯 Custom experiment")
    print("4. ❌ Exit")
    print("\n" + "="*70)

def run_complete_experiment():
    """Run demonstration experiment with defined parameters"""
    print("\n🎯 DEMO EXPERIMENT")
    
    experiment = ResearchExperiment()
    results = experiment.run_custom_experiment()
    
    if results:
        print(f"\n✅ DEMO experiment completed!")
        print(f"Final trust: {results['final_trust']:.3f}")
        print(f"Symbiosis achieved: {'✅ Yes' if results['achieved_symbiosis'] else '❌ No'}")
    
    return results
    
def run_complete_experiment():
    """Run complete research experiment with all configurations"""
    print("This will test all 5 population configurations with statistical analysis.")
    print("\nSelect experimental scale:")
    scales = list(ExperimentScale)
    for i, scale in enumerate(scales, 1):
        population, replications = scale.value
        expected_time = population * replications * 0.015  # Estimated time
        print(f"{i}. {scale.name} ({population} agents, {replications} replications) [~{expected_time:.1f}s]")
    try:
        choice = int(input("\nChoose scale (1-3): ")) - 1
        if 0 <= choice < len(scales):
            scale = scales[choice]
            
            cycles = input("Cycles per simulation (default 1000): ").strip()
            cycles = int(cycles) if cycles else 1000
            
            print(f"\nStarting complete experiment...")
            print(f"Scale: {scale.name}")
            print(f"Cycles: {cycles}")
            print("This may take several minutes...")
            
            experiment = ResearchExperiment()
            results = experiment.run_complete_experiment(scale = scale, cycles = cycles)
            
            print(f"\n✅ Experiment completed!")
            print(f"Best configuration: {results['best_config']}")
            print(f"Results saved to: results/report/")
            
            return results
        else:
            print("❌ Invalid scale selection")
            return None
    except (ValueError, KeyboardInterrupt):
        print("\n❌ Operation cancelled")
        return None

def run_custom_experiment():
    """Run custom experiment with user-defined parameters"""
    print("\n🎯 CUSTOM EXPERIMENT")
    print("Configure your own experiment parameters:")
    
    experiment = ResearchExperiment()
    results = experiment.run_custom_experiment()
    
    if results:
        print(f"\n✅ Custom experiment completed!")
        print(f"Final trust: {results['final_trust']:.3f}")
        print(f"Symbiosis achieved: {'✅ Yes' if results['achieved_symbiosis'] else '❌ No'}")
    
    return results


def main():
    """Main CLI interface"""
    try:
        while True:
            display_menu()
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == "2":
                run_demo_experiment()
                
            if choice == "2":
                run_complete_experiment()
                
            elif choice == "3":
                run_custom_experiment()
                
            elif choice == "4":
                print("\nThank you for using the 3C-BOT Research Simulator!")
                print("\nFor academic citation:")
                print("Silva, A. N. et al. Computational modeling of behavioral dynamics")
                print("in human-robot organizational communities.")
                break
                
            else:
                print("❌ Invalid option. Please choose 1, 2, or 3.")
                
            input("\nPress Enter to continue...")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Operation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
