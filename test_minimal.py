"""Teste funcional mínimo - 100 ciclos"""
print("🧪 Testing minimal simulation (100 cycles)...\n")

from src import ResearchExperiment, ConfigurationType, ExperimentScale

runner = ResearchExperiment()

result = runner.run_single_experiment(
    ConfigurationType.BALANCED,
    ExperimentScale.SMALL,
    cycles=100,
    seed=42
)

print(f"\n✅ Simulation completed!")
print(f"   Final trust: {result['final_trust']:.3f}")
print(f"   Achieved symbiosis: {result['achieved_symbiosis']}")
print(f"   Execution time: {result['execution_time']:.2f}s")
print(f"\n✅ All systems operational!")