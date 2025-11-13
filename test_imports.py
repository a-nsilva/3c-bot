"""Teste rápido de imports"""
print("🧪 Testing imports...")

try:
    import src
    print(f"✅ Package version: {src.__version__}")
except Exception as e:
    print(f"❌ Package import failed: {e}")
    exit(1)

try:
    from src.core import Agent, HumanAgent, RobotAgent
    print("✅ Core imports OK")
except Exception as e:
    print(f"❌ Core import failed: {e}")
    exit(1)

try:
    from src.experiments import ResearchExperiment
    print("✅ Experiments import OK")
except Exception as e:
    print(f"❌ Experiments import failed: {e}")
    exit(1)

try:
    from src.visualization import ResearchVisualizer
    print("✅ Visualization import OK")
except Exception as e:
    print(f"❌ Visualization import failed: {e}")
    exit(1)

try:
    from src.analysis import AdvancedAnalysis
    print("✅ Analysis import OK")
except Exception as e:
    print(f"❌ Analysis import failed: {e}")
    exit(1)

try:
    from src.main import main
    print("✅ Main import OK")
except Exception as e:
    print(f"❌ Main import failed: {e}")
    exit(1)

print("\n🎉 All imports successful!")