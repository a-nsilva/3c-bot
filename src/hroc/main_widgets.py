"""
3C-BOT Research Simulator - Jupyter Widgets Interface
Interactive interface for Jupyter notebooks with the same menu structure
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
import time
import threading

from .core import ResearchExperiment, ExperimentScale, ConfigurationType


class ResearchSimulatorWidgets:
    """Jupyter widgets interface for the research simulator"""
    
    def __init__(self):
        self.experiment = ResearchExperiment()
        self.results = None
        self.output = widgets.Output()
        
    def create_main_menu(self):
        """Create the main menu widgets"""
        # Title
        title = widgets.HTML(
            "<h2>🧬 3C-BOT RESEARCH SIMULATOR</h2>"
            "<p><i>Computational Modeling of Behavioral Dynamics in Human-Robot Organizational Communities</i></p>"
            "<hr>"
        )
        
        # Menu buttons
        self.complete_btn = widgets.Button(
            description="🔬 Complete Research Experiment",
            button_style='primary',
            layout=widgets.Layout(width='300px', height='40px')
        )
        
        self.custom_btn = widgets.Button(
            description="🎯 Custom Experiment", 
            button_style='info',
            layout=widgets.Layout(width='300px', height='40px')
        )
        
        self.exit_btn = widgets.Button(
            description="❌ Exit",
            button_style='danger',
            layout=widgets.Layout(width='300px', height='40px')
        )
        
        # Connect buttons to functions
        self.complete_btn.on_click(self._on_complete_experiment)
        self.custom_btn.on_click(self._on_custom_experiment)
        self.exit_btn.on_click(self._on_exit)
        
        # Layout
        menu_box = widgets.VBox([
            title,
            widgets.HTML("<h3>Select Experiment Type:</h3>"),
            self.complete_btn,
            self.custom_btn, 
            self.exit_btn,
            self.output
        ])
        
        return menu_box
    
    def _on_complete_experiment(self, btn):
        """Handle complete experiment button click"""
        with self.output:
            clear_output()
            self._run_complete_experiment()
    
    def _on_custom_experiment(self, btn):
        """Handle custom experiment button click"""
        with self.output:
            clear_output()
            self._run_custom_experiment()
    
    def _on_exit(self, btn):
        """Handle exit button click"""
        with self.output:
            clear_output()
            print("👋 Thank you for using the 3C-BOT Research Simulator!")
            print("\nFor academic citation:")
            print("Silva, A.N. et al. Computational modeling of behavioral dynamics")
            print("in human-robot organizational communities.")
    
    def _run_complete_experiment(self):
        """Run complete experiment in a separate thread"""
        print("🔬 COMPLETE RESEARCH EXPERIMENT")
        print("This will test all 5 population configurations with statistical analysis.")
        print("Please wait...")
        
        def run_experiment():
            self.results = self.experiment.run_complete_experiment()
            
        # Run in thread to avoid blocking
        thread = threading.Thread(target=run_experiment)
        thread.start()
        
        # Simple progress indicator
        for i in range(10):
            time.sleep(2)
            print(".", end='', flush=True)
        
        thread.join()
        
        if self.results:
            print(f"\n✅ Experiment completed!")
            print(f"Best configuration: {self.results['best_config']}")
            print(f"Results saved to: results/")
    
    def _run_custom_experiment(self):
        """Run custom experiment with interactive parameter selection"""
        print("🎯 CUSTOM EXPERIMENT")
        print("Configure your experiment parameters:")
        
        # Configuration selection
        config_dropdown = widgets.Dropdown(
            options=[(f"{config.name} ({int(config.value[0]*100)}%H/{int(config.value[1]*100)}%R)", config) 
                    for config in ConfigurationType],
            description='Configuration:',
            layout=widgets.Layout(width='400px')
        )
        
        # Scale selection  
        scale_dropdown = widgets.Dropdown(
            options=[(f"{scale.name} ({scale.value[0]} agents)", scale) 
                    for scale in ExperimentScale],
            description='Scale:',
            layout=widgets.Layout(width='400px')
        )
        
        # Cycles input
        cycles_slider = widgets.IntSlider(
            value=1000,
            min=500,
            max=2000,
            step=100,
            description='Cycles:',
            layout=widgets.Layout(width='400px')
        )
        
        # Run button
        run_btn = widgets.Button(
            description="🚀 Run Custom Experiment",
            button_style='success',
            layout=widgets.Layout(width='200px')
        )
        
        # Output for this section
        custom_output = widgets.Output()
        
        def on_run_clicked(btn):
            with custom_output:
                clear_output()
                print("Starting custom experiment...")
                print(f"Configuration: {config_dropdown.label}")
                print(f"Scale: {scale_dropdown.label}")
                print(f"Cycles: {cycles_slider.value}")
                print("Please wait...")
                
                # Run the experiment
                results = self.experiment.run_single_experiment(
                    config_dropdown.value,
                    scale_dropdown.value, 
                    cycles_slider.value,
                    seed=42
                )
                
                if results:
                    print(f"\n✅ Custom experiment completed!")
                    print(f"Final trust: {results['final_trust']:.3f}")
                    print(f"Symbiosis achieved: {'✅ Yes' if results['achieved_symbiosis'] else '❌ No'}")
        
        run_btn.on_click(on_run_clicked)
        
        # Display the custom experiment interface
        display(widgets.VBox([
            config_dropdown,
            scale_dropdown, 
            cycles_slider,
            run_btn,
            custom_output
        ]))


def main():
    """Main function to display the widgets interface"""
    simulator = ResearchSimulatorWidgets()
    menu = simulator.create_main_menu()
    display(menu)


# For direct execution in Jupyter
if __name__ == "__main__":
    main()
