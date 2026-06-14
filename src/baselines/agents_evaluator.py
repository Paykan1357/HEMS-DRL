"""
HEMS Agents Evaluator for Home Energy Management Systems.

Evaluates DRL agents (TD3, SAC, ACKTR) on metrics including:
- Total operational cost
- Thermal comfort (CVI)
- Battery degradation
- PV self-consumption
- Unmet load
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class AgentsEvaluator:
    """
    Evaluate and compare multiple DRL agents on HEMS environment.
    """
    
    def __init__(self, env, agents_list):
        """
        Args:
            env: HEMS environment
            agents_list: List of agent instances to evaluate
        """
        self.env = env
        self.agents_list = agents_list
    
    def evaluate_all(
        self,
        exploration=False,
        plot_metrics=True,
        plot_power_flows=True,
        save_folder=None,
    ):
        """
        Evaluate all agents and generate comparison plots.
        
        Args:
            exploration: Whether to use exploration during evaluation
            plot_metrics: Whether to generate metric comparison plots
            plot_power_flows: Whether to plot power flow time series
            save_folder: Optional folder to save figures
            
        Returns:
            results_df: DataFrame with agent performance metrics
        """
        agent_results = []
        
        for agent in self.agents_list:
            print(f"Evaluating {agent.name}...")
            
            # Evaluate agent
            total_reward, step_infos, final_metrics = agent.evaluate(
                self.env, exploration=exploration
            )
            
            # Store results
            result = {
                'agent': agent.name,
                'total_reward': total_reward,
                'total_energy_cost': final_metrics.get('total_energy_cost', 0),
                'total_unmet_load': final_metrics.get('total_unmet_load', 0),
                'avg_soc': final_metrics.get('avg_soc', 0),
                'pv_utilization': final_metrics.get('pv_utilization', 0),
                'avg_comfort_penalty': final_metrics.get('avg_comfort_penalty', 0),
            }
            
            # Add CVI if available
            if 'cvi' in final_metrics:
                result['cvi'] = final_metrics['cvi']
            
            agent_results.append(result)
            
            # Save step infos for plotting
            agent.step_infos = step_infos
        
        # Create results DataFrame
        results_df = pd.DataFrame(agent_results).set_index('agent')
        
        print("\n=== Agent Performance Metrics ===")
        print(results_df)
        print("\n")
        
        # Generate plots
        if plot_metrics:
            self._plot_metrics_comparison(results_df, save_folder)
        
        if plot_power_flows:
            self._plot_power_flows(save_folder)
        
        return results_df
    
    def _plot_metrics_comparison(self, results_df, save_folder=None):
        """
        Create bar charts comparing agent performance across metrics.
        """
        metrics_to_plot = [
            'total_energy_cost',
            'total_unmet_load', 
            'pv_utilization',
            'avg_comfort_penalty'
        ]
        
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics_to_plot):
            values = results_df[metric].values
            agents = results_df.index.values
            
            bars = ax.bar(agents, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_folder:
            plt.savefig(f"{save_folder}/metrics_comparison.png", dpi=150, bbox_inches="tight")
        
        plt.show()
    
    def _plot_power_flows(self, save_folder=None):
        """
        Plot power flow time series for each agent.
        """
        for agent in self.agents_list:
            if not hasattr(agent, 'step_infos'):
                continue
            
            step_infos = agent.step_infos
            
            # Extract time series data
            steps = range(len(step_infos))
            grid_import = [info.get('grid_import', 0) for info in step_infos]
            pv_used = [info.get('pv_used', 0) for info in step_infos]
            battery_power = [info.get('battery_power', 0) for info in step_infos]
            load_served = [info.get('load_served', 0) for info in step_infos]
            soc = [info.get('soc', 0) for info in step_infos]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Power Flows - {agent.name}", fontsize=14)
            
            # Plot 1: Power balance
            ax = axes[0, 0]
            ax.plot(steps, pv_used, label='PV Used', color='green', alpha=0.8)
            ax.plot(steps, grid_import, label='Grid Import', color='red', alpha=0.8)
            ax.plot(steps, battery_power, label='Battery', color='blue', alpha=0.8)
            ax.plot(steps, load_served, label='Load Served', color='black', linestyle='--', alpha=0.8)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Power (kW)')
            ax.set_title('Power Balance')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Battery SoC
            ax = axes[0, 1]
            ax.plot(steps, soc, color='blue')
            ax.axhline(y=0.9, color='red', linestyle='--', label='SoC max')
            ax.axhline(y=0.1, color='red', linestyle='--', label='SoC min')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('State of Charge')
            ax.set_title('Battery SoC')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Unmet load
            ax = axes[1, 0]
            unmet_load = [info.get('unmet_load', 0) for info in step_infos]
            ax.fill_between(steps, 0, unmet_load, color='orange', alpha=0.5)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Unmet Load (kW)')
            ax.set_title('Unmet Load')
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Comfort
            ax = axes[1, 1]
            temp_in = [info.get('temperature_in', 22) for info in step_infos]
            temp_set = [info.get('temperature_set', 22) for info in step_infos]
            ax.plot(steps, temp_in, label='Indoor Temp', color='orange')
            ax.axhline(y=22, color='red', linestyle='--', label='Setpoint')
            ax.fill_between(steps, 21.5, 22.5, alpha=0.3, color='green', label='Comfort Zone (±0.5°C)')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Temperature (°C)')
            ax.set_title('Thermal Comfort')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_folder:
                plt.savefig(f"{save_folder}/power_flows_{agent.name}.png", dpi=150, bbox_inches="tight")
            
            plt.show()
    
    def plot_cvi_comparison(self, save_folder=None):
        """
        Plot CVI comparison across agents.
        """
        agents = []
        cvi_values = []
        
        for agent in self.agents_list:
            if hasattr(agent, 'step_infos'):
                cvi = sum([
                    info.get('comfort_penalty', 0) 
                    for info in agent.step_infos
                ])
                agents.append(agent.name)
                cvi_values.append(cvi)
        
        if not agents:
            print("No CVI data available.")
            return
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(agents, cvi_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        ax.set_xlabel('Agent')
        ax.set_ylabel('CVI (Comfort Violation Index)')
        ax.set_title('Thermal Comfort Comparison')
        
        for bar, val in zip(bars, cvi_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_folder:
            plt.savefig(f"{save_folder}/cvi_comparison.png", dpi=150, bbox_inches="tight")
        
        plt.show()
