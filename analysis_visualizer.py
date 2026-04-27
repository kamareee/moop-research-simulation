import matplotlib.pyplot as plt
import pickle
import numpy as np

def plot_pareto():
    with open('results.pkl', 'rb') as f:
        res = pickle.load(f)
    
    # Sort for plot
    F = res.F[res.F[:, 0].argsort()]
    
    plt.figure(figsize=(8, 5))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.plot(F[:, 0], F[:, 1], 'o--', color='darkblue', label='Pareto Front')
    plt.fill_between(F[:, 0], F[:, 1], F[:, 1].max(), alpha=0.1, color='blue')
    
    plt.title("Fleet Charging Pareto Front: Operational Stress Day", fontsize=14)
    plt.xlabel("Total Charging & Detour Cost ($)", fontsize=12)
    plt.ylabel("Total Fleet Energy Shortfall (kWh)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pareto_front.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_pareto()