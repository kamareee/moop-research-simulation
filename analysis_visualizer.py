# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring, wrong-import-order

import matplotlib.pyplot as plt
import pickle
import numpy as np
from optimizer import ChargingProblem


def save_individual_plots():
    # Load the results and synthetic data
    with open("results.pkl", "rb") as f:
        res = pickle.load(f)
    with open("sim_data.pkl", "rb") as f:
        data = pickle.load(f)

    # Identify the 'Knee' Point for detailed analysis
    F = res.F
    best_idx = np.argmin(np.sum(F**2, axis=1))
    best_x = res.X[best_idx].astype(int)

    # --- Re-simulate the 'Best' solution to extract data ---
    T = data["T"]
    depot_load = np.zeros(T)
    cost_breakdown = {"Depot ToU": 0, "Public Energy": 0, "Detour/Wait": 0}
    energy_dist = {"Depot L2": 0, "Public L2": 0, "Public DC": 0}

    idx = 0
    for i, v in enumerate(data["fleet"]):
        soc_i = v["soc_init"] * v["bat_max"]
        for j, dwell in enumerate(v["dwells"]):
            choice = best_x[idx]
            idx += 1
            start, end = dwell

            if choice == 1:  # Depot Assignment
                p = min(22.0, (v["bat_max"] - soc_i) / data["dt"])
                soc_i += p * data["dt"] * 0.9
                cost_breakdown["Depot ToU"] += np.sum(
                    data["pi_grid"][start:end] * p * data["dt"]
                )
                depot_load[start:end] += p
                energy_dist["Depot L2"] += p * data["dt"]
            elif choice > 1:  # Public Assignment
                f_idx = (choice - 2) % data["F"]
                c_type = "L2" if choice < 2 + data["F"] else "DC"
                st = data["stations"][f_idx]
                wait = st["wait_profile"][start]
                net_min = ((end - start) * data["dt"] * 60) - v["detour"][f_idx] - wait
                if net_min > 0:
                    e = min(st["power"][c_type] * (net_min / 60), v["bat_max"] - soc_i)
                    soc_i += e * 0.95
                    cost_breakdown["Public Energy"] += e * st["pi_pub"][c_type]
                    cost_breakdown["Detour/Wait"] += data["c_d_t"] * (
                        v["detour"][f_idx] + wait
                    )
                    energy_dist["Public L2" if c_type == "L2" else "Public DC"] += e

    plt.style.use("ggplot")

    # 1. Pareto Front
    plt.figure(figsize=(8, 6))
    sorted_F = F[F[:, 0].argsort()]
    plt.plot(sorted_F[:, 0], sorted_F[:, 1], "o--", color="navy", markersize=4)
    plt.scatter(
        F[best_idx, 0], F[best_idx, 1], color="red", s=100, label="Selected Optimal"
    )
    plt.title("Pareto Front: Cost vs. Reliability")
    plt.xlabel("Total Cost ($)")
    plt.ylabel("Energy Shortfall (kWh)")
    plt.legend()
    plt.savefig("1_pareto_front.png", dpi=300)
    plt.close()

    # 2. Cost Composition
    plt.figure(figsize=(8, 6))
    plt.bar(
        cost_breakdown.keys(),
        cost_breakdown.values(),
        color=["teal", "orange", "crimson"],
    )
    plt.title("Total Cost Composition")
    plt.ylabel("Cost ($)")
    plt.savefig("2_cost_composition.bar.png", dpi=300)
    plt.close()

    # 3. Depot Power Profile
    plt.figure(figsize=(10, 6))
    plt.plot(
        depot_load + data["p_base"],
        color="purple",
        linewidth=2,
        label="Total Depot Load",
    )
    plt.axhline(
        y=data["p_site_max"], color="red", linestyle="--", label="Site Power Limit"
    )
    plt.fill_between(range(T), depot_load + data["p_base"], color="purple", alpha=0.2)
    plt.title("Depot Load Profile vs. Infrastructure Capacity")
    plt.xlabel("Time Slot (15-min)")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.savefig("3_depot_load_profile.png", dpi=300)
    plt.close()

    # 4. Energy Distribution
    plt.figure(figsize=(8, 6))
    plt.pie(
        energy_dist.values(),
        labels=energy_dist.keys(),
        autopct="%1.1f%%",
        colors=["#66b3ff", "#99ff99", "#ffcc99"],
    )
    plt.title("Fleet Energy Source Distribution")
    plt.savefig("4_energy_distribution_pie.png", dpi=300)
    plt.close()

    print("Success: All results are saved.")


if __name__ == "__main__":
    save_individual_plots()
