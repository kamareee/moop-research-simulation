# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


# --- 1. DATA GENERATION & CACHING ---
def get_advanced_case_data(N, F_count, M, alpha):
    """
    Generates stochastic data based on Research Question 3 formulation[cite: 1].
    Includes depot chargers (M) [cite: 47], site power limits[cite: 56],
    and heterogeneous public stations (F)[cite: 47].
    """
    T, dt = 96, 0.25
    p_site_max = (M / 15) * 200.0  # Scaling site limit based on EVSE count [cite: 56]
    pi_grid = np.zeros(T)
    # Australian Commercial TOU Tariff [cite: 68]
    pi_grid[0:28], pi_grid[28:60], pi_grid[60:84], pi_grid[84:88], pi_grid[88:96] = (
        0.1781,
        0.2087,
        0.3121,
        0.2087,
        0.1781,
    )

    t = np.arange(T)
    p_pv = (p_site_max * 0.15) * np.maximum(0, np.sin(np.pi * (t - 24) / 48))
    p_base = (p_site_max * 0.125) + 10 * np.random.rand(T)

    stations = []
    for f in range(F_count):
        stations.append(
            {
                "pi_pub": {"L2": 0.25, "DC": 0.40},
                "power": {"L2": 7.0, "DC": 50.0},
                "wait_profile": 10 + 15 * np.sin(np.pi * t / 96) ** 2,
            }
        )

    fleet = []
    total_fleet_demand = 0
    for i in range(N):
        # Strategy 2: Stochastic Dwell Windows (Staggered shifts)
        shift_offset = random.randint(-6, 6)
        dwells = [
            [max(0, 0), min(96, 28 + shift_offset)],
            [max(0, 42 + shift_offset), min(96, 68 + shift_offset)],
            [max(0, 82 + shift_offset), 96],
        ]

        bat_max = 75.0
        # Strategy 1: Energy intensity alpha (Operational Stress)
        total_demand = bat_max * (alpha + random.uniform(-0.05, 0.05))
        trips = [total_demand * 0.45, total_demand * 0.55]
        total_fleet_demand += sum(trips)

        fleet.append(
            {
                "bat_max": bat_max,
                "soc_init": 0.20,
                "e_trip": trips,
                "dwells": dwells,
                "detour": np.random.randint(10, 20, size=F_count),
            }
        )

    return {
        "N": N,
        "T": T,
        "F": F_count,
        "dt": dt,
        "pi_grid": pi_grid,
        "p_pv": p_pv,
        "p_base": p_base,
        "stations": stations,
        "fleet": fleet,
        "p_site_max": p_site_max,
        "evse_m": M,
        "c_d_t": 0.30,
        "total_fleet_demand": total_fleet_demand,
    }


def load_or_generate_case_data(case_dir, N, F, M, alpha):
    data_path = os.path.join(case_dir, f"sim_data_N{N}_M{M}_alpha{alpha}.pkl")
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            return pickle.load(f)
    case_data = get_advanced_case_data(N, F, M, alpha)
    with open(data_path, "wb") as f:
        pickle.dump(case_data, f)
    return case_data


# --- 2. PROBLEM DEFINITION ---
class ChargingProblem(ElementwiseProblem):
    def __init__(self, data):
        self.d = data
        n_windows = sum(len(v["dwells"]) for v in self.d["fleet"])
        # Objectives: f1 (Cost), f2 (Shortfall)
        # Constraints: C5 (Power Limit), C6 (EVSE Limit)
        super().__init__(
            n_var=n_windows, n_obj=2, n_constr=2, xl=0, xu=1 + 2 * self.d["F"]
        )

    def _evaluate(self, x, out, *args, **kwargs):
        c, s, load, evses, _, _ = self.simulate(x)
        p_viol = np.sum(
            np.maximum(
                0, load + self.d["p_base"] - self.d["p_pv"] - self.d["p_site_max"]
            )
        )  #
        e_viol = np.sum(np.maximum(0, evses - self.d["evse_m"]))  #
        out["F"], out["G"] = [c, s], [p_viol, e_viol]

    def simulate(self, x):
        x = np.round(x).astype(int)
        d = self.d
        soc = np.array([v["soc_init"] * v["bat_max"] for v in d["fleet"]])
        cost, shortfall = 0, 0
        load, evses = np.zeros(d["T"]), np.zeros(d["T"])
        e_depot, e_pub = 0, 0

        idx = 0
        for i, v in enumerate(d["fleet"]):
            for dwell in v["dwells"]:
                choice, start, end = x[idx], dwell[0], dwell[1]
                idx += 1
                if choice == 1:  # Depot Assignment (C10)
                    p = min(22.0, (v["bat_max"] - soc[i]) / d["dt"])
                    energy = p * d["dt"] * 0.9
                    soc[i] += energy
                    e_depot += energy
                    cost += np.sum(d["pi_grid"][start:end] * p * d["dt"])
                    load[start:end] += p
                    evses[start:end] += 1
                elif choice > 1:  # Public Assignment (C11)
                    f_idx, c_type = (choice - 2) % d["F"], (
                        "L2" if choice < 2 + d["F"] else "DC"
                    )
                    st = d["stations"][f_idx]
                    wait = st["wait_profile"][start]
                    net = ((end - start) * d["dt"] * 60) - v["detour"][f_idx] - wait
                    if net > 0:
                        e = min(st["power"][c_type] * (net / 60), v["bat_max"] - soc[i])
                        soc[i] += e * 0.95
                        e_pub += e * 0.95
                        cost += e * st["pi_pub"][c_type] + d["c_d_t"] * (
                            v["detour"][f_idx] + wait
                        )
            for trip_e in v["e_trip"]:  # Shortfall Calculation (C14) [cite: 167]
                if soc[i] < trip_e:
                    shortfall += trip_e - soc[i]
                    soc[i] = 0
                else:
                    soc[i] -= trip_e
        return cost, shortfall, load, evses, e_depot, e_pub


# --- 3. MASTER SIMULATION LOOP ---
def run_comparison_sim():
    scenarios = [[45, 10, 15], [60, 10, 20], [90, 20, 25]]
    pop_sizes = [100, 150, 200]
    alpha = 0.75
    base_dir = os.path.dirname(os.path.abspath(__file__))

    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.8})

    for N, F, M in scenarios:
        case_id = f"N{N}_F{F}_M{M}"
        case_dir = os.path.join(base_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)
        print(f"\n--- Scenario: {case_id} ---")

        data = load_or_generate_case_data(case_dir, N, F, M, alpha)
        problem = ChargingProblem(data)

        # 1. Pareto Front Graph
        fig_p, ax_p = plt.subplots(figsize=(3.5, 3.0))
        colors = plt.get_cmap("viridis")(np.linspace(0, 0.8, len(pop_sizes)))
        best_res = None

        for i, pop in enumerate(pop_sizes):
            print(f"  Optimizing Pop: {pop}...")
            res = minimize(problem, NSGA2(pop_size=pop), ("n_gen", 100), seed=1)
            if pop == 200:
                best_res = res

            f_sort = res.F[np.argsort(res.F[:, 0])]
            ax_p.plot(
                f_sort[:, 0],
                f_sort[:, 1],
                "o-",
                markersize=3,
                color=colors[i],
                label=f"P:{pop}",
            )

        ax_p.set_title(f"Pareto Front: {case_id}")
        ax_p.set_xlabel("Cost ($)")
        ax_p.set_ylabel("Shortfall (kWh)")
        ax_p.legend(fontsize=6)
        ax_p.grid(True, alpha=0.2)
        fig_p.tight_layout()
        fig_p.savefig(os.path.join(case_dir, "pareto.png"), dpi=300)

        # 2. Extract Knee Point (Optimal Solution)
        z_mask = best_res.F[:, 1] <= 1e-5
        idx = (
            np.where(z_mask)[0][np.argmin(best_res.F[z_mask, 0])]
            if np.any(z_mask)
            else np.argmin(best_res.F[:, 0] + best_res.F[:, 1] * 1e6)
        )
        _, _, load, _, e_dep, e_pub = problem.simulate(best_res.X[idx])

        # 3. Fulfillment Chart
        fig_f, ax_f = plt.subplots(figsize=(3.5, 2.5))
        d_pct, p_pct = (e_dep / data["total_fleet_demand"]) * 100, (
            e_pub / data["total_fleet_demand"]
        ) * 100
        ax_f.bar(0, d_pct, color="#0077b6", edgecolor="white", linewidth=0.5)
        ax_f.bar(
            0, p_pct, bottom=d_pct, color="#00b4d8", edgecolor="white", linewidth=0.5
        )

        total_f = d_pct + p_pct
        ax_f.text(
            0,
            total_f - 10,
            f"{int(total_f)}%",
            ha="center",
            color="white",
            fontweight="bold",
            fontsize=7,
        )
        ax_f.set_ylabel("Demand Fulfilled (%)")
        ax_f.set_title(f"Fulfillment: {case_id}", fontweight="bold")
        ax_f.set_ylim(0, 110)
        ax_f.set_xticks([])
        ax_f.axhline(100, color="black", linestyle="--", linewidth=0.5)

        ax_f.legend(
            handles=[
                mpatches.Patch(color="#0077b6", label="Depot"),
                mpatches.Patch(color="#00b4d8", label="Public"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=2,
            frameon=False,
            fontsize=6,
        )
        fig_f.tight_layout()
        fig_f.savefig(os.path.join(case_dir, "fulfillment.png"), dpi=300)

        plt.close("all")


if __name__ == "__main__":
    run_comparison_sim()
