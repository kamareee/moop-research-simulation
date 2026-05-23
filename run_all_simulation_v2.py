# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


# --- 1. DATA GENERATION & CACHING ---
def get_advanced_case_data(N, F_count, M, p_site_max, alpha=0.65):
    """
    Generates stochastic data for the orchestration problem[cite: 1, 201].
    - Strategy 1: Spillover via 'alpha' energy intensity[cite: 204].
    - Strategy 2: Stochastic dwell windows via random shift offsets[cite: 14, 206].
    """
    T, dt = 96, 0.25
    pi_grid = np.zeros(T)
    # AU Commercial TOU Tariff [cite: 68, 79]
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
                "power": {"L2": 7.0, "DC": 80.0},
                "wait_profile": 10 + 15 * np.sin(np.pi * t / 96) ** 2,
            }
        )

    fleet = []
    for i in range(N):
        shift_offset = random.randint(-6, 6)  # Dwell-time compression logic [cite: 206]
        dwells = [
            [max(0, 0), min(96, 28 + shift_offset)],
            [max(0, 42 + shift_offset), min(96, 68 + shift_offset)],
            [max(0, 82 + shift_offset), 96],
        ]

        bat_max = 75.0
        total_demand = bat_max * (alpha + random.uniform(-0.1, 0.1))
        fleet.append(
            {
                "bat_max": bat_max,
                "soc_init": random.uniform(0.15, 0.30),
                "e_trip": [total_demand * 0.45, total_demand * 0.55],
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
    }


def load_or_generate_case_data(case_dir, N, F, M, p_site_max, alpha):
    """Ensures algorithm comparisons use identical data sessions[cite: 220]."""
    data_path = os.path.join(case_dir, f"sim_data_alpha{alpha}.pkl")
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            return pickle.load(f)

    case_data = get_advanced_case_data(N, F, M, p_site_max, alpha)
    with open(data_path, "wb") as f:
        pickle.dump(case_data, f)
    return case_data


# --- 2. PROBLEM DEFINITION ---
class ChargingProblem(ElementwiseProblem):
    def __init__(self, problem_data):
        self.case_data = problem_data
        n_windows = sum(len(v["dwells"]) for v in self.case_data["fleet"])
        # Objectives: f1 (Cost), f2 (Shortfall) [cite: 76, 90]
        # Constraints: C5 (Site Power), C6 (EVSE Limit) [cite: 121, 123]
        super().__init__(
            n_var=n_windows, n_obj=2, n_constr=2, xl=0, xu=1 + 2 * self.case_data["F"]
        )

    def _evaluate(self, x, out, *args, **kwargs):
        cost, shortfall, load, evses, _, _ = self.simulate_solution(x)
        p_viol = np.sum(
            np.maximum(
                0,
                load
                + self.case_data["p_base"]
                - self.case_data["p_pv"]
                - self.case_data["p_site_max"],
            )
        )
        e_viol = np.sum(np.maximum(0, evses - self.case_data["evse_m"]))
        out["F"], out["G"] = [cost, shortfall], [p_viol, e_viol]

    def simulate_solution(self, x):
        x = np.round(x).astype(int)
        d = self.case_data
        soc = np.array([v["soc_init"] * v["bat_max"] for v in d["fleet"]])
        cost, shortfall = 0.0, 0.0
        load, evses = np.zeros(d["T"]), np.zeros(d["T"])
        e_depot_t, e_pub_t = np.zeros(d["T"]), np.zeros(d["T"])

        idx = 0
        for i, v in enumerate(d["fleet"]):
            for dwell in v["dwells"]:
                choice, start, end = x[idx], dwell[0], dwell[1]
                idx += 1
                if choice == 1:  # Depot Assignment [cite: 5, 79, 113]
                    p = min(22.0, (v["bat_max"] - soc[i]) / d["dt"])
                    energy = p * d["dt"] * 0.9
                    soc[i] += energy
                    cost += np.sum(d["pi_grid"][start:end] * p * d["dt"])
                    load[start:end] += p
                    evses[start:end] += 1
                    e_depot_t[start:end] += (p * d["dt"]) / (end - start)
                elif choice > 1:  # Public Assignment [cite: 5, 81, 127]
                    f_idx = (choice - 2) % d["F"]
                    c_type = "L2" if choice < 2 + d["F"] else "DC"
                    st = d["stations"][f_idx]
                    wait = st["wait_profile"][start]
                    net_min = ((end - start) * d["dt"] * 60) - v["detour"][f_idx] - wait
                    if net_min > 0:
                        e = min(
                            st["power"][c_type] * (net_min / 60), v["bat_max"] - soc[i]
                        )
                        soc[i] += e * 0.95
                        cost += e * st["pi_pub"][c_type] + d["c_d_t"] * (
                            v["detour"][f_idx] + wait
                        )
                        e_pub_t[start] += e
            for trip_e in v["e_trip"]:  # Reliability tracking [cite: 94, 167]
                if soc[i] < trip_e:
                    shortfall += trip_e - soc[i]
                    soc[i] = 0
                else:
                    soc[i] -= trip_e
        return cost, shortfall, load, evses, e_depot_t, e_pub_t


# --- 3. MASTER SIMULATION ---
def run_comparison_sim():
    scenarios = [[45, 10, 15], [60, 10, 20], [90, 25, 25]]
    pop_sizes = [100, 150, 200]
    alpha = 0.65
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Formatting for single-column A4 width (3.5")
    plt.rcParams.update({"font.size": 8, "figure.titlesize": 10, "axes.labelsize": 8})

    for N, F, M in scenarios:
        case_id = f"N{N}_F{F}_M{M}"
        case_dir = os.path.join(base_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)
        print(f"\n--- Scenario: {case_id} (Alpha={alpha}) ---")

        # Site limit logic [cite: 56, 121]
        p_site_limit = (M / 15) * 200.0
        case_data = load_or_generate_case_data(case_dir, N, F, M, p_site_limit, alpha)
        problem = ChargingProblem(case_data)

        # 1. Pareto Comparison Plot
        fig1, ax1 = plt.subplots(figsize=(3.5, 3.0))
        colors = plt.get_cmap("viridis")(np.linspace(0, 0.8, len(pop_sizes)))
        best_res = None

        for i, pop in enumerate(pop_sizes):
            res = minimize(problem, NSGA2(pop_size=pop), ("n_gen", 100), seed=1)
            if pop == 200:
                best_res = res
            F_res = res.F
            idx = np.argsort(F_res[:, 0])
            ax1.plot(
                F_res[idx, 0],
                F_res[idx, 1],
                "o-",
                markersize=3,
                color=colors[i],
                label=f"P:{pop}",
                alpha=0.8,
            )

        ax1.set_title(f"Pareto Front: {case_id}")
        ax1.set_xlabel("Total Cost ($)")
        ax1.set_ylabel("Shortfall (kWh)")
        ax1.legend(prop={"size": 6})
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        fig1.savefig(os.path.join(case_dir, "pareto.png"), dpi=300)

        if best_res is not None:
            # 2. Extract Knee Point (Min cost at Zero Shortfall [cite: 33, 93])
            z_mask = best_res.F[:, 1] <= 1e-5
            idx = (
                np.where(z_mask)[0][np.argmin(best_res.F[z_mask, 0])]
                if np.any(z_mask)
                else np.argmin(best_res.F[:, 0] + best_res.F[:, 1] * 1e6)
            )
            _, _, load, _, e_depot, e_pub = problem.simulate_solution(best_res.X[idx])

            # Depot Load Profile
            fig2, ax2 = plt.subplots(figsize=(3.5, 2.5))
            net_load = load + case_data["p_base"] - case_data["p_pv"]
            ax2.plot(net_load, color="navy", linewidth=1, label="Net Load")
            ax2.axhline(
                y=case_data["p_site_max"],
                color="red",
                linestyle="--",
                linewidth=1,
                label="Limit",
            )
            ax2.set_title(f"Load Profile: {case_id}")
            ax2.set_xlabel("Time Slot")
            ax2.set_ylabel("Power (kW)")
            ax2.legend(prop={"size": 6})
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()
            fig2.savefig(os.path.join(case_dir, "load_profile.png"), dpi=300)

            # 3. Stacked Energy Delivery Bar Chart
            fig3, ax3 = plt.subplots(figsize=(3.5, 2.5))
            ax3.bar(range(96), e_depot, color="skyblue", label="Depot")
            ax3.bar(range(96), e_pub, bottom=e_depot, color="orange", label="Public")
            ax3.set_title(f"Energy Source: {case_id}")
            ax3.set_xlabel("Time Slot")
            ax3.set_ylabel("Energy (kWh)")
            ax3.legend(prop={"size": 6})
            ax3.grid(True, axis="y", alpha=0.3)
            fig3.tight_layout()
            fig3.savefig(os.path.join(case_dir, "energy_stacked.png"), dpi=300)

        plt.close("all")


if __name__ == "__main__":
    run_comparison_sim()
