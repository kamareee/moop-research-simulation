# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


# --- 1. ADVANCED DATA GENERATION ---
def get_advanced_case_data(N, F_count, M, p_site_max, alpha=0.6):
    """
    Generates heterogeneous fleet data with stochastic dwells and energy intensity.
    Strategy 1: Controlled via 'alpha' (Fleet energy intensity)[cite: 2].
    Strategy 2: Controlled via 'shift_offset' (Stochastic dwell windows)[cite: 2].
    """
    T, dt = 96, 0.25
    pi_grid = np.zeros(T)
    # Australian Commercial TOU Tariff logic[cite: 1]
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
                "power": {"L2": 7.0, "DC": 150.0},
                "wait_profile": 10 + 15 * np.sin(np.pi * t / 96) ** 2,
            }
        )

    fleet = []
    for i in range(N):
        # Strategy 2: Staggered shifts using random offsets[cite: 2]
        shift_offset = random.randint(-6, 6)
        dwells = [
            [max(0, 0), min(96, 28 + shift_offset)],
            [max(0, 42 + shift_offset), min(96, 68 + shift_offset)],
            [max(0, 82 + shift_offset), 96],
        ]

        # Strategy 1: Energy demand based on stress factor alpha[cite: 2]
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


def load_or_generate_case_data(case_dir, N, F_count, M, p_site_max, alpha):
    data_path = os.path.join(case_dir, f"sim_data_alpha{alpha}.pkl")
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            return pickle.load(f)

    case_data = get_advanced_case_data(N, F_count, M, p_site_max, alpha)
    with open(data_path, "wb") as f:
        pickle.dump(case_data, f)
    return case_data


# --- 2. PROBLEM DEFINITION ---
class ChargingProblem(ElementwiseProblem):
    def __init__(self, problem_data):
        self.case_data = problem_data  # Renamed to avoid pymoo KeyError[cite: 1]
        n_windows = sum(len(v["dwells"]) for v in self.case_data["fleet"])
        super().__init__(
            n_var=n_windows, n_obj=2, n_constr=2, xl=0, xu=1 + 2 * self.case_data["F"]
        )

    def _evaluate(self, x, out, *args, **kwargs):
        total_cost, total_shortfall, depot_load, active_evses = self.simulate_solution(
            x
        )
        power_violation = np.sum(
            np.maximum(
                0,
                depot_load
                + self.case_data["p_base"]
                - self.case_data["p_pv"]
                - self.case_data["p_site_max"],
            )
        )
        evse_violation = np.sum(np.maximum(0, active_evses - self.case_data["evse_m"]))
        out["F"], out["G"] = [total_cost, total_shortfall], [
            power_violation,
            evse_violation,
        ]

    def simulate_solution(self, x):
        x = np.round(x).astype(int)
        d = self.case_data
        soc = np.array([v["soc_init"] * v["bat_max"] for v in d["fleet"]])
        total_cost, total_shortfall = 0.0, 0.0
        depot_load, active_evses = np.zeros(d["T"]), np.zeros(d["T"])

        idx = 0
        for i, v in enumerate(d["fleet"]):
            for dwell in v["dwells"]:
                choice, start, end = x[idx], dwell[0], dwell[1]
                idx += 1
                if choice == 1:  # Depot Assignment (3-phase 22kW logic)[cite: 1, 2]
                    p = min(22.0, (v["bat_max"] - soc[i]) / d["dt"])
                    soc[i] += p * d["dt"] * 0.9
                    total_cost += np.sum(d["pi_grid"][start:end] * p * d["dt"])
                    depot_load[start:end] += p
                    active_evses[start:end] += 1
                elif choice > 1:  # Public Assignment[cite: 1, 2]
                    f_idx, c_type = (choice - 2) % d["F"], (
                        "L2" if choice < 2 + d["F"] else "DC"
                    )
                    st = d["stations"][f_idx]
                    wait = st["wait_profile"][start]
                    net = ((end - start) * d["dt"] * 60) - v["detour"][f_idx] - wait
                    if net > 0:
                        e = min(st["power"][c_type] * (net / 60), v["bat_max"] - soc[i])
                        soc[i] += e * 0.95
                        total_cost += e * st["pi_pub"][c_type] + d["c_d_t"] * (
                            v["detour"][f_idx] + wait
                        )
            for trip_e in v["e_trip"]:
                if soc[i] < trip_e:
                    total_shortfall += trip_e - soc[i]
                    soc[i] = 0
                else:
                    soc[i] -= trip_e
        return total_cost, total_shortfall, depot_load, active_evses


# --- 3. MASTER SIMULATION LOOP ---
def run_comparison_sim():
    scenarios = [[45, 10, 15], [60, 10, 20], [90, 15, 30]]
    pop_sizes = [100, 150, 200]
    alpha_stress = 0.65  # High energy intensity to force depot spillover[cite: 2]

    base_dir = os.path.dirname(os.path.abspath(__file__))

    for N, F, M in scenarios:
        case_id = f"N{N}_F{F}_M{M}"
        case_dir = os.path.join(base_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)
        print(f"\n--- Scenario: {case_id} (Alpha={alpha_stress}) ---")

        p_site_limit = (M / 15) * 200.0  # Scaling site limit[cite: 1]
        case_data = load_or_generate_case_data(
            case_dir, N, F, M, p_site_limit, alpha_stress
        )
        problem = ChargingProblem(case_data)

        plt.figure(figsize=(10, 7))
        colors = plt.get_cmap("plasma")(np.linspace(0, 1, len(pop_sizes)))
        best_res_overall = None

        for i, pop in enumerate(pop_sizes):
            print(f"  Pop Size: {pop}...")
            res = minimize(problem, NSGA2(pop_size=pop), ("n_gen", 100), seed=1)
            if pop == 250:
                best_res_overall = res

            F_res = res.F
            if F_res is not None:
                idx = np.argsort(F_res[:, 0])
                plt.plot(
                    F_res[idx, 0],
                    F_res[idx, 1],
                    "o--",
                    color=colors[i],
                    label=f"Pop: {pop}",
                    alpha=0.7,
                )

        # Plot 1: Pareto Front
        plt.title(f"Pareto Front: {case_id} (N={N}, F={F}, M={M})")
        plt.xlabel("Total Cost ($)")
        plt.ylabel("Shortfall (kWh)")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.savefig(os.path.join(case_dir, "pareto_comparison.png"), dpi=300)
        plt.close()

        # Plot 2: Depot Load Profile for Optimal Solution
        if best_res_overall is not None:
            # Select lowest cost among zero-shortfall solutions[cite: 1]
            z_mask = best_res_overall.F[:, 1] <= 1e-5
            best_idx = (
                np.where(z_mask)[0][np.argmin(best_res_overall.F[z_mask, 0])]
                if np.any(z_mask)
                else np.argmin(
                    best_res_overall.F[:, 0] + best_res_overall.F[:, 1] * 1e6
                )
            )

            _, _, b_load, _ = problem.simulate_solution(best_res_overall.X[best_idx])
            plt.figure(figsize=(10, 6))
            net_load = b_load + case_data["p_base"] - case_data["p_pv"]
            plt.plot(net_load, color="navy", label="Net Depot Load")
            plt.fill_between(range(96), net_load, color="navy", alpha=0.1)
            plt.axhline(
                y=case_data["p_site_max"], color="red", linestyle="--", label="Limit"
            )
            plt.title(f"Load Profile: {case_id} (Lowest Cost @ Zero Shortfall)")
            plt.xlabel("Time Slot")
            plt.ylabel("Power (kW)")
            plt.legend()
            plt.grid(True, alpha=0.2)
            plt.savefig(os.path.join(case_dir, "depot_load_profile.png"), dpi=300)
            plt.close()


if __name__ == "__main__":
    run_comparison_sim()
