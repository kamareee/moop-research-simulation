# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring, invalid-name

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


# --- 1. DATA GENERATION (Returns Dictionary Directly) ---
def get_case_data(N, F_count, M, p_site_max):
    """Generates synthetic data and returns the dictionary in memory."""
    T, dt = 96, 0.25
    pi_grid = np.zeros(T)

    # Australian Commercial TOU Tariff logic
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
        fleet.append(
            {
                "bat_max": 75.0,
                "soc_init": 0.20,
                "e_trip": [35.0, 35.0],  # 70kWh demand
                "dwells": [[0, 30], [42, 70], [82, 96]],
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


def load_or_generate_case_data(case_dir, N, F_count, M, p_site_max):
    data_path = os.path.join(case_dir, "sim_data.pkl")
    required_keys = {
        "N",
        "T",
        "F",
        "dt",
        "pi_grid",
        "p_pv",
        "p_base",
        "stations",
        "fleet",
        "p_site_max",
        "evse_m",
        "c_d_t",
    }

    case_data = None
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            case_data = pickle.load(f)

    if (
        case_data is None
        or not isinstance(case_data, dict)
        or not required_keys.issubset(case_data.keys())
    ):
        case_data = get_case_data(N, F_count, M, p_site_max)
        with open(data_path, "wb") as f:
            pickle.dump(case_data, f)

    return case_data


# --- 2. PROBLEM DEFINITION ---
class ChargingProblem(ElementwiseProblem):
    def __init__(self, problem_data):
        # Renamed from self.data to self.case_data to avoid pymoo overwrite
        self.case_data = problem_data
        n_windows = sum(len(v["dwells"]) for v in self.case_data["fleet"])

        # n_constr changed to 2 to accommodate the EVSE count limit (M)
        super().__init__(
            n_var=n_windows, n_obj=2, n_constr=2, xl=0, xu=1 + 2 * self.case_data["F"]
        )

    def _evaluate(self, x, out, *args, **kwargs):
        total_cost, total_shortfall, depot_load, active_evses = self.simulate_solution(
            x
        )

        # Constraint 1: Site power import limit violation (C5)
        power_violation = np.sum(
            np.maximum(
                0,
                depot_load
                + self.case_data["p_base"]
                - self.case_data["p_pv"]
                - self.case_data["p_site_max"],
            )
        )

        # Constraint 2: EVSE count limit violation (C6)
        evse_violation = np.sum(np.maximum(0, active_evses - self.case_data["evse_m"]))

        out["F"] = [total_cost, total_shortfall]
        out["G"] = [power_violation, evse_violation]

    def simulate_solution(self, x):
        x = np.round(x).astype(int)
        d = self.case_data
        soc = np.array([v["soc_init"] * v["bat_max"] for v in d["fleet"]])

        total_cost = 0.0
        total_shortfall = 0.0
        depot_load = np.zeros(d["T"])
        active_evses = np.zeros(d["T"])

        idx = 0
        for i, v in enumerate(d["fleet"]):
            for dwell in v["dwells"]:
                choice, start, end = x[idx], dwell[0], dwell[1]
                idx += 1

                if choice == 1:  # Depot Assignment
                    # 3-Phase 22kW charging logic
                    p = min(22.0, (v["bat_max"] - soc[i]) / d["dt"])
                    soc[i] += p * d["dt"] * 0.9
                    total_cost += np.sum(d["pi_grid"][start:end] * p * d["dt"])
                    depot_load[start:end] += p
                    active_evses[start:end] += 1  # Track concurrent charging sessions

                elif choice > 1:  # Public Assignment
                    f_idx = (choice - 2) % d["F"]
                    c_type = "L2" if choice < 2 + d["F"] else "DC"
                    st = d["stations"][f_idx]
                    wait = st["wait_profile"][start]
                    net = ((end - start) * d["dt"] * 60) - v["detour"][f_idx] - wait

                    if net > 0:
                        e = min(st["power"][c_type] * (net / 60), v["bat_max"] - soc[i])
                        soc[i] += e * 0.95
                        total_cost += e * st["pi_pub"][c_type] + d["c_d_t"] * (
                            v["detour"][f_idx] + wait
                        )

            # Energy Shortfall (Reliability) tracking
            for trip_e in v["e_trip"]:
                if soc[i] < trip_e:
                    total_shortfall += trip_e - soc[i]
                    soc[i] = 0
                else:
                    soc[i] -= trip_e

        return total_cost, total_shortfall, depot_load, active_evses


# --- 3. MASTER SIMULATION ---
def run_comparison_sim():
    # Case Matrix: [Fleet Size N, Public Stations F, Depot Chargers M]
    scenarios = [[45, 10, 15], [60, 10, 20], [90, 10, 30], [90, 15, 30]]
    pop_sizes = [100, 120, 150]

    base_dir = os.path.dirname(os.path.abspath(__file__))

    for N, F, M in scenarios:
        case_id = f"N{N}_F{F}_M{M}"
        case_dir = os.path.join(base_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)
        print(f"\n--- Running Scenario: {case_id} ---")

        # Scaling P_site_max based on 3-phase 22kW chargers (15 chargers = 200kW)
        p_site_limit = (M / 15) * 200.0

        # Load cached data if valid; regenerate if required keys are missing
        case_data = load_or_generate_case_data(case_dir, N, F, M, p_site_limit)
        problem = ChargingProblem(case_data)

        plt.figure(figsize=(10, 7))
        cmap = plt.get_cmap("plasma")
        colors = cmap(np.linspace(0, 1, len(pop_sizes)))

        best_res_overall = None

        for i, pop in enumerate(pop_sizes):
            print(f"  Optimizing with Pop Size: {pop}...")
            res = minimize(problem, NSGA2(pop_size=pop), ("n_gen", 100), seed=1)

            # Keep the result from the largest population for the Load Profile
            if pop == 150 or best_res_overall is None:
                best_res_overall = res

            F_res = res.F
            if F_res is not None:
                if F_res.ndim == 1:
                    F_res = np.array([F_res])
                idx = np.argsort(F_res[:, 0])
                plt.plot(
                    F_res[idx, 0],
                    F_res[idx, 1],
                    "o--",
                    color=colors[i],
                    label=f"Pop Size: {pop}",
                    alpha=0.8,
                )

        # 1. Pareto Front Comparison Graph
        plt.title(f"Pareto Front Comparison: {case_id}\n(N={N}, F={F}, M={M})")
        plt.xlabel("Total Cost ($)")
        plt.ylabel("Energy Shortfall (kWh)")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.savefig(os.path.join(case_dir, "pareto_comparison.png"), dpi=300)
        plt.close()

        # 2. Depot Load Profile (Extract from best result: Pop=250)
        if best_res_overall is not None and best_res_overall.F is not None:
            F_res = best_res_overall.F
            if F_res.ndim == 1:
                F_res = np.array([F_res])
                X_res = np.array([best_res_overall.X])
            else:
                X_res = best_res_overall.X

            # FIX: Find the lowest cost strictly where shortfall is zero
            zero_shortfall_mask = F_res[:, 1] <= 1e-5

            if np.any(zero_shortfall_mask):
                valid_costs = F_res[zero_shortfall_mask, 0]
                best_valid_idx = np.argmin(valid_costs)
                best_idx = np.where(zero_shortfall_mask)[0][best_valid_idx]
            else:
                # Fallback to knee point if no zero-shortfall solution was found
                best_idx = np.argmin(F_res[:, 0] + F_res[:, 1] * 1e6)

            _, _, best_load, _ = problem.simulate_solution(X_res[best_idx])

            plt.figure(figsize=(10, 6))
            net_load = best_load + case_data["p_base"] - case_data["p_pv"]
            plt.plot(net_load, label="Net Depot Load", color="navy", linewidth=2)
            plt.fill_between(range(96), net_load, color="navy", alpha=0.1)
            plt.axhline(
                y=case_data["p_site_max"],
                color="red",
                linestyle="--",
                label="Site Power Limit",
            )
            plt.title(f"Depot Load Profile: {case_id}\n(Optimal Solution from Pop=250)")
            plt.xlabel("Time Slot (15-min)")
            plt.ylabel("Power (kW)")
            plt.legend()
            plt.grid(True, alpha=0.2)
            plt.savefig(os.path.join(case_dir, "depot_load_profile.png"), dpi=300)
            plt.close()


if __name__ == "__main__":
    run_comparison_sim()
