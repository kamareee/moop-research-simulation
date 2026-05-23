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
from pymoo.decomposition.asf import ASF  # For better knee-point selection


# --- 1. DATA GENERATION & CACHING ---
def get_advanced_case_data(N, F_count, M, alpha):
    T, dt = 96, 0.25
    # Site limit scaling
    p_site_max = (M / 15) * 200.0
    pi_grid = np.zeros(T)
    # AU Commercial TOU
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
                "pi_pub": {"L2": 0.25, "DC": 0.60},
                "power": {"L2": 7.0, "DC": 50.0},
                "wait_profile": 10 + 15 * np.sin(np.pi * t / 96) ** 2,
            }
        )

    fleet = []
    total_trip_demand = 0
    for i in range(N):
        shift_offset = random.randint(-6, 6)
        dwells = [
            [max(0, 0), min(96, 28 + shift_offset)],
            [max(0, 42 + shift_offset), min(96, 68 + shift_offset)],
            [max(0, 82 + shift_offset), 96],
        ]

        bat_max = 75.0
        # Energy intensity alpha [cite: 204]
        trip_total = bat_max * (alpha + random.uniform(-0.05, 0.05))
        trips = [trip_total * 0.45, trip_total * 0.55]
        total_trip_demand += sum(trips)

        fleet.append(
            {
                "bat_max": bat_max,
                "soc_init": 0.20,  # Initial SoC [cite: 53]
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
        "total_trip_demand": total_trip_demand,
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
        # Objectives: f1 (Cost), f2 (Shortfall) [cite: 76, 90]
        super().__init__(
            n_var=n_windows, n_obj=2, n_constr=2, xl=0, xu=1 + 2 * self.d["F"]
        )

    def _evaluate(self, x, out, *args, **kwargs):
        res = self.simulate(x)
        p_viol = np.sum(
            np.maximum(
                0,
                res["load"] + self.d["p_base"] - self.d["p_pv"] - self.d["p_site_max"],
            )
        )
        e_viol = np.sum(np.maximum(0, res["evses"] - self.d["evse_m"]))
        out["F"] = [res["cost"], res["shortfall"]]
        out["G"] = [p_viol, e_viol]

    def simulate(self, x):
        x = np.round(x).astype(int)
        d = self.d
        soc = np.array([v["soc_init"] * v["bat_max"] for v in d["fleet"]])
        cost, total_shortfall = 0, 0
        load, evses = np.zeros(d["T"]), np.zeros(d["T"])
        # Track energy utilized for trips
        utilized_depot, utilized_public = 0, 0

        idx = 0
        for i, v in enumerate(d["fleet"]):
            for j, dwell in enumerate(v["dwells"]):
                choice, start, end = x[idx], dwell[0], dwell[1]
                idx += 1

                # Charge vehicle before trip
                energy_gained = 0
                if choice == 1:  # Depot
                    p = min(22.0, (v["bat_max"] - soc[i]) / d["dt"])
                    energy_gained = p * d["dt"] * 0.9
                    cost += np.sum(d["pi_grid"][start:end] * p * d["dt"])
                    load[start:end] += p
                    evses[start:end] += 1
                    utilized_depot += energy_gained
                elif choice > 1:  # Public
                    f_idx = (choice - 2) % d["F"]
                    c_type = "L2" if choice < 2 + d["F"] else "DC"
                    st = d["stations"][f_idx]
                    net_min = (
                        ((end - start) * d["dt"] * 60)
                        - v["detour"][f_idx]
                        - st["wait_profile"][start]
                    )
                    if net_min > 0:
                        energy_gained = (
                            min(
                                st["power"][c_type] * (net_min / 60),
                                v["bat_max"] - soc[i],
                            )
                            * 0.95
                        )
                        cost += (energy_gained / 0.95) * st["pi_pub"][c_type] + d[
                            "c_d_t"
                        ] * (v["detour"][f_idx] + st["wait_profile"][start])
                        utilized_public += energy_gained

                soc[i] = min(v["bat_max"], soc[i] + energy_gained)

                # Execute trip if window concludes a cycle
                if j < len(v["e_trip"]):
                    trip_e = v["e_trip"][j]
                    if soc[i] < trip_e:
                        total_shortfall += trip_e - soc[i]
                        soc[i] = 0
                    else:
                        soc[i] -= trip_e

        return {
            "cost": cost,
            "shortfall": total_shortfall,
            "load": load,
            "evses": evses,
            "e_depot": utilized_depot,
            "e_pub": utilized_public,
        }


# --- 3. MASTER SIMULATION LOOP ---
def run_comparison_sim():
    scenarios = [[45, 10, 15], [60, 10, 20], [90, 20, 25]]
    alpha = 0.75
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plt.rcParams.update({"font.size": 7, "font.family": "serif"})

    for N, F, M in scenarios:
        case_id = f"N{N}_F{F}_M{M}"
        case_dir = os.path.join(base_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)

        data = load_or_generate_case_data(case_dir, N, F, M, alpha)
        problem = ChargingProblem(data)

        # Scaling population/generations for N=90
        pop_size = 250 if N == 90 else 150
        n_gen = 300 if N == 90 else 150

        print(f"Optimizing {case_id}...")
        res = minimize(
            problem, NSGA2(pop_size=pop_size), ("n_gen", n_gen), seed=1, verbose=False
        )

        # 1. Pareto Plot
        fig, ax = plt.subplots(figsize=(3.5, 3))
        ax.scatter(res.F[:, 0], res.F[:, 1], s=10, color="purple", label="Pareto Front")
        ax.set_title(f"Pareto Front: {case_id}")
        ax.set_xlabel("Cost ($)")
        ax.set_ylabel("Shortfall (kWh)")
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        fig.savefig(os.path.join(case_dir, "pareto.png"), dpi=300)

        # 2. Knee Point Selection via ASF (Avoids empty masks)
        weights = np.array([0.5, 0.5])
        decomp = ASF()
        # Normalize objectives for selection
        norm_F = (res.F - res.F.min(axis=0)) / (
            res.F.max(axis=0) - res.F.min(axis=0) + 1e-6
        )
        best_idx = decomp.do(norm_F, 1 / weights).argmin()

        sim_res = problem.simulate(res.X[best_idx])

        # 3. Fulfillment Chart
        fig_f, ax_f = plt.subplots(figsize=(3.5, 2.5))
        d_pct = (sim_res["e_depot"] / data["total_trip_demand"]) * 100
        p_pct = (sim_res["e_pub"] / data["total_trip_demand"]) * 100
        # Cap at 100% logic [cite: 173]
        shortfall_pct = (sim_res["shortfall"] / data["total_trip_demand"]) * 100

        ax_f.bar(0, d_pct, color="#0077b6", label="Depot")
        ax_f.bar(0, p_pct, bottom=d_pct, color="#00b4d8", label="Public")

        total_fulfilled = min(100, d_pct + p_pct)
        ax_f.text(
            0,
            total_fulfilled / 2,
            f"{int(total_fulfilled)}%",
            ha="center",
            color="white",
            fontweight="bold",
        )
        ax_f.set_ylabel("Trip Demand Fulfilled (%)")
        ax_f.set_ylim(0, 110)
        ax_f.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=2)
        fig_f.tight_layout()
        fig_f.savefig(os.path.join(case_dir, "fulfillment.png"), dpi=300)
        plt.close("all")


if __name__ == "__main__":
    run_comparison_sim()
