# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.decomposition.asf import ASF


# --- 1. DATA GENERATION & CACHING ---
def get_advanced_case_data(N, F_count, M, alpha):
    """
    Generates stochastic data including site power limits[cite: 56],
    ToU pricing[cite: 68], and the new availability mask for public stations.
    """
    T, dt = 96, 0.25
    p_site_max = (M / 15) * 200.0
    pi_grid = np.zeros(T)
    # AU Commercial TOU Tariff
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

    # Availability Mask (T x F): Simulates physical contention or downtime [cite: 144]
    pub_availability = np.ones((T, F_count))
    for f in range(F_count):
        # Randomly disable stations for 20% of the day to simulate "full" status
        offline_slots = np.random.choice(T, size=int(T * 0.2), replace=False)
        pub_availability[offline_slots, f] = 0

    stations = []
    for f in range(F_count):
        stations.append(
            {
                "pi_pub": {"L2": 0.25, "DC": 0.40},
                "power": {"L2": 11.0, "DC": 50.0},
                "wait_profile": 10 + 15 * np.sin(np.pi * t / 96) ** 2,
            }
        )

    fleet = []
    total_trip_demand = 0
    for _ in range(N):
        shift_offset = random.randint(-6, 6)
        dwells = [
            [0, 28 + shift_offset],
            [42 + shift_offset, 68 + shift_offset],
            [82 + shift_offset, 96],
        ]
        bat_max = 75.0
        trip_total = bat_max * (alpha + random.uniform(-0.05, 0.05))
        trips = [trip_total * 0.45, trip_total * 0.55]
        total_trip_demand += sum(trips)

        fleet.append(
            {
                "bat_max": bat_max,
                "soc_init": 0.40,
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
        "c_d_t": 0.50,
        "pi_demand": 0.0,  # AU Demand Charge Rate ($/kW)
        "total_trip_demand": total_trip_demand,
        "pub_availability": pub_availability,
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
        n_vars = sum(len(v["dwells"]) for v in self.d["fleet"])
        super().__init__(
            n_var=n_vars, n_obj=2, n_constr=2, xl=0, xu=1 + 2 * self.d["F"]
        )

    def _evaluate(self, x, out, *args, **kwargs):
        res = self.simulate(x)
        # G1: Site Power Limit (C5) [cite: 121]
        p_viol = np.sum(np.maximum(0, res["net_load"] - self.d["p_site_max"]))
        # G2: EVSE Count Limit (C6) [cite: 123]
        e_viol = np.sum(np.maximum(0, res["evses"] - self.d["evse_m"]))

        out["F"] = [
            res["cost"],
            res["shortfall"],
        ]  # f1: Cost, f2: Shortfall [cite: 76, 90]
        out["G"] = [p_viol, e_viol]

    def simulate(self, x):
        x = np.round(x).astype(int)
        d = self.d
        soc = np.array([v["soc_init"] * v["bat_max"] for v in d["fleet"]])
        cost, total_shortfall = 0, 0
        load, evses = np.zeros(d["T"]), np.zeros(d["T"])
        e_depot_utilized, e_pub_utilized = 0, 0

        idx = 0
        for i, v in enumerate(d["fleet"]):
            for j, dwell in enumerate(v["dwells"]):
                choice, start, end = x[idx], int(dwell[0]), int(dwell[1])
                idx += 1

                energy_gained = 0
                if choice == 1:  # Depot [cite: 112, 149]
                    p = min(22.0, (v["bat_max"] - soc[i]) / d["dt"])
                    energy_gained = p * d["dt"] * 0.9
                    cost += np.sum(d["pi_grid"][start:end] * p * d["dt"])
                    load[start:end] += p
                    evses[start:end] += 1
                    e_depot_utilized += energy_gained
                elif choice > 1:  # Public [cite: 126, 155]
                    f_idx = (choice - 2) % d["F"]
                    c_type = "L2" if choice < 2 + d["F"] else "DC"
                    st = d["stations"][f_idx]

                    # Logic: If station is unavailable at the start of the dwell, energy = 0
                    is_available = d["pub_availability"][start, f_idx]
                    wait = st["wait_profile"][start]
                    net_min = ((end - start) * d["dt"] * 60) - v["detour"][f_idx] - wait

                    # Detour/Wait costs are paid regardless of energy gain [cite: 86]
                    cost += d["c_d_t"] * (v["detour"][f_idx] + wait)

                    if net_min > 0 and is_available:
                        energy_gained = (
                            min(
                                st["power"][c_type] * (net_min / 60),
                                v["bat_max"] - soc[i],
                            )
                            * 0.95
                        )
                        cost += (energy_gained / 0.95) * st["pi_pub"][c_type]
                        e_pub_utilized += energy_gained

                soc[i] = min(v["bat_max"], soc[i] + energy_gained)

                # Trip Energy Consumption & Shortfall Calculation (C14) [cite: 167]
                if j < len(v["e_trip"]):
                    trip_e = v["e_trip"][j]
                    if soc[i] < trip_e:
                        total_shortfall += trip_e - soc[i]
                        soc[i] = 0
                    else:
                        soc[i] -= trip_e

        # Demand Charge Tracking [cite: 89]
        net_load = load + d["p_base"] - d["p_pv"]
        p_peak = np.maximum(0, net_load).max()
        cost += p_peak * d["pi_demand"]

        return {
            "cost": cost,
            "shortfall": total_shortfall,
            "net_load": net_load,
            "evses": evses,
            "e_depot": e_depot_utilized,
            "e_pub": e_pub_utilized,
        }


# --- 3. MASTER SIMULATION LOOP ---
def run_comparison_sim():
    scenarios = [[45, 10, 15], [60, 10, 20], [90, 20, 25]]
    pop_sizes = [150, 200, 250]
    alpha = 0.75
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plt.rcParams.update({"font.size": 7, "font.family": "serif"})

    for N, F, M in scenarios:
        case_id = f"N{N}_F{F}_M{M}"
        case_dir = os.path.join(base_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)
        print(f"\n--- Scenario: {case_id} ---")

        data = load_or_generate_case_data(case_dir, N, F, M, alpha)
        problem = ChargingProblem(data)

        # 1. Pareto Front Comparison Graph
        fig_p, ax_p = plt.subplots(figsize=(3.5, 3.0))
        colors = plt.get_cmap("viridis")(np.linspace(0, 0.8, len(pop_sizes)))
        best_res = None

        for i, pop in enumerate(pop_sizes):
            print(f"  Optimizing Population: {pop}...")
            # Increased generations for N=90 complexity
            n_gens = 300 if N == 90 else 150
            res = minimize(problem, NSGA2(pop_size=pop), ("n_gen", n_gens), seed=1)

            if pop == max(pop_sizes):
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
        ax_p.legend()
        ax_p.grid(True, alpha=0.2)
        fig_p.tight_layout()
        fig_p.savefig(os.path.join(case_dir, "pareto.png"), dpi=300)

        # 2. Knee Point Extraction (using ASF for stability)
        decomp = ASF()
        norm_F = (best_res.F - best_res.F.min(axis=0)) / (
            best_res.F.max(axis=0) - best_res.F.min(axis=0) + 1e-6
        )
        idx = decomp.do(norm_F, 1 / np.array([0.5, 0.5])).argmin()
        sim_res = problem.simulate(best_res.X[idx])

        # 3. Fulfillment Chart (Strictly Demand-Based)
        fig_f, ax_f = plt.subplots(figsize=(3.5, 2.5))
        d_pct = (sim_res["e_depot"] / data["total_trip_demand"]) * 100
        p_pct = (sim_res["e_pub"] / data["total_trip_demand"]) * 100

        # We cap visual fulfillment at 100% relative to demand met [cite: 173]
        total_met = (
            (data["total_trip_demand"] - sim_res["shortfall"])
            / data["total_trip_demand"]
        ) * 100

        ax_f.bar(0, d_pct, color="#0077b6", label="Depot")
        ax_f.bar(0, p_pct, bottom=d_pct, color="#00b4d8", label="Public")
        ax_f.text(
            0,
            total_met / 2,
            f"{int(total_met)}%",
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
