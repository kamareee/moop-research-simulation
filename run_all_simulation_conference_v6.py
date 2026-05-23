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
    Generates data for Depot and Public L2 only.
    Maintains high-friction L2 blackouts to ensure depot priority.
    """
    T, dt = 96, 0.25
    p_site_max = (M / 15) * 200.0

    # Depot: Constant 0.30/kWh
    pi_depot = np.full(T, 0.30)

    # Public L2: TOU Profile + 0.20 Premium
    pi_tou_base = np.zeros(T)
    (
        pi_tou_base[0:28],
        pi_tou_base[28:60],
        pi_tou_base[60:84],
        pi_tou_base[84:88],
        pi_tou_base[88:96],
    ) = (
        0.1781,
        0.2087,
        0.3121,
        0.2087,
        0.1781,
    )
    pi_l2_premium = pi_tou_base + 0.20

    # Availability Mask: Strategic L2 Blackouts
    pub_availability = np.ones((T, F_count))
    for f in range(F_count):
        # L2 offline during primary depot dwell periods (0-30 and 70-96)
        pub_availability[0:30, f] = 0
        pub_availability[70:96, f] = 0
        # Random downtime for remaining slots
        offline_slots = np.random.choice(range(30, 70), size=5, replace=False)
        pub_availability[offline_slots, f] = 0

    stations = []
    for f in range(F_count):
        stations.append(
            {
                "pi_pub": pi_l2_premium,
                "power": 11.0,
                "wait_profile": 10 + 15 * np.sin(np.pi * np.arange(T) / 96) ** 2,
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

        detours = np.random.randint(10, 20, size=F_count)

        fleet.append(
            {
                "bat_max": bat_max,
                "soc_init": 0.30,
                "e_trip": trips,
                "dwells": dwells,
                "detour": detours,
            }
        )

    return {
        "N": N,
        "T": T,
        "F": F_count,
        "dt": dt,
        "pi_grid": pi_depot,
        "p_site_max": p_site_max,
        "evse_m": M,
        "c_d_t": 0.35,
        "pi_demand": 0.0,
        "total_trip_demand": total_trip_demand,
        "pub_availability": pub_availability,
        "stations": stations,
        "fleet": fleet,
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
        # Upper bound restricted to 1 + F since DC chargers are removed
        super().__init__(n_var=n_vars, n_obj=2, n_constr=2, xl=0, xu=1 + self.d["F"])

    def _evaluate(self, x, out, *args, **kwargs):
        res = self.simulate(x)
        p_viol = np.sum(np.maximum(0, res["net_load"] - self.d["p_site_max"]))
        e_viol = np.sum(np.maximum(0, res["evses"] - self.d["evse_m"]))
        out["F"] = [res["cost"], res["shortfall"]]
        out["G"] = [p_viol, e_viol]

    def simulate(self, x):
        x = np.round(x).astype(int)
        d = self.d
        soc = np.array([v["soc_init"] * v["bat_max"] for v in d["fleet"]])
        cost, total_shortfall = 0, 0
        load, evses = np.zeros(d["T"]), np.zeros(d["T"])
        e_depot, e_L2 = 0, 0

        idx = 0
        for i, v in enumerate(d["fleet"]):
            for j, dwell in enumerate(v["dwells"]):
                choice, start, end = x[idx], int(dwell[0]), int(dwell[1])
                idx += 1

                energy_gained = 0
                if choice == 1:  # Depot Assignment
                    total_dwell_energy = 0
                    # Dynamic slot-by-slot tracking to break early when full
                    for t in range(start, end):
                        if soc[i] >= v["bat_max"]:
                            break

                        p = min(22.0, (v["bat_max"] - soc[i]) / (d["dt"] * 0.9))
                        slot_energy = p * d["dt"] * 0.9

                        cost += d["pi_grid"][t] * p * d["dt"]
                        load[t] += p
                        evses[t] += 1

                        soc[i] += slot_energy
                        total_dwell_energy += slot_energy

                    e_depot += total_dwell_energy
                    energy_gained = 0

                elif choice > 1:  # Public L2 Assignment
                    f_idx = choice - 2
                    st = d["stations"][f_idx]

                    is_avail = d["pub_availability"][start, f_idx]
                    wait = st["wait_profile"][start]
                    net_min = ((end - start) * d["dt"] * 60) - v["detour"][f_idx] - wait
                    cost += d["c_d_t"] * (v["detour"][f_idx] + wait)

                    if net_min > 0 and is_avail:
                        energy_gained = (
                            min(st["power"] * (net_min / 60), v["bat_max"] - soc[i])
                            * 0.95
                        )
                        cost += (energy_gained / 0.95) * st["pi_pub"][start]
                        e_L2 += energy_gained

                soc[i] = min(v["bat_max"], soc[i] + energy_gained)
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
            "net_load": load,
            "evses": evses,
            "e_depot": e_depot,
            "e_L2": e_L2,
        }


# --- 3. MASTER SIMULATION LOOP ---
def run_comparison_sim():
    scenarios = [[45, 15, 15], [60, 25, 20], [90, 30, 25]]
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
        pop_fulfillment_stats = []

        # 1. Pareto Front Comparison Graph
        fig_p, ax_p = plt.subplots(figsize=(3.5, 3.0))
        colors = plt.get_cmap("viridis")(np.linspace(0, 0.8, len(pop_sizes)))

        for i, pop in enumerate(pop_sizes):
            print(f"  Optimizing Population: {pop}...")
            res = minimize(
                problem, NSGA2(pop_size=pop), ("n_gen", 300 if N == 90 else 150), seed=1
            )

            f_sort = res.F[np.argsort(res.F[:, 0])]
            ax_p.plot(
                f_sort[:, 0],
                f_sort[:, 1],
                "o-",
                markersize=3,
                color=colors[i],
                label=f"P:{pop}",
            )

            # 2. Extract Knee Point for THIS population
            decomp = ASF()
            norm_F = (res.F - res.F.min(axis=0)) / (
                res.F.max(axis=0) - res.F.min(axis=0) + 1e-6
            )
            idx = decomp.do(norm_F, 1 / np.array([0.5, 0.5])).argmin()
            sim_res = problem.simulate(res.X[idx])

            pop_fulfillment_stats.append(
                {
                    "pop": pop,
                    "depot": (sim_res["e_depot"] / data["total_trip_demand"]) * 100,
                    "L2": (sim_res["e_L2"] / data["total_trip_demand"]) * 100,
                    "shortfall": (sim_res["shortfall"] / data["total_trip_demand"])
                    * 100,
                    "total_met": (
                        (data["total_trip_demand"] - sim_res["shortfall"])
                        / data["total_trip_demand"]
                    )
                    * 100,
                }
            )

        ax_p.set_title(f"Pareto Front: {case_id}")
        ax_p.set_xlabel("Cost ($)")
        ax_p.set_ylabel("Shortfall (kWh)")
        ax_p.legend()
        fig_p.tight_layout()
        fig_p.savefig(os.path.join(case_dir, "pareto.png"), dpi=300)

        # 3. Stacked Fulfillment Comparison (Updated to stack Shortfall)
        fig_f, ax_f = plt.subplots(figsize=(4.0, 3.0))
        x_idx = np.arange(len(pop_sizes))
        d_vals = [s["depot"] for s in pop_fulfillment_stats]
        l2_vals = [s["L2"] for s in pop_fulfillment_stats]
        sf_vals = [s["shortfall"] for s in pop_fulfillment_stats]

        # Stack layers: Depot -> Public L2 -> Shortfall to guarantee 100% boundary
        ax_f.bar(x_idx, d_vals, color="#0077b6", label="Depot")
        ax_f.bar(x_idx, l2_vals, bottom=d_vals, color="#00b4d8", label="Public L2")
        ax_f.bar(
            x_idx,
            sf_vals,
            bottom=np.array(d_vals) + np.array(l2_vals),
            color="#e63946",
            label="Shortfall",
        )

        # Label each bar with the total met percentage and shortfall percentage
        for idx, s in enumerate(pop_fulfillment_stats):
            # Total met percentage (centered in the met demand portion)
            ax_f.text(
                idx,
                s["total_met"] / 2,
                f"{int(s['total_met'])}%",
                ha="center",
                color="black",
                fontweight="bold",
                fontsize=6,
            )

            # Shortfall percentage (centered in the shortfall portion)
            if s["shortfall"] > 0:
                ax_f.text(
                    idx,
                    s["total_met"] + (s["shortfall"] / 2),
                    f"{int(s['shortfall'])}%",
                    ha="center",
                    color="black",
                    fontweight="bold",
                    fontsize=6,
                )

        ax_f.set_xticks(x_idx)
        ax_f.set_xticklabels([f"P:{p}" for p in pop_sizes])
        ax_f.set_ylabel("Trip Demand Breakdown (%)")
        ax_f.set_ylim(0, 110)
        ax_f.set_title(f"Fulfillment Comparison: {case_id}")
        ax_f.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3
        )  # Changed ncol to 3 for legibility
        fig_f.tight_layout()
        fig_f.savefig(os.path.join(case_dir, "fulfillment.png"), dpi=300)
        plt.close("all")


if __name__ == "__main__":
    run_comparison_sim()
