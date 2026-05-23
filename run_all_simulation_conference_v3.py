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
    T, dt = 96, 0.25
    p_site_max = (M / 15) * 200.0

    # Depot uses constant 0.30; Public L2 uses TOU
    pi_depot = np.full(T, 0.30)
    pi_tou = np.zeros(T)
    pi_tou[0:28], pi_tou[28:60], pi_tou[60:84], pi_tou[84:88], pi_tou[88:96] = (
        0.1781,
        0.2087,
        0.3121,
        0.2087,
        0.1781,
    )

    t = np.arange(T)
    p_pv = (p_site_max * 0.15) * np.maximum(0, np.sin(np.pi * (t - 24) / 48))
    p_base = (p_site_max * 0.125) + 10 * np.random.rand(T)

    pub_availability = np.ones((T, F_count))
    for f in range(F_count):
        offline_slots = np.random.choice(T, size=int(T * 0.2), replace=False)
        pub_availability[offline_slots, f] = 0

    stations = []
    for f in range(F_count):
        stations.append(
            {
                "pi_pub": {
                    "L2": pi_tou,
                    "DC": round(random.uniform(0.40, 0.50), 3),  # DC range [0.40-0.50]
                },
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
                "soc_init": 0.30,
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
        "pi_grid": pi_tou,
        "p_pv": p_pv,
        "p_base": p_base,
        "stations": stations,
        "fleet": fleet,
        "p_site_max": p_site_max,
        "evse_m": M,
        "c_d_t": 0.40,
        "pi_demand": 0.0,
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


class ChargingProblem(ElementwiseProblem):
    def __init__(self, data):
        self.d = data
        n_vars = sum(len(v["dwells"]) for v in self.d["fleet"])
        super().__init__(
            n_var=n_vars, n_obj=2, n_constr=2, xl=0, xu=1 + 2 * self.d["F"]
        )

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
        e_depot_utilized, e_pub_L2, e_pub_DC = 0, 0, 0

        # Step 1: Pre-calculate decisions to evaluate depot occupancy for prioritization
        idx = 0
        for i, v in enumerate(d["fleet"]):
            for j, dwell in enumerate(v["dwells"]):
                choice, start, end = x[idx], int(dwell[0]), int(dwell[1])
                idx += 1

                energy_gained = 0
                if choice == 1:  # Depot
                    p = min(22.0, (v["bat_max"] - soc[i]) / d["dt"])
                    energy_gained = p * d["dt"] * 0.9
                    cost += np.sum(d["pi_grid"][start:end] * p * d["dt"])
                    load[start:end] += p
                    evses[start:end] += 1
                    e_depot_utilized += energy_gained
                elif choice > 1:  # Public
                    f_idx = (choice - 2) % d["F"]
                    c_type = "L2" if choice < 2 + d["F"] else "DC"
                    st = d["stations"][f_idx]

                    # Force Priority Penalty: If depot EVSEs are free, public charging is penalized
                    idle_evses = d["evse_m"] - evses[start]
                    if idle_evses > 0:
                        cost += 5.0  # Preference penalty ($) for ignoring available depot capacity

                    is_available = d["pub_availability"][start, f_idx]
                    wait = st["wait_profile"][start]
                    net_min = ((end - start) * d["dt"] * 60) - v["detour"][f_idx] - wait
                    cost += d["c_d_t"] * (v["detour"][f_idx] + wait)

                    if net_min > 0 and is_available:
                        energy_gained = (
                            min(
                                st["power"][c_type] * (net_min / 60),
                                v["bat_max"] - soc[i],
                            )
                            * 0.95
                        )
                        unit_price = (
                            st["pi_pub"]["L2"][start]
                            if c_type == "L2"
                            else st["pi_pub"]["DC"]
                        )
                        cost += (energy_gained / 0.95) * unit_price
                        if c_type == "L2":
                            e_pub_L2 += energy_gained
                        else:
                            e_pub_DC += energy_gained

                soc[i] = min(v["bat_max"], soc[i] + energy_gained)
                if j < len(v["e_trip"]):
                    trip_e = v["e_trip"][j]
                    if soc[i] < trip_e:
                        total_shortfall += trip_e - soc[i]
                        soc[i] = 0
                    else:
                        soc[i] -= trip_e

        net_load = load + d["p_base"] - d["p_pv"]
        cost += np.maximum(0, net_load).max() * d["pi_demand"]

        return {
            "cost": cost,
            "shortfall": total_shortfall,
            "net_load": net_load,
            "evses": evses,
            "e_depot": e_depot_utilized,
            "e_L2": e_pub_L2,
            "e_DC": e_pub_DC,
        }


def run_comparison_sim():
    scenarios = [[45, 10, 15], [60, 10, 20], [90, 20, 25]]
    pop_sizes = [150, 200, 250]
    alpha = 0.70
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
                    "DC": (sim_res["e_DC"] / data["total_trip_demand"]) * 100,
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

        fig_f, ax_f = plt.subplots(figsize=(4.0, 3.0))
        x_idx = np.arange(len(pop_sizes))
        d_vals = [s["depot"] for s in pop_fulfillment_stats]
        l2_vals = [s["L2"] for s in pop_fulfillment_stats]
        dc_vals = [s["DC"] for s in pop_fulfillment_stats]

        ax_f.bar(x_idx, d_vals, color="#0077b6", label="Depot")
        ax_f.bar(x_idx, l2_vals, bottom=d_vals, color="#00b4d8", label="Public L2")
        ax_f.bar(
            x_idx,
            dc_vals,
            bottom=np.array(d_vals) + np.array(l2_vals),
            color="#90e0ef",
            label="Public DC",
        )

        for idx, s in enumerate(pop_fulfillment_stats):
            ax_f.text(
                idx,
                s["total_met"] / 2,
                f"{int(s['total_met'])}%",
                ha="center",
                color="black",
                fontweight="bold",
                fontsize=6,
            )

        ax_f.set_xticks(x_idx)
        ax_f.set_xticklabels([f"P:{p}" for p in pop_sizes])
        ax_f.set_ylabel("Trip Demand Fulfilled (%)")
        ax_f.set_ylim(0, 110)
        ax_f.set_title(f"Fulfillment Comparison: {case_id}")
        ax_f.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
        fig_f.tight_layout()
        fig_f.savefig(os.path.join(case_dir, "fulfillment.png"), dpi=300)
        plt.close("all")


if __name__ == "__main__":
    run_comparison_sim()
