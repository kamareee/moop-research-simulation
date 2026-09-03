# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.decomposition.asf import ASF
from pymoo.indicators.hv import Hypervolume
import csv


# --- 1. DATA GENERATION & CACHING ---
def get_advanced_case_data(N, F_count, M, alpha):
    """
    Generates data for Depot and Public L2 only.
    Maintains high-friction L2 blackouts to ensure depot priority.

    n_conn is recorded per station for reporting-only connector
    over-subscription (NOT a hard constraint — see ChargingProblem).
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
                "n_conn": 2,  # reporting-only (Uncoord infeasibility label)
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
        # Upper bound restricted to 1 + F since DC chargers are removed.
        # n_constr = 2: site power + EVSE count (NO connector capacity —
        # connector over-subscription is tracked for reporting only).
        super().__init__(n_var=n_vars, n_obj=2, n_constr=2, xl=0, xu=1 + self.d["F"])

    def _evaluate(self, x, out, *args, **kwargs):
        res = self.simulate(x)
        p_viol = np.sum(np.maximum(0, res["net_load"] - self.d["p_site_max"]))
        e_viol = np.sum(np.maximum(0, res["evses"] - self.d["evse_m"]))
        out["F"] = [res["cost"], res["shortfall"]]
        out["G"] = [p_viol, e_viol]

    def simulate(self, x):
        """Forward-simulate one chromosome.

        Source accounting: every kWh in a vehicle's battery is tagged as
        either depot-origin or public-origin. Initial SoC is tagged as
        depot-origin (the fleet was charged overnight at the depot before
        the horizon). Invariant: b_dep[i] + b_pub[i] == soc[i] at all times.
        e_dep / e_pub count *trip-delivered* energy from each source, so
        e_dep + e_pub + shortfall == total trip demand and the fulfilment
        bars sum to 100%. (e_depot / e_L2 still report charged-in energy
        for backward compatibility / debugging.)

        conn_occ[f, t] tracks public connector occupancy for reporting-only
        over-subscription checks (used to label Uncoord infeasible).
        """
        x = np.round(x).astype(int)
        d = self.d
        soc = np.array([v["soc_init"] * v["bat_max"] for v in d["fleet"]])
        # Source buckets: initial SoC tagged as depot-origin
        b_dep = soc.copy()
        b_pub = np.zeros(d["N"])

        cost, total_shortfall = 0, 0
        load, evses = np.zeros(d["T"]), np.zeros(d["T"])
        conn_occ = np.zeros((d["F"], d["T"]))
        # Attempted occupancy: what the schedule *would* claim regardless of
        # whether the station is available at that time. Used for plotting to
        # expose availability violations by the uncoordinated baseline.
        conn_occ_attempted = np.zeros((d["F"], d["T"]))
        e_depot, e_L2 = 0, 0  # charged-in energy (debug)
        e_dep_trips, e_pub_trips = 0, 0  # trip-delivered energy by source

        idx = 0
        for i, v in enumerate(d["fleet"]):
            for j, dwell in enumerate(v["dwells"]):
                choice, start, end = x[idx], int(dwell[0]), int(dwell[1])
                idx += 1

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
                    b_dep[i] += total_dwell_energy

                elif choice > 1:  # Public L2 Assignment
                    f_idx = choice - 2
                    st = d["stations"][f_idx]

                    is_avail = d["pub_availability"][start, f_idx]
                    wait = st["wait_profile"][start]
                    net_min = ((end - start) * d["dt"] * 60) - v["detour"][f_idx] - wait
                    cost += d["c_d_t"] * (v["detour"][f_idx] + wait)

                    # Attempted-occupancy footprint (used for reporting; recorded
                    # whether or not the station is actually available).
                    slot_min = d["dt"] * 60
                    occ_start_att = min(
                        d["T"], start + int(np.ceil(v["detour"][f_idx] / slot_min))
                    )
                    if net_min > 0:
                        occ_end_att = min(
                            d["T"], occ_start_att + int(np.ceil(net_min / slot_min))
                        )
                        conn_occ_attempted[f_idx, occ_start_att:occ_end_att] += 1

                    if net_min > 0 and is_avail:
                        energy_gained = (
                            min(st["power"] * (net_min / 60), v["bat_max"] - soc[i])
                            * 0.95
                        )
                        cost += (energy_gained / 0.95) * st["pi_pub"][start]
                        e_L2 += energy_gained
                        soc[i] = min(v["bat_max"], soc[i] + energy_gained)
                        b_pub[i] += energy_gained

                        # Track connector occupancy (reporting-only)
                        occ_end = min(
                            d["T"], occ_start_att + int(np.ceil(net_min / slot_min))
                        )
                        conn_occ[f_idx, occ_start_att:occ_end] += 1

                # Trip after this dwell (if any)
                if j < len(v["e_trip"]):
                    trip_e = v["e_trip"][j]
                    if soc[i] >= trip_e:
                        # All trip energy is tagged (b_dep + b_pub == soc)
                        share_dep = b_dep[i] / soc[i] if soc[i] > 0 else 0
                        share_pub = b_pub[i] / soc[i] if soc[i] > 0 else 0
                        e_dep_trips += trip_e * share_dep
                        e_pub_trips += trip_e * share_pub
                        b_dep[i] -= trip_e * share_dep
                        b_pub[i] -= trip_e * share_pub
                        soc[i] -= trip_e
                    else:
                        # Shortfall: deliver entire remaining SoC, all tagged
                        if soc[i] > 0:
                            share_dep = b_dep[i] / soc[i]
                            share_pub = b_pub[i] / soc[i]
                            e_dep_trips += soc[i] * share_dep
                            e_pub_trips += soc[i] * share_pub
                        total_shortfall += trip_e - soc[i]
                        soc[i] = 0
                        b_dep[i] = 0
                        b_pub[i] = 0

        return {
            "cost": cost,
            "shortfall": total_shortfall,
            "net_load": load,
            "evses": evses,
            "conn_occ": conn_occ,
            "conn_occ_attempted": conn_occ_attempted,
            "e_depot": e_depot,
            "e_L2": e_L2,
            "e_dep": e_dep_trips,
            "e_pub": e_pub_trips,
        }


# --- 3. HEURISTIC BASELINES ---
def depot_only_chrom(data):
    """Capacity-respecting depot-greedy: schedule by urgency, respect EVSE
    count M and site power. One filling pulse per scheduled dwell window.
    """
    T, M = data["T"], data["evse_m"]
    p_max = 22.0
    dt = data["dt"]
    eta = 0.9
    used = np.zeros(T, dtype=int)
    cum_load = np.zeros(T)

    items = []
    for vi, v in enumerate(data["fleet"]):
        running = v["soc_init"] * v["bat_max"]
        for wi in range(len(v["dwells"])):
            trip_after = v["e_trip"][wi] if wi < len(v["e_trip"]) else 0
            need = max(0, trip_after - running)
            items.append((need, vi, wi, v["dwells"][wi], v["bat_max"], running))
            running -= trip_after
    items.sort(reverse=True, key=lambda r: r[0])

    n_var = sum(len(v["dwells"]) for v in data["fleet"])
    chrom = np.zeros(n_var, dtype=int)
    flat = {}
    k = 0
    for vi, v in enumerate(data["fleet"]):
        for wi in range(len(v["dwells"])):
            flat[(vi, wi)] = k
            k += 1

    for _, vi, wi, (s, e), bat_v, soc_v in items:
        s, e = int(s), int(e)
        headroom = bat_v - soc_v
        if headroom <= 0:
            continue
        n_slots_needed = int(np.ceil(headroom / (p_max * dt * eta)))
        n_slots = min(n_slots_needed, e - s)
        # Site-power + EVSE-count check: depot load stays under p_site_max
        # and concurrent EVSE count stays under M, across the needed slots.
        ok = all(
            used[t] + 1 <= M and cum_load[t] + p_max <= data["p_site_max"]
            for t in range(s, s + n_slots)
        )
        if ok:
            chrom[flat[(vi, wi)]] = 1
            for t in range(s, s + n_slots):
                used[t] += 1
                cum_load[t] += p_max
    return chrom


def uncoord_chrom(data):
    """Uncoordinated baseline: each vehicle plugs in at its NEAREST public
    station (shortest detour) at every dwell window. Models drivers acting
    independently with no system-level coordination, so load spreads across
    stations by proximity rather than piling onto a single station.

    Gene encoding: choice = 2 + f_idx selects public station f_idx.
    """
    chrom = []
    for v in data["fleet"]:
        nearest_f = int(np.argmin(v["detour"]))  # station with shortest detour
        for _ in range(len(v["dwells"])):
            chrom.append(2 + nearest_f)
    return np.array(chrom, dtype=int)


def is_feasible(data, res):
    """Hard feasibility on the two enforced constraints (p_site, EVSE)."""
    p_viol = np.sum(np.maximum(0, res["net_load"] - data["p_site_max"]))
    e_viol = np.sum(np.maximum(0, res["evses"] - data["evse_m"]))
    return p_viol < 1e-6 and e_viol < 1e-6


def connector_oversubscribed(data, res):
    """Reporting-only: True if any public connector capacity is exceeded."""
    c_viol = sum(
        np.sum(np.maximum(0, res["conn_occ"][f] - data["stations"][f]["n_conn"]))
        for f in range(data["F"])
    )
    return c_viol > 1e-6


class WarmStartSampling(Sampling):
    """NSGA-II initial population seeded with the heuristic schedules.

    Random init rarely produces the all-depot corner, so NSGA-II tends to
    miss the cheap high-reliability region. Seeding a few individuals with
    (i) the depot-greedy heuristic, (ii) an all-depot schedule, and
    (iii) the uncoordinated schedule anchors the search at the known corners;
    the remaining individuals are random for diversity.
    """

    def __init__(self, data):
        super().__init__()
        self.data = data

    def _do(self, problem, n_samples, **kwargs):
        hi = int(problem.xu[0]) + 1
        X = np.random.randint(0, hi, size=(n_samples, problem.n_var)).astype(float)
        seeds = [
            depot_only_chrom(self.data),
            np.ones(problem.n_var, dtype=int),  # all-depot
            uncoord_chrom(self.data),
        ]
        for k, s in enumerate(seeds):
            if k < n_samples:
                X[k] = s.astype(float)
        return X


# --- 3b. PARETO-FRONT METRICS ---
def reference_point(fronts, margin=0.10):
    """Shared HV reference point for a scenario: per-axis worst over the pooled
    archive of all seeds' fronts, inflated by `margin`. Held constant across
    seeds so HV values are directly comparable. Returns None if no points.
    """
    pooled = np.vstack([F for F in fronts if F is not None and len(F) > 0])
    if pooled.size == 0:
        return None
    nadir = pooled.max(axis=0)
    span = pooled.max(axis=0) - pooled.min(axis=0)
    span[span == 0] = 1.0  # avoid zero-margin on a degenerate axis
    return nadir + margin * span


def hv_of(points, ref):
    """Hypervolume of a 2-D objective set against ref. 0.0 if empty/None."""
    if points is None or len(points) == 0 or ref is None:
        return 0.0
    return float(Hypervolume(ref_point=ref).do(np.asarray(points)))


def heuristic_domination_gap(front, heur_point, ref):
    """Claim-1 number: HV lost when a heuristic point is added to the NSGA-II
    front. A positive gap means the front dominates territory the heuristic
    does not reach. Returns (hv_front, hv_with_heur, gap).
    """
    hv_front = hv_of(front, ref)
    combined = np.vstack([front, np.asarray(heur_point).reshape(1, -1)])
    hv_comb = hv_of(combined, ref)
    return hv_front, hv_comb, hv_front - hv_comb


# --- 4. PLOTTING ---
def plot_pareto(case_id, fronts, do_res, do_feas, un_res, un_feas, out_path):
    """Multi-population Pareto fronts + heuristic baseline markers."""
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    colors = plt.get_cmap("viridis")(np.linspace(0, 0.8, len(fronts)))
    for (lbl, F_pts), c in zip(fronts, colors):
        if F_pts is None or len(F_pts) == 0:
            continue
        F_pts = F_pts[np.argsort(F_pts[:, 0])]
        ax.plot(
            F_pts[:, 0],
            F_pts[:, 1],
            "o-",
            markersize=3,
            linewidth=1.0,
            color=c,
            label=f"NSGA-II ({lbl})",
        )
    ax.plot(
        do_res["cost"],
        do_res["shortfall"],
        "s",
        markersize=9,
        markerfacecolor="#e69f00" if do_feas else "white",
        markeredgecolor="#e69f00",
        markeredgewidth=1.5,
        label="Depot-Only" + ("" if do_feas else " (infeas)"),
    )
    ax.plot(
        un_res["cost"],
        un_res["shortfall"],
        "^",
        markersize=9,
        markerfacecolor="#d55e00" if un_feas else "white",
        markeredgecolor="#d55e00",
        markeredgewidth=1.5,
        label="Uncoordinated" + ("" if un_feas else " (infeas)"),
    )
    ax.set_title(f"Pareto Front: {case_id}")
    ax.set_xlabel("Cost ($)")
    ax.set_ylabel("Shortfall (kWh)")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_fulfilment(case_id, results, out_path):
    """Three-strategy fulfilment chart. Each bar = % of trip energy demand
    split into depot / public L2 / shortfall, summing to 100%. Initial SoC
    is attributed to depot. Infeasible strategies get a hatched shortfall
    segment + an "(infeas.)" label.
    """
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    x_pos = np.arange(len(results))
    for i, (name, r, feas, total_demand) in enumerate(results):
        d_pct = 100 * r["e_dep"] / total_demand
        p_pct = 100 * r["e_pub"] / total_demand
        s_pct = 100 * r["shortfall"] / total_demand
        ax.bar(i, d_pct, color="#0077b6", edgecolor="white", linewidth=0.5)
        ax.bar(
            i, p_pct, bottom=d_pct, color="#00b4d8", edgecolor="white", linewidth=0.5
        )
        ax.bar(
            i,
            s_pct,
            bottom=d_pct + p_pct,
            color="#d62728",
            edgecolor="white",
            linewidth=0.5,
            hatch="//" if not feas else None,
        )
        # Per-segment labels, centred in each segment (skip if too thin to fit)
        for value, bottom, txt_color in (
            (d_pct, 0.0, "white"),
            (p_pct, d_pct, "black"),
            (s_pct, d_pct + p_pct, "white"),
        ):
            if value >= 5:  # only label segments tall enough to read
                ax.text(
                    i,
                    bottom + value / 2,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=6,
                    fontweight="bold",
                    color=txt_color,
                )
        # Total delivered above the bar
        delivered = d_pct + p_pct
        ax.text(
            i,
            101.5,
            f"{delivered:.0f}% met",
            ha="center",
            fontsize=6,
            fontweight="bold",
        )
        if not feas:
            ax.text(
                i,
                -7,
                "(infeas.)",
                ha="center",
                fontsize=6,
                color="#d62728",
                style="italic",
            )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r[0] for r in results], fontsize=7)
    ax.set_ylabel("% of trip energy demand")
    ax.set_title(f"Trip Energy Delivered by Source: {case_id}", fontweight="bold")
    ax.set_ylim(-10, 110)
    ax.axhline(100, color="black", linestyle="--", linewidth=0.5)
    ax.legend(
        handles=[
            mpatches.Patch(color="#0077b6", label="Depot"),
            mpatches.Patch(color="#00b4d8", label="Public L2"),
            mpatches.Patch(color="#d62728", label="Shortfall"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        fontsize=6,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_charger_utilisation(case_id, data, uncoord_res, proposed_res, out_path):
    """Per-station horizontal stacked bars of public charger utilisation
    through the working day (9am-6pm) for the Proposed (NSGA-II) method.

    One thin bar per public station, stacking the hours the station spent in
    each occupancy state relative to connector capacity, so each bar sums to
    the 9-hour window:
        0          -> Available
        1..cap-1   -> Partially used
        >= cap     -> Fully utilised
    (uncoord_res is accepted for call-site compatibility but no longer shown.)
    """
    del uncoord_res  # uncoordinated baseline intentionally omitted

    dt = data["dt"]
    F = data["F"]
    n_conn = data["stations"][0]["n_conn"]
    # 9am = slot 36, 6pm = slot 72 (T=96 at 15-min slots from midnight)
    s0, s1 = int(9 / dt), int(18 / dt)

    colors = ["#eaecee", "#f0b429", "#cb4d28"]  # available / partial / full

    occ = proposed_res["conn_occ"][:, s0:s1]  # (F, window)
    avail = (occ == 0).sum(axis=1) * dt
    partial = ((occ >= 1) & (occ < n_conn)).sum(axis=1) * dt
    full = (occ >= n_conn).sum(axis=1) * dt
    states = np.column_stack([avail, partial, full])  # (F, 3)

    bar_h = 0.5  # thin bar; stations packed close together
    gap = 0.5  # vertical gap between station bars
    step = bar_h + gap
    y = np.arange(F) * step + gap

    fig, ax = plt.subplots(figsize=(6.0, 0.32 * F + 1.2))

    for f in range(F):
        left = 0.0
        for val, c in zip(states[f], colors):
            ax.barh(
                y[f], val, left=left, height=bar_h,
                color=c, edgecolor="#5b5b5b", linewidth=0.4, zorder=3,
            )
            left += val

    ax.set_xlim(0, (s1 - s0) * dt)
    ax.set_ylim(0, F * step + gap)
    ax.set_xlabel("Hours in state (09:00-18:00)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"S{f + 1}" for f in range(F)], fontsize=6)
    ax.set_ylabel("Public station")
    ax.set_title(f"Public Charger Utilisation (Proposed): {case_id}", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            mpatches.Patch(facecolor="#eaecee", edgecolor="#5b5b5b", label="Available"),
            mpatches.Patch(facecolor="#f0b429", edgecolor="#5b5b5b", label="Partially used"),
            mpatches.Patch(facecolor="#cb4d28", edgecolor="#5b5b5b", label="Fully utilised"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=3, frameon=False, fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_comparison_sim():
    scenarios = [[45, 15, 15], [60, 25, 20], [90, 30, 25]]
    alpha = 0.70  # single stress level; change manually to inspect others
    # NOTE: full (alpha, beta) Operational Stress Day sweep is future work.
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "simulation_results"
    )
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.8})

    pop_size = 200
    seeds = list(range(5))

    for N, F, M in scenarios:
        case_id = f"N{N}_F{F}_M{M}"
        case_dir = os.path.join(base_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)
        print(f"\n--- Scenario: {case_id} ---")

        data = load_or_generate_case_data(case_dir, N, F, M, alpha)
        problem = ChargingProblem(data)
        demand = data["total_trip_demand"]
        print(
            f"  Total trip demand: {demand:.1f} kWh   "
            f"Initial SoC: {sum(v['soc_init'] * v['bat_max'] for v in data['fleet']):.1f} kWh"
        )

        # Heuristic baselines (independent of NSGA-II generation budget)
        do_r = problem.simulate(depot_only_chrom(data))
        do_feas = is_feasible(data, do_r)
        print(
            f"  Depot-Only: cost=${do_r['cost']:.0f} short={do_r['shortfall']:.0f} "
            f"depot={100*do_r['e_dep']/demand:.0f}% L2={100*do_r['e_pub']/demand:.0f}% feas={do_feas}"
        )
        un_r = problem.simulate(uncoord_chrom(data))
        un_feas = is_feasible(data, un_r)
        un_conn = connector_oversubscribed(data, un_r)
        # Uncoord shown infeasible if it over-subscribes connectors (reporting)
        un_feas_display = un_feas and not un_conn
        print(
            f"  Uncoord:    cost=${un_r['cost']:.0f} short={un_r['shortfall']:.0f} "
            f"depot={100*un_r['e_dep']/demand:.0f}% L2={100*un_r['e_pub']/demand:.0f}% "
            f"feas(p/e)={un_feas} conn_oversub={un_conn}"
        )

        # Generation budgets per scenario. All cases run at 500; N=90 also runs
        # at 800 to show the front under a larger budget (two fronts overlaid).
        gen_budgets = [500, 800] if N == 90 else [500]
        multi = len(gen_budgets) > 1  # suffix output filenames only when >1 front

        # NSGA-II: multiple independent seeds at a single population size, run
        # once per generation budget. Seed-to-seed variance is the meaningful
        # axis for a thesis (report HV mean +/- std).
        pareto_fronts = []  # (label, front) per budget, for the overlaid Pareto plot
        csv_rows = []  # one metrics block per budget

        for n_gen in gen_budgets:
            seed_fronts = []  # res.F per seed
            seed_results = []  # full res objects per seed
            for sd in seeds:
                print(f"  NSGA-II pop={pop_size}, gen={n_gen}, seed={sd}...")
                res = minimize(
                    problem,
                    NSGA2(pop_size=pop_size, sampling=WarmStartSampling(data)),
                    ("n_gen", n_gen),
                    seed=sd,
                    verbose=False,
                )
                seed_fronts.append(res.F)
                seed_results.append(res)

            # Shared reference point across seeds (pooled nadir + 10% margin)
            ref = reference_point(seed_fronts)
            hvs = np.array([hv_of(F, ref) for F in seed_fronts])
            valid = hvs > 0
            if not valid.any():
                print(f"  gen={n_gen}: no feasible front on any seed; skipping.")
                continue
            hv_mean, hv_std = float(hvs[valid].mean()), float(hvs[valid].std())

            # Representative seed = the one whose HV is closest to the mean
            rep_idx = int(np.argmin(np.abs(hvs - hv_mean)))
            best = seed_results[rep_idx]
            rep_front = seed_fronts[rep_idx]
            print(
                f"  gen={n_gen}: HV across {valid.sum()} valid seeds: "
                f"mean={hv_mean:.1f} std={hv_std:.1f} (rep seed={seeds[rep_idx]})"
            )

            # Collect this budget's representative front for the Pareto overlay
            label = f"NSGA-II gen {n_gen}" if multi else f"seed {seeds[rep_idx]}"
            pareto_fronts.append((label, rep_front))

            # Claim-1 domination gaps for each heuristic, on the representative seed
            do_pt = [do_r["cost"], do_r["shortfall"]]
            un_pt = [un_r["cost"], un_r["shortfall"]]
            _, _, gap_do = heuristic_domination_gap(rep_front, do_pt, ref)
            _, _, gap_un = heuristic_domination_gap(rep_front, un_pt, ref)
            print(
                f"  gen={n_gen}: domination gap vs Depot-Only={gap_do:.1f}  "
                f"vs Uncoordinated={gap_un:.1f}"
            )

            csv_rows.append(
                {
                    "n_gen": n_gen,
                    "pop_size": pop_size,
                    "n_seeds": len(seeds),
                    "n_valid_seeds": int(valid.sum()),
                    "hv_mean": hv_mean,
                    "hv_std": hv_std,
                    "ref_cost": ref[0],
                    "ref_shortfall": ref[1],
                    "domination_gap_depot_only": gap_do,
                    "domination_gap_uncoord": gap_un,
                    "per_seed_hv": ";".join(f"{h:.2f}" for h in hvs),
                }
            )

            # Knee point of this budget's representative front.
            Fb = best.F
            norm_F = (Fb - Fb.min(axis=0)) / (Fb.max(axis=0) - Fb.min(axis=0) + 1e-6)
            # Reliability-weighted point: 80% weight on shortfall (obj 1), 20% on
            # cost (obj 0). ASF uses 1/weight, so the larger shortfall weight pulls
            # the selected solution toward the high-fulfilment end of the front,
            # reflecting an operator who prioritises service over marginal cost.
            rel_weights = np.array([0.2, 0.8])
            knee = int(ASF().do(norm_F, 1 / rel_weights).argmin())
            nsga_r = problem.simulate(best.X[knee])
            print(
                f"  gen={n_gen}: NSGA-II (rel-weighted): cost=${nsga_r['cost']:.0f} "
                f"short={nsga_r['shortfall']:.0f} "
                f"fulfil={100*(nsga_r['e_dep']+nsga_r['e_pub'])/demand:.0f}% "
                f"depot={100*nsga_r['e_dep']/demand:.0f}% L2={100*nsga_r['e_pub']/demand:.0f}%"
            )

            # Per-budget fulfilment + utilisation figures. Suffix the filename
            # with the generation budget only when more than one front exists.
            suffix = f"_gen{n_gen}" if multi else ""
            plot_fulfilment(
                case_id,
                [
                    ("Depot-Only", do_r, do_feas, demand),
                    ("Uncoord.", un_r, un_feas_display, demand),
                    ("NSGA-II", nsga_r, True, demand),
                ],
                os.path.join(case_dir, f"fulfillment{suffix}.png"),
            )
            plot_charger_utilisation(
                case_id,
                data,
                un_r,
                nsga_r,
                os.path.join(case_dir, f"charger_utilisation{suffix}.png"),
            )

        if not pareto_fronts:
            print("  No feasible front on any budget; skipping Pareto plot.")
            continue

        # Single Pareto figure overlaying all generation budgets for this case.
        plot_pareto(
            case_id,
            pareto_fronts,
            do_r,
            do_feas,
            un_r,
            un_feas_display,
            os.path.join(case_dir, "pareto.png"),
        )

        # Metrics CSV: one block per generation budget.
        with open(os.path.join(case_dir, "metrics.csv"), "w", newline="") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["scenario", "alpha", "n_gen", "pop_size", "n_seeds",
                        "n_valid_seeds", "hv_mean", "hv_std", "ref_cost",
                        "ref_shortfall", "depot_only_feasible",
                        "uncoord_feasible_display", "domination_gap_depot_only",
                        "domination_gap_uncoord", "per_seed_hv"])
            for r in csv_rows:
                w.writerow([case_id, alpha, r["n_gen"], r["pop_size"],
                            r["n_seeds"], r["n_valid_seeds"], r["hv_mean"],
                            r["hv_std"], r["ref_cost"], r["ref_shortfall"],
                            do_feas, un_feas_display,
                            r["domination_gap_depot_only"],
                            r["domination_gap_uncoord"], r["per_seed_hv"]])


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run_comparison_sim()
