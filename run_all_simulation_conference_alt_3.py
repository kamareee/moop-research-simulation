# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.decomposition.asf import ASF


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
                "soc_init": 0.35,
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
    """All vehicles plug in at L2 station 0 every dwell."""
    n_var = sum(len(v["dwells"]) for v in data["fleet"])
    return np.full(n_var, 2, dtype=int)


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


# --- 4. PLOTTING ---
def plot_pareto(case_id, fronts, do_res, do_feas, un_res, un_feas, out_path):
    """Multi-population Pareto fronts + heuristic baseline markers."""
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    colors = plt.get_cmap("viridis")(np.linspace(0, 0.8, len(fronts)))
    for (pop, F_pts), c in zip(fronts, colors):
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
            label=f"NSGA-II P:{pop}",
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


def plot_charger_utilisation(case_id, data, basic_res, proposed_res, out_path):
    """Parallel-bars view of public charger utilisation over the day.

    For each public station, two horizontal bars are drawn:
      - blue:  Basic case      (uncoordinated baseline)
      - red:   Proposed method (NSGA-II)
    Bars are drawn from *attempted* occupancy, so any time the basic case
    tries to use a station that is offline at that time, the blue bar will
    overlap a grey availability-blackout band -- making the violation
    visible. The proposed method should keep its bars clear of grey bands.
    All public stations are shown; the time axis covers 06:00 -> 00:00.
    """
    dt = data["dt"]
    F = data["F"]
    pub_avail = data["pub_availability"]  # (T, F): 1 = available, 0 = offline

    # Active operating window: 06:00 -> 24:00 (slot indices 24..96 at dt=0.25)
    t0_slot, t1_slot = int(6 / dt), int(24 / dt)

    def runs(mask):
        """Convert a boolean 1-D mask into (start_h, end_h) intervals in hours."""
        spans, t = [], 0
        while t < len(mask):
            if mask[t]:
                s = t
                while t < len(mask) and mask[t]:
                    t += 1
                spans.append((s * dt, t * dt))
            else:
                t += 1
        return spans

    def busy_intervals(res):
        """Per-station list of busy (start_h, end_h) intervals, from attempted
        occupancy if present, else from successful occupancy."""
        occ = res.get("conn_occ_attempted", res["conn_occ"])
        return [runs(occ[f] > 0) for f in range(F)]

    basic_spans = busy_intervals(basic_res)
    prop_spans = busy_intervals(proposed_res)

    blue, red = "#1f4ed8", "#d11f1f"  # match sample: basic = blue, proposed = red
    grey = "#d6dadf"  # availability-blackout shading
    bar_h = 0.32
    offset = 0.22  # vertical offset between the two bars within a station row

    fig, ax = plt.subplots(figsize=(6.0, 0.45 * F + 1.8))

    for f in range(F):
        y = f + 1  # ascending: station 1 at bottom, station F at top (matches sample)
        # Availability blackout: shade slots where station f is offline
        offline = pub_avail[:, f] < 0.5
        for s, e in runs(offline):
            ax.barh(
                y, e - s, left=s, height=0.82, color=grey, edgecolor="none", zorder=1
            )
        # Basic case (blue) -- drawn slightly above centre
        for s, e in basic_spans[f]:
            ax.barh(
                y + offset,
                e - s,
                left=s,
                height=bar_h,
                color=blue,
                edgecolor=blue,
                zorder=3,
            )
        # Proposed method (red) -- drawn slightly below centre
        for s, e in prop_spans[f]:
            ax.barh(
                y - offset,
                e - s,
                left=s,
                height=bar_h,
                color=red,
                edgecolor=red,
                zorder=3,
            )

    # X-axis: 06:00 -> 00:00, ticks every 2 hours
    ax.set_xlim(t0_slot * dt, t1_slot * dt)
    xt = list(range(6, 25, 2))
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{h % 24:02d}:00" for h in xt], fontsize=7)
    ax.set_xlabel("Time (hh:mm)")

    # Y-axis: one row per public station
    ax.set_ylim(0.4, F + 0.6)
    ax.set_yticks(range(1, F + 1))
    ax.set_yticklabels([f"S{i + 1}" for i in range(F)], fontsize=7)
    ax.set_ylabel("Public station")

    ax.set_title(f"Public Charger Utilisation: {case_id}", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            mpatches.Patch(color=blue, label="Basic case"),
            mpatches.Patch(color=red, label="Proposed method"),
            mpatches.Patch(color=grey, label="Station offline"),
        ],
        loc="lower left",
        fontsize=7,
        frameon=True,
        ncol=3,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_comparison_sim():
    scenarios = [[45, 15, 15], [60, 25, 20], [90, 30, 25]]
    pop_sizes = [150, 200, 250]
    alpha = 0.70
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "simulation_results"
    )
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.8})

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

        # Heuristic baselines
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

        # NSGA-II at each population size
        n_gen = 300 if N == 90 else 200
        fronts = []
        best = None
        for pop in pop_sizes:
            print(f"  NSGA-II pop={pop}, gen={n_gen}...")
            res = minimize(
                problem, NSGA2(pop_size=pop), ("n_gen", n_gen), seed=1, verbose=False
            )
            fronts.append((pop, res.F))
            if pop == pop_sizes[-1] and res.F is not None and len(res.F) > 0:
                best = res

        plot_pareto(
            case_id,
            fronts,
            do_r,
            do_feas,
            un_r,
            un_feas_display,
            os.path.join(case_dir, "pareto.png"),
        )

        if best is None:
            print("  No feasible front; skipping fulfilment plot.")
            continue

        # Knee point of largest-population front for the fulfilment chart
        Fb = best.F
        norm_F = (Fb - Fb.min(axis=0)) / (Fb.max(axis=0) - Fb.min(axis=0) + 1e-6)
        knee = int(ASF().do(norm_F, 1 / np.array([0.5, 0.5])).argmin())
        nsga_r = problem.simulate(best.X[knee])
        print(
            f"  NSGA-II knee: cost=${nsga_r['cost']:.0f} short={nsga_r['shortfall']:.0f} "
            f"depot={100*nsga_r['e_dep']/demand:.0f}% L2={100*nsga_r['e_pub']/demand:.0f}%"
        )

        plot_fulfilment(
            case_id,
            [
                ("Depot-Only", do_r, do_feas, demand),
                ("Uncoord.", un_r, un_feas_display, demand),
                ("NSGA-II", nsga_r, True, demand),
            ],
            os.path.join(case_dir, "fulfillment.png"),
        )

        plot_charger_utilisation(
            case_id,
            data,
            un_r,
            nsga_r,
            os.path.join(case_dir, "charger_utilisation.png"),
        )


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run_comparison_sim()
