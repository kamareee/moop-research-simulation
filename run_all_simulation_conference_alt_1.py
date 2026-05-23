"""
Joint depot–public L2 charging coordination — v6 (minimal).

Bi-objective NSGA-II on (cost, shortfall) for a commercial EV fleet over
24 hours, with depot and L2-only public stations. Produces a Pareto plot
and a fulfilment chart per scenario.
"""

import os
import pickle
import random

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.decomposition.asf import ASF
from pymoo.optimize import minimize


# ---------- data generation ----------
def make_case(N, F, M, alpha):
    T, dt = 96, 0.25
    p_site = (M / 15) * 200.0

    # Depot ToU price (commercial AU profile)
    pi_depot = np.zeros(T)
    pi_depot[0:28] = 0.1781
    pi_depot[28:60] = 0.2087
    pi_depot[60:84] = 0.3121
    pi_depot[84:88] = 0.2087
    pi_depot[88:96] = 0.1781

    t = np.arange(T)
    p_pv = (p_site * 0.15) * np.maximum(0, np.sin(np.pi * (t - 24) / 48))
    p_base = (p_site * 0.30) + 10 * np.random.rand(T)

    # Public L2 retail = ToU + $0.20 premium
    pi_l2 = pi_depot + 0.20

    # Per-station availability mask (some offline overnight / mid-day)
    avail = np.ones((T, F))
    for f in range(F):
        avail[0:24, f] = 0
        avail[88:96, f] = 0
        if f % 2 == 0:
            avail[48:56, f] = 0

    stations = [
        {
            "price": pi_l2,
            "wait": 10 + 15 * np.sin(np.pi * t / 96) ** 2,
            "n_conn": 2,
        }
        for _ in range(F)
    ]

    fleet = []
    total_demand = 0.0
    for _ in range(N):
        shift = random.randint(-6, 6)
        dwells = [
            (0, 28 + shift),
            (42 + shift, 68 + shift),
            (82 + shift, 96),
        ]
        bat = 75.0
        trip_total = bat * (alpha + random.uniform(-0.05, 0.05))
        trips = [trip_total * 0.45, trip_total * 0.55]
        total_demand += sum(trips)
        fleet.append(
            {
                "bat": bat,
                "soc0": 0.30 * bat,
                "trips": trips,
                "dwells": dwells,
                "detour": np.random.randint(8, 25, size=F),
            }
        )

    return {
        "T": T,
        "dt": dt,
        "F": F,
        "M": M,
        "N": N,
        "pi_depot": pi_depot,
        "p_base": p_base,
        "p_pv": p_pv,
        "p_site": p_site,
        "avail": avail,
        "stations": stations,
        "fleet": fleet,
        "total_demand": total_demand,
        "L2_kw": 11.0,
        "depot_kw": 21.0,
        "depot_eta": 0.90,
        "l2_eta": 0.92,
        "detour_cost": 0.40,
        "demand_charge": 5.0,
        "min_session_min": 15.0,
    }


def load_case(case_dir, N, F, M, alpha):
    path = os.path.join(case_dir, f"data_N{N}_M{M}_a{alpha}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    data = make_case(N, F, M, alpha)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data


# ---------- simulation ----------
def simulate(data, x):
    """
    Forward-simulate one chromosome. Returns (cost, shortfall, e_depot,
    e_public, depot_load, evses, conn_occ).

    Source accounting: every kWh in a vehicle's battery is tagged as
    either 'from depot' or 'from public'. Initial SoC is tagged as
    depot-origin (the fleet was charged overnight at the depot before
    the planning horizon began). Invariant: b_dep[i] + b_pub[i] == soc[i]
    at all times. e_dep and e_pub count trip energy delivered from each
    source; together with shortfall they sum to total trip demand.
    """
    x = np.round(x).astype(int)
    T, dt, F = data["T"], data["dt"], data["F"]
    pi_depot = data["pi_depot"]
    fleet = data["fleet"]

    soc = np.array([v["soc0"] for v in fleet], dtype=float)
    bat = np.array([v["bat"] for v in fleet], dtype=float)
    # Initial SoC is tagged as depot-origin (overnight depot charge)
    b_dep = soc.copy()
    b_pub = np.zeros(data["N"])

    cost = 0.0
    shortfall = 0.0
    e_dep_trips = 0.0
    e_pub_trips = 0.0
    charged_dep = 0.0  # in-horizon depot kWh delivered to batteries
    charged_pub = 0.0  # in-horizon public kWh delivered to batteries
    depot_load = np.zeros(T)
    evses = np.zeros(T)
    conn_occ = np.zeros((F, T))

    idx = 0
    for i, v in enumerate(fleet):
        for j, (start, end) in enumerate(v["dwells"]):
            choice = x[idx]
            idx += 1
            energy_in = 0.0
            source = None

            if choice == 1:  # depot
                headroom = bat[i] - soc[i]
                if headroom > 0:
                    # Charge for as many slots as needed to fill battery,
                    # capped by dwell length. Each slot draws depot_kw.
                    n_slots_needed = int(
                        np.ceil(headroom / (data["depot_kw"] * dt * data["depot_eta"]))
                    )
                    n_slots = min(n_slots_needed, end - start)
                    if n_slots > 0:
                        p = data["depot_kw"]
                        # Last slot may be partial — cap energy at headroom
                        energy_in = min(p * dt * data["depot_eta"] * n_slots, headroom)
                        # Charge for the actual slots used (start to start+n_slots)
                        cost += np.sum(pi_depot[start : start + n_slots]) * p * dt
                        depot_load[start : start + n_slots] += p
                        evses[start : start + n_slots] += 1
                        source = "dep"

            elif choice > 1 and choice <= 1 + F:  # L2 station (choice-2)
                f_idx = choice - 2
                if data["avail"][start, f_idx] > 0:
                    st = data["stations"][f_idx]
                    detour = float(v["detour"][f_idx])
                    wait = float(st["wait"][start])
                    net_min = (end - start) * dt * 60 - detour - wait
                    if net_min >= data["min_session_min"]:
                        headroom = bat[i] - soc[i]
                        if headroom > 0:
                            e_max = data["L2_kw"] * (net_min / 60) * data["l2_eta"]
                            energy_in = min(e_max, headroom)
                            cost += (energy_in / data["l2_eta"]) * float(
                                st["price"][start]
                            )
                            cost += data["detour_cost"] * (detour + wait)
                            slot_min = dt * 60
                            occ_start = min(T, start + int(np.ceil(detour / slot_min)))
                            occ_end = min(
                                T, occ_start + int(np.ceil(net_min / slot_min))
                            )
                            conn_occ[f_idx, occ_start:occ_end] += 1
                            source = "pub"

            if source == "dep":
                soc[i] += energy_in
                b_dep[i] += energy_in
                charged_dep += energy_in
            elif source == "pub":
                soc[i] += energy_in
                b_pub[i] += energy_in
                charged_pub += energy_in

            # Trip after this dwell (if any)
            if j < len(v["trips"]):
                trip = v["trips"][j]
                if soc[i] >= trip:
                    # All trip energy is tagged (b_dep + b_pub == soc invariant)
                    share_dep = b_dep[i] / soc[i]
                    share_pub = b_pub[i] / soc[i]
                    e_dep_trips += trip * share_dep
                    e_pub_trips += trip * share_pub
                    b_dep[i] -= trip * share_dep
                    b_pub[i] -= trip * share_pub
                    soc[i] -= trip
                else:
                    # Shortfall: deliver entire remaining SoC, all tagged
                    if soc[i] > 0:
                        share_dep = b_dep[i] / soc[i]
                        share_pub = b_pub[i] / soc[i]
                        e_dep_trips += soc[i] * share_dep
                        e_pub_trips += soc[i] * share_pub
                    shortfall += trip - soc[i]
                    soc[i] = 0
                    b_dep[i] = 0
                    b_pub[i] = 0

    # Demand charge on incremental peak above base
    net_import = depot_load + data["p_base"] - data["p_pv"]
    peak = float(np.max(np.maximum(0, net_import)))
    baseline = float(np.max(np.maximum(0, data["p_base"] - data["p_pv"])))
    cost += max(0, peak - baseline) * data["demand_charge"]

    return {
        "cost": cost,
        "shortfall": shortfall,
        "e_dep": e_dep_trips,
        "e_pub": e_pub_trips,
        "charged_dep": charged_dep,
        "charged_pub": charged_pub,
        "final_soc": float(soc.sum()),
        "depot_load": depot_load,
        "evses": evses,
        "conn_occ": conn_occ,
    }


# ---------- problem & baselines ----------
class ChargingProblem(ElementwiseProblem):
    def __init__(self, data):
        self.d = data
        n_var = sum(len(v["dwells"]) for v in data["fleet"])
        super().__init__(n_var=n_var, n_obj=2, n_constr=3, xl=0, xu=1 + data["F"])

    def _evaluate(self, x, out, *args, **kwargs):
        r = simulate(self.d, x)
        net = r["depot_load"] + self.d["p_base"] - self.d["p_pv"]
        p_viol = np.sum(np.maximum(0, net - self.d["p_site"])) * self.d["dt"]
        e_viol = np.sum(np.maximum(0, r["evses"] - self.d["M"]))
        c_viol = sum(
            np.sum(np.maximum(0, r["conn_occ"][f] - self.d["stations"][f]["n_conn"]))
            for f in range(self.d["F"])
        )
        out["F"] = [r["cost"], r["shortfall"]]
        out["G"] = [p_viol, e_viol, c_viol]


def depot_only_chrom(data):
    """Capacity-respecting depot-greedy: schedule by urgency, respect M and site limit.

    Reservation only spans the slots actually needed to fill each vehicle's
    battery, matching the simulator's one-pulse-per-dwell behaviour.
    """
    T, M = data["T"], data["M"]
    p_max = data["depot_kw"]
    dt = data["dt"]
    eta = data["depot_eta"]
    used = np.zeros(T, dtype=int)
    cum_load = np.zeros(T)

    items = []
    for vi, v in enumerate(data["fleet"]):
        running = v["soc0"]
        for wi in range(len(v["dwells"])):
            trip_after = v["trips"][wi] if wi < len(v["trips"]) else 0
            need = max(0, trip_after - running)
            items.append((need, vi, wi, v["dwells"][wi], v["bat"], running))
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
        headroom = bat_v - soc_v
        if headroom <= 0:
            continue
        n_slots_needed = int(np.ceil(headroom / (p_max * dt * eta)))
        n_slots = min(n_slots_needed, e - s)
        ok = all(
            used[t] + 1 <= M
            and cum_load[t] + p_max + data["p_base"][t] - data["p_pv"][t]
            <= data["p_site"]
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


def is_feasible(data, r):
    net = r["depot_load"] + data["p_base"] - data["p_pv"]
    p_viol = np.sum(np.maximum(0, net - data["p_site"])) * data["dt"]
    e_viol = np.sum(np.maximum(0, r["evses"] - data["M"]))
    c_viol = sum(
        np.sum(np.maximum(0, r["conn_occ"][f] - data["stations"][f]["n_conn"]))
        for f in range(data["F"])
    )
    return p_viol < 1e-6 and e_viol < 1e-6 and c_viol < 1e-6


# ---------- plotting ----------
def plot_pareto(case_id, fronts, do_res, do_feas, un_res, un_feas, out_path):
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
    """For each strategy, show how delivered trip energy splits depot vs public.

    Heights = % of trip energy delivered from each source. Bar totals
    = total delivery rate (delivered / total demand). Shortfall is shown
    above as a separate red segment to make total demand visible.
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
        # Label total delivered (depot + public) at top of charged section
        delivered = d_pct + p_pct
        ax.text(
            i,
            delivered + 1,
            f"{delivered:.0f}%",
            ha="center",
            fontsize=7,
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


# ---------- main ----------
def energy_balance(name, r, data, feas=None):
    """Print one-line strategy summary + energy-balance reconciliation.

    Reconciliation identity (per simulator):
      initial_SoC + charged_dep + charged_pub
        == trip_consumed + shortfall_residual_in_battery + final_SoC
    where trip_consumed = total_demand - shortfall (energy actually delivered).
    """
    init_soc = sum(v["soc0"] for v in data["fleet"])
    demand = data["total_demand"]
    delivered = demand - r["shortfall"]
    lhs = init_soc + r["charged_dep"] + r["charged_pub"]
    rhs = delivered + r["final_soc"]
    resid = lhs - rhs
    flag = "" if feas is None else (" feas" if feas else " INFEAS")
    print(f"  {name}:{flag}")
    print(
        f"    cost=${r['cost']:.0f}  shortfall={r['shortfall']:.1f} kWh  "
        f"(={100 * r['shortfall'] / demand:.1f}% of demand)"
    )
    print(
        f"    trip-delivered: e_dep={r['e_dep']:.1f}  e_pub={r['e_pub']:.1f}  "
        f"sum={r['e_dep'] + r['e_pub']:.1f}  (delivered={delivered:.1f})"
    )
    print(
        f"    balance: init_SoC={init_soc:.1f} + charged_dep={r['charged_dep']:.1f} + "
        f"charged_pub={r['charged_pub']:.1f}"
    )
    print(
        f"             = delivered={delivered:.1f} + final_SoC={r['final_soc']:.1f}  "
        f"(residual={resid:+.2f})"
    )


def run():
    scenarios = [
        (45, 15, 15, 0.75, "N45_F15_M15"),
        (60, 20, 20, 0.75, "N60_F20_M20"),
        (90, 30, 30, 0.75, "N90_F30_M30"),
    ]
    pop_sizes = [100, 150, 200]
    n_gen = 120
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.8})

    for N, F, M, alpha, label in scenarios:
        case_dir = os.path.join(base_dir, label)
        os.makedirs(case_dir, exist_ok=True)
        print(f"\n--- {label} (N={N}, F={F}, M={M}, α={alpha}) ---")
        data = load_case(case_dir, N, F, M, alpha)
        print(
            f"  Total trip demand: {data['total_demand']:.1f} kWh   "
            f"Initial SoC total: {sum(v['soc0'] for v in data['fleet']):.1f} kWh"
        )
        problem = ChargingProblem(data)

        do_r = simulate(data, depot_only_chrom(data))
        do_feas = is_feasible(data, do_r)
        energy_balance("Depot-Only", do_r, data, do_feas)
        un_r = simulate(data, uncoord_chrom(data))
        un_feas = is_feasible(data, un_r)
        energy_balance("Uncoord.", un_r, data, un_feas)

        fronts = []
        best = None
        for pop in pop_sizes:
            print(f"  NSGA-II pop={pop}...")
            res = minimize(
                problem, NSGA2(pop_size=pop), ("n_gen", n_gen), seed=1, verbose=False
            )
            fronts.append((pop, res.F))
            if pop == pop_sizes[-1] and res.F is not None and len(res.F) > 0:
                best = res

        plot_pareto(
            label,
            fronts,
            do_r,
            do_feas,
            un_r,
            un_feas,
            os.path.join(case_dir, "pareto.png"),
        )

        if best is None:
            print("  No feasible front; skipping fulfilment plot.")
            continue

        F = best.F
        norm_F = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-9)
        knee = int(np.argmin(ASF().do(norm_F, 1.0 / np.array([0.5, 0.5]))))
        nsga_r = simulate(data, best.X[knee])
        energy_balance("NSGA-II (knee)", nsga_r, data, True)

        plot_fulfilment(
            label,
            [
                ("Depot-Only", do_r, do_feas, data["total_demand"]),
                ("Uncoord.", un_r, un_feas, data["total_demand"]),
                ("NSGA-II", nsga_r, True, data["total_demand"]),
            ],
            os.path.join(case_dir, "fulfillment.png"),
        )


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run()
