# pylint: disable=missing-module-docstring,missing-function-docstring
"""
Chapter 5 (RQ3) thesis run wrapper.

Imports run_all_simulation_thesis_v5 UNCHANGED (same data, same problem,
same warm start, same NSGA-II configuration, same seeds) and adds:

  1. Saves every array needed to regenerate the chapter's figures and
     numbers (.npz, no pickle):
        - per budget / per seed: final front F, decision vectors X,
          per-generation front history (NaN-padded) for convergence curves
        - baselines: chromosome, cost, shortfall, depot load, EVSE count,
          connector occupancy, trip-delivered energy by source
        - representative (80/20 reliability-weighted) operating point per
          budget: same fields as baselines
        - depot peak load for every point on the representative front
          (tracked metric, Section 5.3.4)
  2. Metric fix A: ONE scenario-wide hypervolume reference point pooled
     over ALL budgets and seeds, so 500- vs 800-generation HV is comparable.
     (v5 recomputed the reference per budget.)  The per-budget reference and
     HV are still written for traceability.
  3. Metric fix B: replaces the degenerate "domination gap"
     (HV(front) - HV(front U baseline), which is <= 0 by construction) with
        cost_saving_at_leq_shortfall  = baseline cost - min cost of front
                                         points with shortfall <= baseline's
        shortfall_reduction_at_leq_cost = baseline shortfall - min shortfall
                                         of front points with cost <= baseline's
     plus a boolean 'dominated' flag.  The original gap is still written.

Usage (from the repo directory, same as v5):
    uv run python run_ch5_thesis.py
Quick smoke test (not for the thesis):
    uv run python run_ch5_thesis.py --quick

Outputs go to simulation_results_ch5/<case_id>/ .  Existing
simulation_results/<case_id>/sim_data_*.pkl files are reused if present so
inputs are identical to the v5 run; otherwise they are generated with the
same seeds as v5.
"""

import argparse
import csv
import json
import os
import random
import shutil

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pymoo.algorithms.moo.nsga2 import NSGA2  # noqa: E402
from pymoo.core.callback import Callback  # noqa: E402
from pymoo.decomposition.asf import ASF  # noqa: E402
from pymoo.optimize import minimize  # noqa: E402

import run_all_simulation_thesis_v5 as S  # noqa: E402  (unchanged v5)

HERE = os.path.dirname(os.path.abspath(__file__))


# --- history callback ---------------------------------------------------------
class FrontHistory(Callback):
    """Record the current non-dominated (feasible) front after every generation."""

    def __init__(self):
        super().__init__()
        self.fronts = []

    def notify(self, algorithm):
        opt = algorithm.opt
        F = opt.get("F") if opt is not None and len(opt) > 0 else None
        self.fronts.append(None if F is None else np.asarray(F, dtype=float))


def pad_history(fronts, pop_size):
    """(n_gen, pop_size, 2) float array, NaN-padded; empty gens are all-NaN."""
    out = np.full((len(fronts), pop_size, 2), np.nan)
    for g, F in enumerate(fronts):
        if F is not None and len(F) > 0:
            k = min(len(F), pop_size)
            out[g, :k, :] = F[:k]
    return out


# --- metric fix B -------------------------------------------------------------
def baseline_gain(front, cost_b, short_b):
    """Interpretable domination numbers for one baseline point."""
    front = np.asarray(front)
    m_short = front[:, 1] <= short_b + 1e-9
    m_cost = front[:, 0] <= cost_b + 1e-9
    cost_saving = (cost_b - front[m_short, 0].min()) if m_short.any() else np.nan
    short_red = (short_b - front[m_cost, 1].min()) if m_cost.any() else np.nan
    # Strict domination: <= on both objectives and < on at least one. Needed
    # because both baseline schedules are warm-start seeds and can sit ON the
    # front, which would make a weak test trivially true.
    strict = (front[:, 0] < cost_b - 1e-9) | (front[:, 1] < short_b - 1e-9)
    dominated = bool(np.any(m_short & m_cost & strict))
    return cost_saving, short_red, dominated


def res_to_dict(prefix, r):
    return {
        f"{prefix}_cost": r["cost"],
        f"{prefix}_shortfall": r["shortfall"],
        f"{prefix}_net_load": r["net_load"],
        f"{prefix}_evses": r["evses"],
        f"{prefix}_conn_occ": r["conn_occ"],
        f"{prefix}_conn_occ_attempted": r["conn_occ_attempted"],
        f"{prefix}_e_dep": r["e_dep"],
        f"{prefix}_e_pub": r["e_pub"],
        f"{prefix}_e_depot_charged": r["e_depot"],
        f"{prefix}_e_L2_charged": r["e_L2"],
        f"{prefix}_peak_load": float(np.max(r["net_load"])),
    }


# --- main ---------------------------------------------------------------------
def run(scenarios, alpha, pop_size, seeds, budgets_fn, out_root, v5_root, allow_generate=False):
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.8})
    for N, F, M in scenarios:
        case_id = f"N{N}_F{F}_M{M}"
        out_dir = os.path.join(out_root, case_id)
        os.makedirs(out_dir, exist_ok=True)
        v5_dir = os.path.join(v5_root, case_id)
        os.makedirs(v5_dir, exist_ok=True)
        print(f"\n=== Scenario {case_id} ===")

        # Reuse the EXISTING v5 sim_data pickle so inputs are identical to the
        # v5 run. Accept both the v5 name (alpha0.7) and the upload-sanitised
        # name (alpha0_7). Never regenerate silently: regenerated data would
        # differ for N60/N90 because NSGA-II advances the NumPy RNG between
        # scenarios, so the fleet/detour/availability draws would not match.
        cands = [os.path.join(v5_dir, f"sim_data_N{N}_M{M}_alpha{alpha}.pkl"),
                 os.path.join(v5_dir, f"sim_data_N{N}_M{M}_alpha{str(alpha).replace('.', '_')}.pkl")]
        found = [c for c in cands if os.path.exists(c)]
        if found:
            import pickle
            with open(found[0], "rb") as fpk:
                data = pickle.load(fpk)
            print(f"  inputs: {found[0]}")
        elif allow_generate:
            print("  WARNING: no cached sim_data found; generating new data (will NOT match v5 run)")
            data = S.load_or_generate_case_data(v5_dir, N, F, M, alpha)
            found = [cands[0]]
        else:
            raise FileNotFoundError(
                f"No sim_data pickle for {case_id} in {v5_dir}. Expected one of:\n  "
                + "\n  ".join(cands) + "\nRe-run with --allow-generate only if you intend new data.")
        shutil.copy(found[0], os.path.join(out_dir, os.path.basename(cands[0])))
        problem = S.ChargingProblem(data)
        demand = data["total_trip_demand"]
        soc0 = sum(v["soc_init"] * v["bat_max"] for v in data["fleet"])

        # Scenario inputs as plain arrays (so nothing needs the pickle)
        np.savez(
            os.path.join(out_dir, f"ch5_{case_id}_inputs.npz"),
            N=N, F=F, M=M, T=data["T"], dt=data["dt"], alpha=alpha,
            p_site_max=data["p_site_max"], c_d_t=data["c_d_t"],
            pi_demand=data["pi_demand"], pi_grid=data["pi_grid"],
            pi_pub=data["stations"][0]["pi_pub"],
            station_power=np.array([s["power"] for s in data["stations"]]),
            wait_profile=data["stations"][0]["wait_profile"],
            n_conn=np.array([s["n_conn"] for s in data["stations"]]),
            pub_availability=data["pub_availability"],
            bat_max=np.array([v["bat_max"] for v in data["fleet"]]),
            soc_init=np.array([v["soc_init"] for v in data["fleet"]]),
            e_trip=np.array([v["e_trip"] for v in data["fleet"]]),
            dwells=np.array([v["dwells"] for v in data["fleet"]]),
            detour=np.array([v["detour"] for v in data["fleet"]]),
            total_trip_demand=demand, initial_soc_kwh=soc0,
        )

        # Baselines (unchanged from v5)
        do_x = S.depot_only_chrom(data)
        do_r = problem.simulate(do_x)
        do_feas = S.is_feasible(data, do_r)
        un_x = S.uncoord_chrom(data)
        un_r = problem.simulate(un_x)
        un_feas = S.is_feasible(data, un_r)
        un_conn = S.connector_oversubscribed(data, un_r)
        un_feas_display = un_feas and not un_conn
        print(f"  Depot-Only: cost={do_r['cost']:.1f} short={do_r['shortfall']:.1f} feas={do_feas}")
        print(f"  Uncoord:    cost={un_r['cost']:.1f} short={un_r['shortfall']:.1f} feas={un_feas} oversub={un_conn}")
        bl = {"depot_only_x": do_x, "uncoord_x": un_x,
              "depot_only_feasible": do_feas, "uncoord_feasible_pe": un_feas,
              "uncoord_conn_oversubscribed": un_conn,
              "uncoord_feasible_display": un_feas_display,
              "total_trip_demand": demand}
        bl.update(res_to_dict("depot_only", do_r))
        bl.update(res_to_dict("uncoord", un_r))
        np.savez(os.path.join(out_dir, f"ch5_{case_id}_baselines.npz"), **bl)

        budgets = budgets_fn(N)
        runs = {}  # (n_gen, seed) -> dict(F, X, hist, n_evals)
        for n_gen in budgets:
            for sd in seeds:
                print(f"  NSGA-II pop={pop_size} gen={n_gen} seed={sd} ...", flush=True)
                cb = FrontHistory()
                res = minimize(
                    problem,
                    NSGA2(pop_size=pop_size, sampling=S.WarmStartSampling(data)),
                    ("n_gen", n_gen),
                    seed=sd,
                    callback=cb,
                    verbose=False,
                )
                F_final = None if res.F is None else np.atleast_2d(res.F)
                X_final = None if res.X is None else np.atleast_2d(res.X)
                runs[(n_gen, sd)] = dict(
                    F=F_final, X=X_final, hist=pad_history(cb.fronts, pop_size),
                    n_evals=int(res.algorithm.evaluator.n_eval),
                    exec_time=float(res.exec_time),
                )

        # --- Metric fix A: one scenario-wide reference point ------------------
        all_fronts = [r["F"] for r in runs.values() if r["F"] is not None]
        ref_scn = S.reference_point(all_fronts)
        if ref_scn is None:
            print("  No feasible front on any run; skipping scenario.")
            continue

        fronts_npz = {"ref_scenario": ref_scn, "pop_size": pop_size,
                      "seeds": np.array(seeds), "budgets": np.array(budgets)}
        csv_rows = []
        summary = {"case_id": case_id, "N": N, "F": F, "M": M, "alpha": alpha,
                   "total_trip_demand": demand, "initial_soc_kwh": soc0,
                   "ref_scenario": ref_scn.tolist(), "budgets": {}}
        pareto_fronts = []
        for n_gen in budgets:
            seed_F = [runs[(n_gen, sd)]["F"] for sd in seeds]
            ref_b = S.reference_point(seed_F)  # v5 per-budget ref (traceability)
            hv_scn = np.array([S.hv_of(Fk, ref_scn) for Fk in seed_F])
            hv_b = np.array([S.hv_of(Fk, ref_b) for Fk in seed_F])
            valid = hv_scn > 0
            if not valid.any():
                print(f"  gen={n_gen}: no feasible front on any seed.")
                continue
            hv_mean, hv_std = float(hv_scn[valid].mean()), float(hv_scn[valid].std())
            rep = int(np.argmin(np.abs(hv_scn - hv_mean)))
            rep_F = seed_F[rep]
            rep_X = runs[(n_gen, seeds[rep])]["X"]
            print(f"  gen={n_gen}: HV(scenario ref) mean={hv_mean:.1f} std={hv_std:.1f} rep seed={seeds[rep]}")

            # HV convergence history against the scenario-wide reference
            hist_hv = np.zeros((len(seeds), n_gen))
            for k, sd in enumerate(seeds):
                H = runs[(n_gen, sd)]["hist"]
                for g in range(H.shape[0]):
                    pts = H[g][~np.isnan(H[g][:, 0])]
                    hist_hv[k, g] = S.hv_of(pts, ref_scn) if len(pts) else 0.0

            # Metric fix B + original gap (traceability)
            do_pt = (do_r["cost"], do_r["shortfall"])
            un_pt = (un_r["cost"], un_r["shortfall"])
            do_cs, do_sr, do_dom = baseline_gain(rep_F, *do_pt)
            un_cs, un_sr, un_dom = baseline_gain(rep_F, *un_pt)
            _, _, gap_do = S.heuristic_domination_gap(rep_F, list(do_pt), ref_b)
            _, _, gap_un = S.heuristic_domination_gap(rep_F, list(un_pt), ref_b)
            print(f"    vs Depot-Only: dominated={do_dom} cost saving @<=shortfall={do_cs:.1f} $ ; "
                  f"shortfall reduction @<=cost={do_sr:.1f} kWh")
            print(f"    vs Uncoord:    dominated={un_dom} cost saving @<=shortfall={un_cs:.1f} $ ; "
                  f"shortfall reduction @<=cost={un_sr:.1f} kWh")

            # Representative operating point (unchanged v5 rule: ASF, 0.2/0.8)
            norm_F = (rep_F - rep_F.min(axis=0)) / (rep_F.max(axis=0) - rep_F.min(axis=0) + 1e-6)
            knee = int(ASF().do(norm_F, 1 / np.array([0.2, 0.8])).argmin())
            nsga_r = problem.simulate(rep_X[knee])
            # Tracked metric: depot peak for every point on the representative front
            rep_peak = np.array([np.max(problem.simulate(x)["net_load"]) for x in rep_X])
            rep_evse_peak = np.array([np.max(problem.simulate(x)["evses"]) for x in rep_X])
            rep_e_dep = np.array([problem.simulate(x)["e_dep"] for x in rep_X])
            rep_e_pub = np.array([problem.simulate(x)["e_pub"] for x in rep_X])
            kn = {"knee_index": knee, "knee_x": rep_X[knee],
                  "rep_seed": seeds[rep], "rep_front_F": rep_F, "rep_front_X": rep_X,
                  "rep_front_peak_load": rep_peak, "rep_front_evse_peak": rep_evse_peak,
                  "rep_front_e_dep": rep_e_dep, "rep_front_e_pub": rep_e_pub,
                  "total_trip_demand": demand}
            kn.update(res_to_dict("knee", nsga_r))
            np.savez(os.path.join(out_dir, f"ch5_{case_id}_rep_gen{n_gen}.npz"), **kn)
            print(f"    knee: cost={nsga_r['cost']:.1f} short={nsga_r['shortfall']:.1f} "
                  f"depot={100*nsga_r['e_dep']/demand:.1f}% L2={100*nsga_r['e_pub']/demand:.1f}% "
                  f"peak={np.max(nsga_r['net_load']):.1f} kW")

            for k, sd in enumerate(seeds):
                r = runs[(n_gen, sd)]
                fronts_npz[f"gen{n_gen}_seed{sd}_F"] = r["F"] if r["F"] is not None else np.zeros((0, 2))
                fronts_npz[f"gen{n_gen}_seed{sd}_X"] = r["X"] if r["X"] is not None else np.zeros((0, problem.n_var))
                fronts_npz[f"gen{n_gen}_seed{sd}_hist"] = r["hist"]
                fronts_npz[f"gen{n_gen}_seed{sd}_n_evals"] = r["n_evals"]
                fronts_npz[f"gen{n_gen}_seed{sd}_exec_time"] = r["exec_time"]
            fronts_npz[f"gen{n_gen}_hv_scenario_ref"] = hv_scn
            fronts_npz[f"gen{n_gen}_hv_budget_ref"] = hv_b
            fronts_npz[f"gen{n_gen}_ref_budget"] = ref_b
            fronts_npz[f"gen{n_gen}_hist_hv"] = hist_hv
            fronts_npz[f"gen{n_gen}_rep_seed"] = seeds[rep]

            csv_rows.append(dict(
                scenario=case_id, alpha=alpha, n_gen=n_gen, pop_size=pop_size,
                n_seeds=len(seeds), n_valid_seeds=int(valid.sum()),
                hv_mean=hv_mean, hv_std=hv_std,
                ref_cost=ref_scn[0], ref_shortfall=ref_scn[1],
                hv_mean_budget_ref=float(hv_b[valid].mean()), hv_std_budget_ref=float(hv_b[valid].std()),
                ref_cost_budget=ref_b[0], ref_shortfall_budget=ref_b[1],
                rep_seed=seeds[rep], n_front_points_rep=len(rep_F),
                depot_only_cost=do_r["cost"], depot_only_shortfall=do_r["shortfall"],
                depot_only_feasible=do_feas,
                uncoord_cost=un_r["cost"], uncoord_shortfall=un_r["shortfall"],
                uncoord_feasible_display=un_feas_display,
                depot_only_dominated=do_dom,
                depot_only_cost_saving_at_leq_shortfall=do_cs,
                depot_only_shortfall_reduction_at_leq_cost=do_sr,
                uncoord_dominated=un_dom,
                uncoord_cost_saving_at_leq_shortfall=un_cs,
                uncoord_shortfall_reduction_at_leq_cost=un_sr,
                domination_gap_depot_only_v5=gap_do, domination_gap_uncoord_v5=gap_un,
                knee_cost=nsga_r["cost"], knee_shortfall=nsga_r["shortfall"],
                knee_pct_depot=100 * nsga_r["e_dep"] / demand,
                knee_pct_public=100 * nsga_r["e_pub"] / demand,
                knee_pct_shortfall=100 * nsga_r["shortfall"] / demand,
                knee_peak_load_kw=float(np.max(nsga_r["net_load"])),
                depot_only_peak_load_kw=float(np.max(do_r["net_load"])),
                uncoord_peak_load_kw=float(np.max(un_r["net_load"])),
                front_cost_min=float(rep_F[:, 0].min()), front_cost_max=float(rep_F[:, 0].max()),
                front_shortfall_min=float(rep_F[:, 1].min()), front_shortfall_max=float(rep_F[:, 1].max()),
                mean_exec_time_s=float(np.mean([runs[(n_gen, sd)]["exec_time"] for sd in seeds])),
                per_seed_hv=";".join(f"{h:.2f}" for h in hv_scn),
            ))
            summary["budgets"][str(n_gen)] = {k: (v.item() if hasattr(v, "item") else v)
                                              for k, v in csv_rows[-1].items()}
            pareto_fronts.append((f"gen {n_gen}", rep_F))

            # v5 figures for cross-checking (unchanged functions)
            suffix = f"_gen{n_gen}" if len(budgets) > 1 else ""
            S.plot_fulfilment(case_id, [("Depot-Only", do_r, do_feas, demand),
                                        ("Uncoord.", un_r, un_feas_display, demand),
                                        ("NSGA-II", nsga_r, True, demand)],
                              os.path.join(out_dir, f"v5_fulfillment{suffix}.png"))
            S.plot_charger_utilisation(case_id, data, un_r, nsga_r,
                                       os.path.join(out_dir, f"v5_charger_utilisation{suffix}.png"))

        np.savez(os.path.join(out_dir, f"ch5_{case_id}_fronts.npz"), **fronts_npz)
        if pareto_fronts:
            S.plot_pareto(case_id, pareto_fronts, do_r, do_feas, un_r, un_feas_display,
                          os.path.join(out_dir, "v5_pareto.png"))
        with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        with open(os.path.join(out_dir, "summary.json"), "w") as fj:
            json.dump(summary, fj, indent=2, default=float)
        print(f"  wrote {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 seeds x 4 gens")
    ap.add_argument("--out", default=os.path.join(HERE, "simulation_results_ch5"))
    ap.add_argument("--allow-generate", action="store_true",
                    help="generate scenario data if no cached pickle exists (inputs will differ from the v5 run)")
    args = ap.parse_args()

    random.seed(42)   # identical to v5 __main__
    np.random.seed(42)

    SCENARIOS = [[45, 15, 15], [60, 25, 20], [90, 30, 25]]
    ALPHA = 0.70
    POP = 200
    SEEDS = [0, 1] if args.quick else [0, 1, 2, 3, 4]

    def budgets_for(N):
        if args.quick:
            return [4, 6] if N == 90 else [4]
        return [500, 800] if N == 90 else [500]

    run(SCENARIOS, ALPHA, POP, SEEDS, budgets_for, args.out,
        v5_root=os.path.join(HERE, "simulation_results"),
        allow_generate=args.allow_generate)
