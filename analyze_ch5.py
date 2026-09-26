"""Numbers for Chapter 5 §5.6 text, computed from the saved .npz arrays only."""
import glob, os, sys, json
import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "simulation_results_ch5"
out = {}
for d in sorted(glob.glob(os.path.join(R, "N*_F*_M*"))):
    if not os.path.exists(os.path.join(d, f"ch5_{os.path.basename(d)}_fronts.npz")):
        continue
    cid = os.path.basename(d)
    fr = np.load(os.path.join(d, f"ch5_{cid}_fronts.npz"))
    bl = np.load(os.path.join(d, f"ch5_{cid}_baselines.npz"))
    ins = np.load(os.path.join(d, f"ch5_{cid}_inputs.npz"))
    N, F, M = int(ins["N"]), int(ins["F"]), int(ins["M"])
    demand = float(ins["total_trip_demand"])
    avail = ins["pub_availability"]  # (T, F)
    dwells = ins["dwells"]  # (N, 3, 2)
    o = {"N": N, "F": F, "M": M, "p_site": float(ins["p_site_max"]), "demand": demand,
         "initial_soc_kwh": float(ins["initial_soc_kwh"]),
         "max_concurrent_by_site": int(ins["p_site_max"] // 22.0)}
    budgets = [int(b) for b in fr["budgets"]]
    for g in budgets:
        rep = np.load(os.path.join(d, f"ch5_{cid}_rep_gen{g}.npz"))
        F_ = rep["rep_front_F"]; X_ = np.round(rep["rep_front_X"]).astype(int)
        order = np.argsort(F_[:, 0]); F_ = F_[order]; X_ = X_[order]
        og = {}
        og["n_points"] = len(F_)
        og["cost_range"] = [float(F_[0, 0]), float(F_[-1, 0])]
        og["short_range"] = [float(F_[:, 1].max()), float(F_[:, 1].min())]
        og["short_min_pct"] = 100 * float(F_[:, 1].min()) / demand
        og["short_max_pct"] = 100 * float(F_[:, 1].max()) / demand
        # marginal cost per kWh of shortfall removed: piecewise along the front
        dc = np.diff(F_[:, 0]); ds = -np.diff(F_[:, 1])
        m = ds > 1e-6
        slope = dc[m] / ds[m]
        # quartiles of the front by cost
        q = np.array_split(np.arange(len(slope)), 4)
        og["marginal_cost_per_kwh_quartiles"] = [float(np.median(slope[qq])) for qq in q if len(qq)]
        og["avg_cost_per_kwh_removed_whole_front"] = float((F_[-1, 0] - F_[0, 0]) / (F_[0, 1] - F_[-1, 1]))
        # genes by window: 0 skip,1 depot,>=2 public
        G = X_.reshape(len(X_), N, 3)
        og["pct_evening_skip"] = 100 * float((G[:, :, 2] == 0).mean())
        og["pct_morning_depot"] = 100 * float((G[:, :, 0] == 1).mean())
        og["pct_mid_public_cheapest"] = 100 * float((G[0, :, 1] >= 2).mean())
        og["pct_mid_public_mostreliable"] = 100 * float((G[-1, :, 1] >= 2).mean())
        og["pct_mid_public_all_front"] = 100 * float((G[:, :, 1] >= 2).mean())
        og["pct_mid_depot_all_front"] = 100 * float((G[:, :, 1] == 1).mean())
        og["pct_morning_public_all_front"] = 100 * float((G[:, :, 0] >= 2).mean())
        # knee
        kx = np.round(rep["knee_x"]).astype(int).reshape(N, 3)
        og["knee"] = {"cost": float(rep["knee_cost"]), "short": float(rep["knee_shortfall"]),
                      "pct_depot": 100 * float(rep["knee_e_dep"]) / demand,
                      "pct_pub": 100 * float(rep["knee_e_pub"]) / demand,
                      "pct_short": 100 * float(rep["knee_shortfall"]) / demand,
                      "peak_kw": float(rep["knee_peak_load"]),
                      "n_public_mid": int((kx[:, 1] >= 2).sum()), "n_depot_mid": int((kx[:, 1] == 1).sum()),
                      "n_skip_mid": int((kx[:, 1] == 0).sum()),
                      "n_depot_morning": int((kx[:, 0] == 1).sum()), "n_skip_morning": int((kx[:, 0] == 0).sum()),
                      "n_public_morning": int((kx[:, 0] >= 2).sum()),
                      "n_evening_skip": int((kx[:, 2] == 0).sum()), "n_evening_depot": int((kx[:, 2] == 1).sum()),
                      "n_evening_public": int((kx[:, 2] >= 2).sum())}
        # unavailable public sessions at knee (sent to a station offline at start slot)
        wasted = 0
        for i in range(N):
            for j in range(3):
                c = kx[i, j]
                if c >= 2 and avail[int(dwells[i, j, 0]), c - 2] == 0:
                    wasted += 1
        og["knee"]["public_sessions_at_offline_station"] = wasted
        og["knee"]["stations_used"] = int(len(set(kx[kx >= 2].tolist())))
        occ = rep["knee_conn_occ"]; s0, s1 = 36, 72
        og["knee"]["stations_over_2"] = int((occ.max(1) > 2).sum())
        og["knee"]["max_concurrent_any_station"] = int(occ.max())
        og["knee"]["max_evse_concurrent"] = int(rep["knee_evses"].max())
        # peak along front
        pk = rep["rep_front_peak_load"][order]
        og["peak_range"] = [float(pk.min()), float(pk.max())]
        og["pct_front_at_site_limit"] = 100 * float((pk >= float(ins["p_site_max"]) - 22.5).mean())
        og["evse_peak_range"] = [int(rep["rep_front_evse_peak"].min()), int(rep["rep_front_evse_peak"].max())]
        hv = fr[f"gen{g}_hv_scenario_ref"]
        og["hv_mean"] = float(hv.mean()); og["hv_std"] = float(hv.std()); og["hv_cv_pct"] = 100 * float(hv.std() / hv.mean())
        hb = fr[f"gen{g}_hv_budget_ref"]
        og["hv_budget_ref_mean"] = float(hb.mean()); og["hv_budget_ref_std"] = float(hb.std())
        H = fr[f"gen{g}_hist_hv"]; m_ = H.mean(0)
        for thr in (0.90, 0.95, 0.99, 0.999):
            og[f"gen_at_{thr}"] = int(np.argmax(m_ >= thr * m_[-1]) + 1)
        og["hv_gen1_frac"] = float(m_[0] / m_[-1]); og["hv_gen50_frac"] = float(m_[min(49, len(m_)-1)] / m_[-1])
        og["hv_gen100_frac"] = float(m_[min(99, len(m_)-1)] / m_[-1])
        og["hv_last100_gain_pct"] = 100 * float((m_[-1] - m_[-101]) / m_[-1]) if len(m_) > 100 else None
        og["mean_exec_time_s"] = float(np.mean([fr[f"gen{g}_seed{int(s)}_exec_time"] for s in fr["seeds"]]))
        og["n_evals"] = int(fr[f"gen{g}_seed0_n_evals"])
        # extreme-point spread across seeds
        seeds_F = [fr[f"gen{g}_seed{int(s)}_F"] for s in fr["seeds"]]
        og["seed_cost_min"] = [float(f[:, 0].min()) for f in seeds_F]
        og["seed_short_min"] = [float(f[:, 1].min()) for f in seeds_F]
        og["seed_n_points"] = [int(len(f)) for f in seeds_F]
        o[f"gen{g}"] = og
    # baselines
    o["depot_only"] = {"cost": float(bl["depot_only_cost"]), "short": float(bl["depot_only_shortfall"]),
                       "pct_short": 100 * float(bl["depot_only_shortfall"]) / demand,
                       "pct_depot": 100 * float(bl["depot_only_e_dep"]) / demand,
                       "peak": float(bl["depot_only_peak_load"]), "max_evse": int(bl["depot_only_evses"].max()),
                       "e_depot_charged": float(bl["depot_only_e_depot_charged"])}
    dx = np.round(bl["depot_only_x"]).astype(int).reshape(N, 3)
    o["depot_only"]["n_windows_depot"] = [int((dx[:, j] == 1).sum()) for j in range(3)]
    o["uncoord"] = {"cost": float(bl["uncoord_cost"]), "short": float(bl["uncoord_shortfall"]),
                    "pct_short": 100 * float(bl["uncoord_shortfall"]) / demand,
                    "pct_depot": 100 * float(bl["uncoord_e_dep"]) / demand,
                    "pct_pub": 100 * float(bl["uncoord_e_pub"]) / demand,
                    "oversub": bool(bl["uncoord_conn_oversubscribed"]),
                    "max_concurrent_any_station": int(bl["uncoord_conn_occ"].max()),
                    "stations_over_2": int((bl["uncoord_conn_occ"].max(1) > 2).sum()),
                    "e_L2_charged": float(bl["uncoord_e_L2_charged"])}
    # uncoord detour+wait cost share: 3 sessions per vehicle at c_d=0.35
    # cost of detour = total cost - energy cost; energy cost = e_L2/0.95 * price at start
    o["uncoord"]["energy_cost_est"] = None
    # if gains
    import csv
    rows = list(csv.DictReader(open(os.path.join(d, "metrics.csv"))))
    o["gains"] = {r["n_gen"]: {k: r[k] for k in r if "saving" in k or "reduction" in k or "dominated" in k} for r in rows}
    out[cid] = o

json.dump(out, open(os.path.join(R, "ch5_text_numbers.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
