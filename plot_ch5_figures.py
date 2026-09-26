# pylint: disable=missing-module-docstring,missing-function-docstring
"""
Chapter 5 (RQ3) thesis figures, generated ONLY from the .npz arrays written by
run_ch5_thesis.py (no re-simulation, no placeholder data).

Usage:
    uv run python plot_ch5_figures.py --results simulation_results_ch5 --out figures

Writes figures/ch5_*.png at 300 DPI in the Chapter 3/4 style (no in-figure
titles; captions live in the .tex). Also writes figures/ch5_figure_numbers.csv
with every number that appears in a figure, for cross-checking the text.
"""

import argparse
import csv
import glob
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

# MATLAB default palette, as used in the Chapter 3/4 figures
C_FRONT = "#0072BD"   # blue   — NSGA-II front
C_FRONT2 = "#77AC30"  # green  — 800-generation front
C_DO = "#D95319"      # orange — Depot-Only Greedy
C_UN = "#7E2F8E"      # purple — Uncoordinated Immediate
C_DEP = "#0072BD"     # depot energy
C_PUB = "#EDB120"     # public energy
C_SHORT = "#D95319"   # shortfall
C_AVAIL, C_PART, C_FULL = "#E6E6E6", "#EDB120", "#D95319"
SCN = {"N45": "#0072BD", "N60": "#D95319", "N90": "#77AC30"}
LW = 2.0
MS = 4.5


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 11, "axes.labelsize": 12,
        "axes.linewidth": 1.0, "legend.fontsize": 10, "legend.frameon": True,
        "legend.framealpha": 1.0, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "lines.linewidth": LW,
        "figure.dpi": 300, "savefig.dpi": 300,
    })


def load_case(results_dir, case_id):
    d = os.path.join(results_dir, case_id)
    fr = np.load(os.path.join(d, f"ch5_{case_id}_fronts.npz"))
    bl = np.load(os.path.join(d, f"ch5_{case_id}_baselines.npz"))
    ins = np.load(os.path.join(d, f"ch5_{case_id}_inputs.npz"))
    reps = {}
    for p in sorted(glob.glob(os.path.join(d, f"ch5_{case_id}_rep_gen*.npz"))):
        g = int(os.path.basename(p).split("_gen")[1].split(".")[0])
        reps[g] = np.load(p)
    return fr, bl, ins, reps


def short_label(case_id):
    n = case_id.split("_")[0][1:]
    return f"N={n}"


# ---------------------------------------------------------------------------
def fig_pareto(case_id, fr, bl, reps, out, rows):
    budgets = [int(b) for b in fr["budgets"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    cols = [C_FRONT, C_FRONT2]
    for k, g in enumerate(budgets):
        F = reps[g]["rep_front_F"]
        F = F[np.argsort(F[:, 0])]
        lbl = f"NSGA-II front ({g} generations)" if len(budgets) > 1 else "NSGA-II front"
        ax.plot(F[:, 0], F[:, 1], "o-", ms=MS, lw=LW, color=cols[k], label=lbl, zorder=3)
        rows.append([case_id, f"front_gen{g}", "n_points", len(F)])
        rows.append([case_id, f"front_gen{g}", "cost_min", F[:, 0].min()])
        rows.append([case_id, f"front_gen{g}", "cost_max", F[:, 0].max()])
        rows.append([case_id, f"front_gen{g}", "shortfall_min", F[:, 1].min()])
        rows.append([case_id, f"front_gen{g}", "shortfall_max", F[:, 1].max()])
        # knee marker on the first budget only (the one used in the fulfilment figure)
        if k == 0:
            kx = reps[g]["knee_cost"]; ky = reps[g]["knee_shortfall"]
            ax.plot(kx, ky, "D", ms=10, mfc="white", mec=cols[k], mew=2.2,
                    label="Selected operating point (0.2/0.8)", zorder=5)
            rows.append([case_id, f"knee_gen{g}", "cost", float(kx)])
            rows.append([case_id, f"knee_gen{g}", "shortfall", float(ky)])
    do_feas = bool(bl["depot_only_feasible"]); un_feas = bool(bl["uncoord_feasible_display"])
    ax.plot(bl["depot_only_cost"], bl["depot_only_shortfall"], "s", ms=12,
            mfc=C_DO if do_feas else "white", mec=C_DO, mew=2.2,
            label="Depot-Only Greedy" + ("" if do_feas else " (infeasible)"), zorder=4)
    ax.plot(bl["uncoord_cost"], bl["uncoord_shortfall"], "^", ms=12,
            mfc=C_UN if un_feas else "white", mec=C_UN, mew=2.2,
            label="Uncoordinated Immediate" + ("" if un_feas else " (connectors over-subscribed)"), zorder=4)
    rows += [[case_id, "depot_only", "cost", float(bl["depot_only_cost"])],
             [case_id, "depot_only", "shortfall", float(bl["depot_only_shortfall"])],
             [case_id, "uncoord", "cost", float(bl["uncoord_cost"])],
             [case_id, "uncoord", "shortfall", float(bl["uncoord_shortfall"])]]
    ax.set_xlabel("Total charging cost $f_1$ (AUD)")
    ax.set_ylabel("Fleet energy shortfall $f_2$ (kWh)")
    ax.grid(True, alpha=0.3, lw=0.8)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)


def fig_fulfilment(case_id, bl, rep, out, rows):
    demand = float(bl["total_trip_demand"])
    items = [("Depot-Only\nGreedy", bl["depot_only_e_dep"], bl["depot_only_e_pub"], bl["depot_only_shortfall"], bool(bl["depot_only_feasible"])),
             ("Uncoordinated\nImmediate", bl["uncoord_e_dep"], bl["uncoord_e_pub"], bl["uncoord_shortfall"], bool(bl["uncoord_feasible_display"])),
             ("NSGA-II\n(selected point)", rep["knee_e_dep"], rep["knee_e_pub"], rep["knee_shortfall"], True)]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, (name, ed, ep, sh, feas) in enumerate(items):
        d, p, s = 100 * float(ed) / demand, 100 * float(ep) / demand, 100 * float(sh) / demand
        ax.bar(i, d, color=C_DEP, edgecolor="white", width=0.62)
        ax.bar(i, p, bottom=d, color=C_PUB, edgecolor="white", width=0.62)
        ax.bar(i, s, bottom=d + p, color=C_SHORT, edgecolor="white", width=0.62,
               hatch="//" if not feas else None)
        for val, bot, col, seg in ((d, 0, "white", "d"), (p, d, "black", "p"), (s, d + p, "white", "s")):
            if val >= 5:
                kw = dict(ha="center", va="center", fontsize=10, fontweight="bold", color=col)
                if seg == "s" and not feas:
                    kw.update(color=C_SHORT, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
                ax.text(i, bot + val / 2, f"{val:.0f}%", **kw)
        ax.text(i, 102, f"{d + p:.0f}% delivered", ha="center", fontsize=10, fontweight="bold")
        if not feas:
            ax.text(i, -6, "(connectors over-subscribed)", ha="center", fontsize=9,
                    color=C_SHORT, style="italic")
        tag = ["depot_only", "uncoord", "knee"][i]
        rows += [[case_id, tag, "pct_depot", d], [case_id, tag, "pct_public", p],
                 [case_id, tag, "pct_shortfall", s]]
    ax.set_xticks(range(3)); ax.set_xticklabels([it[0] for it in items])
    ax.set_ylabel("Share of trip energy demand (%)")
    ax.set_ylim(-10, 112); ax.axhline(100, color="black", ls="--", lw=1.0)
    ax.legend(handles=[mpatches.Patch(color=C_DEP, label="Depot"),
                       mpatches.Patch(color=C_PUB, label="Public L2"),
                       mpatches.Patch(color=C_SHORT, label="Shortfall")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_utilisation(case_id, ins, rep, out, rows):
    dt = float(ins["dt"]); F = int(ins["F"]); n_conn = int(ins["n_conn"][0])
    s0, s1 = int(9 / dt), int(18 / dt)
    occ = rep["knee_conn_occ"][:, s0:s1]
    avail = (occ == 0).sum(1) * dt
    part = ((occ >= 1) & (occ < n_conn)).sum(1) * dt
    full = (occ >= n_conn).sum(1) * dt
    fig, ax = plt.subplots(figsize=(6.4, 0.22 * F + 1.8))
    y = np.arange(F)
    ax.barh(y, avail, color=C_AVAIL, edgecolor="#404040", lw=0.6, height=0.72)
    ax.barh(y, part, left=avail, color=C_PART, edgecolor="#404040", lw=0.6, height=0.72)
    ax.barh(y, full, left=avail + part, color=C_FULL, edgecolor="#404040", lw=0.6, height=0.72)
    ax.set_yticks(y); ax.set_yticklabels([f"S{f + 1}" for f in range(F)], fontsize=9)
    ax.set_xlim(0, (s1 - s0) * dt); ax.set_ylim(-0.6, F - 0.4)
    ax.set_xlabel("Hours in state, 09:00–18:00"); ax.set_ylabel("Public station")
    ax.grid(True, axis="x", alpha=0.25, lw=0.5); ax.set_axisbelow(True)
    ax.legend(handles=[mpatches.Patch(facecolor=C_AVAIL, edgecolor="#404040", label="Available"),
                       mpatches.Patch(facecolor=C_PART, edgecolor="#404040", label="One fleet vehicle"),
                       mpatches.Patch(facecolor=C_FULL, edgecolor="#404040", label="Two or more fleet vehicles")],
              loc="upper left", bbox_to_anchor=(0.0, -1.0 / (0.22 * F + 1.8)), ncol=3, frameon=False,
              borderaxespad=0)
    rows += [[case_id, "utilisation", "stations_unused", int((avail >= (s1 - s0) * dt - 1e-9).sum())],
             [case_id, "utilisation", "stations_any_full", int((full > 0).sum())],
             [case_id, "utilisation", "mean_full_hours", float(full.mean())],
             [case_id, "utilisation", "mean_partial_hours", float(part.mean())]]
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_convergence(cases, out, rows):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    cols = SCN
    for case_id, (fr, bl, ins, reps) in cases.items():
        key = case_id.split("_")[0]
        for g in [int(b) for b in fr["budgets"]]:
            H = fr[f"gen{g}_hist_hv"]  # (n_seeds, n_gen)
            ref_final = fr[f"gen{g}_hv_scenario_ref"].mean()
            Hn = H / ref_final
            m, s = Hn.mean(0), Hn.std(0)
            x = np.arange(1, H.shape[1] + 1)
            ls = "-" if g == min(int(b) for b in fr["budgets"]) else "--"
            ax.plot(x, m, ls, color=cols.get(key, "k"), lw=LW, label=f"{short_label(case_id)}, {g} generations")
            ax.fill_between(x, m - s, m + s, color=cols.get(key, "k"), alpha=0.18, lw=0)
            # generation at which mean reaches 95 % / 99 % of its final value
            for thr in (0.95, 0.99):
                idx = np.argmax(m >= thr * m[-1]) + 1
                rows.append([case_id, f"convergence_gen{g}", f"gen_at_{int(thr*100)}pct", int(idx)])
    ax.set_xlabel("Generation"); ax.set_ylabel("Hypervolume / final mean hypervolume")
    ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3, lw=0.8); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_peak(cases, out, rows):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    cols = SCN
    for case_id, (fr, bl, ins, reps) in cases.items():
        key = case_id.split("_")[0]
        g = min(int(b) for b in fr["budgets"])
        F = reps[g]["rep_front_F"]; pk = reps[g]["rep_front_peak_load"]
        o = np.argsort(F[:, 0])
        ax.plot(F[o, 0], pk[o], "o", ms=MS, color=cols.get(key, "k"), alpha=0.85, label=short_label(case_id))
        ax.axhline(float(ins["p_site_max"]), color=cols.get(key, "k"), ls="--", lw=1.6)
        rows += [[case_id, f"peak_gen{g}", "peak_min_kw", float(pk.min())],
                 [case_id, f"peak_gen{g}", "peak_max_kw", float(pk.max())],
                 [case_id, f"peak_gen{g}", "p_site_max_kw", float(ins["p_site_max"])],
                 [case_id, "depot_only", "peak_kw", float(bl["depot_only_peak_load"])],
                 [case_id, "knee", "peak_kw", float(reps[g]["knee_peak_load"])]]
    ax.set_xlabel("Total charging cost $f_1$ (AUD)"); ax.set_ylabel("Depot peak power import (kW)")
    ax.grid(True, alpha=0.3, lw=0.8); ax.legend(loc="lower right", title="Front points; dashed = site limit")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_seed_fronts(case_id, fr, out, rows):
    """All seeds' final fronts overlaid, one budget per panel."""
    budgets = [int(b) for b in fr["budgets"]]
    fig, axes = plt.subplots(1, len(budgets), figsize=(6.4, 3.6), sharey=True, squeeze=False)
    for ax, g in zip(axes[0], budgets):
        for sd in fr["seeds"]:
            F = fr[f"gen{g}_seed{int(sd)}_F"]
            if len(F) == 0:
                continue
            F = F[np.argsort(F[:, 0])]
            ax.plot(F[:, 0], F[:, 1], "-", lw=0.9, alpha=0.85, label=f"seed {int(sd)}")
        hv = fr[f"gen{g}_hv_scenario_ref"]
        ax.set_title(f"{g} generations; HV = {hv.mean():.3g} ± {hv.std():.2g}", fontsize=9)
        ax.set_xlabel("Cost $f_1$ (AUD)"); ax.grid(True, alpha=0.25)
        rows += [[case_id, f"seeds_gen{g}", "hv_mean", float(hv.mean())],
                 [case_id, f"seeds_gen{g}", "hv_std", float(hv.std())],
                 [case_id, f"seeds_gen{g}", "hv_cv_pct", float(100 * hv.std() / hv.mean())]]
    axes[0][0].set_ylabel("Shortfall $f_2$ (kWh)"); axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="simulation_results_ch5")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--seed-fronts", action="store_true", help="also write the per-seed front overlays (not used in the chapter)")
    args = ap.parse_args()
    style()
    os.makedirs(args.out, exist_ok=True)
    case_ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.results, "N*_F*_M*")))
    cases = {c: load_case(args.results, c) for c in case_ids}
    rows = []
    for case_id, (fr, bl, ins, reps) in cases.items():
        n = case_id.split("_")[0].lower()
        g0 = min(int(b) for b in fr["budgets"])
        fig_pareto(case_id, fr, bl, reps, os.path.join(args.out, f"ch5_pareto_{n}.png"), rows)
        fig_fulfilment(case_id, bl, reps[g0], os.path.join(args.out, f"ch5_fulfilment_{n}.png"), rows)
        fig_utilisation(case_id, ins, reps[g0], os.path.join(args.out, f"ch5_utilisation_{n}.png"), rows)
        if args.seed_fronts:
            fig_seed_fronts(case_id, fr, os.path.join(args.out, f"ch5_seed_fronts_{n}.png"), rows)
    fig_convergence(cases, os.path.join(args.out, "ch5_convergence.png"), rows)
    fig_peak(cases, os.path.join(args.out, "ch5_peak_demand.png"), rows)
    with open(os.path.join(args.out, "ch5_figure_numbers.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scenario", "item", "quantity", "value"]); w.writerows(rows)
    print(f"wrote {len(glob.glob(os.path.join(args.out, 'ch5_*.png')))} figures to {args.out}")


if __name__ == "__main__":
    main()
