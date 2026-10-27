"""Phase 4 (Problem C): graph analysis of the feeder + power-flow evaluation of graph candidates.

1. Graph metrics (nodes, edges, loops, number of radial configurations).
2. Reconfiguration: branch-exchange search over radial configurations, every candidate scored
   by AC power flow (losses) under two operating conditions; compared with the impedance-weighted
   minimum spanning tree (a pure graph heuristic) to show why graph optimality != loss optimality.
3. Battery siting: graph ranking (loss moment = downstream load x path resistance) vs AC-PF
   evaluated loss reduction for a 1 MW injection at every bus.
Outputs: results/grid_scenarios/*.csv|json, results/figures/grid_*.png
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.grid import graph as G  # noqa: E402
from src.grid import network as gn  # noqa: E402
from src.power_flow.runner import OperatingPoint, run  # noqa: E402

OUT = ROOT / "results" / "grid_scenarios"
FIG = ROOT / "results" / "figures"
CONDITIONS = {
    "peak_load_no_pv": OperatingPoint(load_scale=1.0, pv_mw=np.zeros(4)),
    "midday_pv": OperatingPoint(load_scale=0.8, pv_mw=np.full(4, 0.9 * gn.PV_RATED_MW)),
}


def eval_config(open_ids, op) -> dict:
    net = gn.base_network()
    gn.set_configuration(net, open_ids)
    if not G.is_radial(net, open_ids):
        return {"converged": False, "radial": False}
    r = run(net, op)
    r["radial"] = True
    return r


def branch_exchange(start, op, max_iter=20):
    cur = list(start)
    cur_loss = eval_config(cur, op)["p_loss_mw"]
    path = [(cur, cur_loss)]
    n_eval = 1
    for _ in range(max_iter):
        net = gn.base_network()
        gn.set_configuration(net, cur)
        best = (cur, cur_loss)
        for cand in G.one_switch_neighbours(net, cur):
            r = eval_config(cand, op)
            n_eval += 1
            if r.get("converged") and not r.get("v_violation_low", False) and r["p_loss_mw"] < best[1] - 1e-9:
                best = (cand, r["p_loss_mw"])
        if best[0] == cur:
            break
        cur, cur_loss = best
        path.append((cur, cur_loss))
    return cur, cur_loss, path, n_eval


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    net = gn.base_network()
    default_open = gn.open_lines(net)

    nodes = G.node_metrics(net)
    edges = G.edge_metrics(net)
    nodes.to_csv(OUT / "graph_node_metrics.csv")
    edges.to_csv(OUT / "graph_edge_metrics.csv")
    full = G.to_graph(net)
    summary = {"buses": full.number_of_nodes(), "lines_total": full.number_of_edges(),
               "lines_in_service": int(net.line.in_service.sum()), "tie_lines_open": default_open,
               "is_tree_default": nx.is_tree(G.radial_tree(net)),
               "independent_loops": len(nx.cycle_basis(full)),
               "radial_configurations_matrix_tree": G.count_radial_configs(net),
               "lines_without_alternative_path": edges.index[edges.no_alternative_path].tolist(),
               "loops": G.loops(net)}

    # ---- reconfiguration ----
    rec = []
    configs = {"default": default_open}
    for cname, op in CONDITIONS.items():
        base = eval_config(default_open, op)
        best, loss, path, n_eval = branch_exchange(default_open, op)
        r = eval_config(best, op)
        configs[f"best_{cname}"] = best
        rec.append({"condition": cname, "config": "default", "open_lines": default_open, **{k: base[k] for k in
                    ["p_loss_mw", "vm_min_pu", "vm_max_pu", "line_loading_max_pct", "p_recv_mw"]}})
        rec.append({"condition": cname, "config": "branch_exchange_best", "open_lines": best, "pf_evaluations": n_eval,
                    "iterations": len(path) - 1, **{k: r[k] for k in ["p_loss_mw", "vm_min_pu", "vm_max_pu",
                                                                      "line_loading_max_pct", "p_recv_mw"]}})
    # Pure graph heuristic: minimum spanning tree on resistance (ignores where load and generation are).
    mst = nx.minimum_spanning_tree(full, weight="r_ohm")
    mst_lines = {d["line"] for _, _, d in mst.edges(data=True)}
    mst_open = sorted(set(net.line.index) - mst_lines)
    configs["min_spanning_tree_r"] = mst_open
    for cname, op in CONDITIONS.items():
        r = eval_config(mst_open, op)
        rec.append({"condition": cname, "config": "min_spanning_tree_r", "open_lines": mst_open,
                    **{k: r[k] for k in ["p_loss_mw", "vm_min_pu", "vm_max_pu", "line_loading_max_pct", "p_recv_mw"]}})
    rec = pd.DataFrame(rec)
    for cname in CONDITIONS:
        b = rec[(rec.condition == cname) & (rec.config == "default")].p_loss_mw.iloc[0]
        rec.loc[rec.condition == cname, "loss_change_vs_default_pct"] = 100 * (rec.p_loss_mw / b - 1)
    rec["open_lines"] = rec.open_lines.map(lambda x: "-".join(map(str, x)))
    rec.to_csv(OUT / "reconfiguration.csv", index=False)

    # ---- battery siting ----
    sit = []
    for cname, op in CONDITIONS.items():
        base = eval_config(default_open, op)["p_loss_mw"]
        for b in range(1, 33):
            n2 = gn.base_network()
            gn.set_battery_bus(n2, b)
            op2 = OperatingPoint(op.load_scale, op.pv_mw, battery_mw=1.0 if cname == "peak_load_no_pv" else -1.0)
            r = run(n2, op2)
            sit.append({"condition": cname, "bus": b, "battery_mw": op2.battery_mw,
                        "loss_reduction_kw": 1e3 * (base - r["p_loss_mw"]), "vm_min_pu": r["vm_min_pu"],
                        "vm_max_pu": r["vm_max_pu"]})
    sit = pd.DataFrame(sit).merge(nodes[["loss_moment", "r_dist_ohm", "betweenness_z", "downstream_load_mw"]],
                                  left_on="bus", right_index=True)
    sit.to_csv(OUT / "battery_siting.csv", index=False)
    corr = {}
    for cname, g in sit.groupby("condition"):
        corr[cname] = {m: float(spearmanr(g[m], g["loss_reduction_kw"]).statistic)
                       for m in ["loss_moment", "r_dist_ohm", "betweenness_z"]}
    # Choose the battery bus that maximises the summed loss reduction over both conditions
    # (discharging at peak, charging from PV at midday).
    tot = sit.groupby("bus")["loss_reduction_kw"].sum()
    battery_bus = int(tot.idxmax())
    graph_top5 = nodes.sort_values("loss_moment", ascending=False).head(5).index.tolist()
    summary.update({"battery_bus_selected": battery_bus, "battery_bus_selection": "max summed AC-PF loss reduction",
                    "graph_top5_by_loss_moment": graph_top5, "spearman_graph_metric_vs_pf_loss_reduction": corr,
                    "battery_bus_total_loss_reduction_kw": float(tot.max())})
    (OUT / "grid_summary.json").write_text(json.dumps(summary, indent=2, default=int))

    # configurations for the receiving-end scenario dataset: default, both branch-exchange optima,
    # MST, plus two random radial neighbours of the default
    rng = np.random.default_rng(0)
    neigh = G.one_switch_neighbours(net, default_open)
    extra = [neigh[i] for i in rng.choice(len(neigh), 3, replace=False)]
    cfg_list = []
    for c in list(configs.values()) + extra:
        if sorted(c) not in cfg_list and G.is_radial(net, c):
            cfg_list.append(sorted(c))
    (OUT / "grid_decisions.json").write_text(json.dumps({"battery_bus": battery_bus, "configs": cfg_list,
                                                         "config_names": list(configs.keys()) + ["random_neighbour"] * 3},
                                                        indent=2, default=int))
    plot_graph(net, nodes, battery_bus, configs)
    plot_siting(sit, nodes)
    print(json.dumps(summary, indent=1, default=int))
    print(rec.round(4).to_string())


def plot_graph(net, nodes, battery_bus, configs):
    full = G.to_graph(net)
    pos = {b: tuple(json.loads(net.bus.at[b, "geo"])["coordinates"]) for b in net.bus.index}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, (title, open_ids) in zip(axes, [("Default radial configuration", configs["default"]),
                                            ("Loss-minimising configuration (peak load, AC PF)", configs["best_peak_load_no_pv"])]):
        closed = [(u, v) for u, v, d in full.edges(data=True) if d["line"] not in open_ids]
        opened = [(u, v) for u, v, d in full.edges(data=True) if d["line"] in open_ids]
        nx.draw_networkx_edges(full, pos, edgelist=closed, ax=ax, width=2.2, edge_color="#444")
        nx.draw_networkx_edges(full, pos, edgelist=opened, ax=ax, width=1.2, style="dashed", edge_color="#bbb")
        col = nodes["r_dist_ohm"].reindex(list(full.nodes))
        nc = nx.draw_networkx_nodes(full, pos, ax=ax, node_color=col, cmap="viridis", node_size=260)
        nx.draw_networkx_labels(full, pos, ax=ax, font_size=7, font_color="white")
        nx.draw_networkx_nodes(full, pos, nodelist=gn.PV_BUSES, ax=ax, node_shape="s", node_color="none",
                               edgecolors="#e69f00", linewidths=2.5, node_size=420)
        nx.draw_networkx_nodes(full, pos, nodelist=[battery_bus], ax=ax, node_shape="D", node_color="none",
                               edgecolors="#d62728", linewidths=2.5, node_size=420)
        nx.draw_networkx_nodes(full, pos, nodelist=[gn.RECEIVING_BUS], ax=ax, node_shape="*", node_color="#d62728", node_size=500)
        ax.set_title(f"{title}\nopen lines: {sorted(open_ids)}", fontsize=10)
        ax.axis("off")
    fig.colorbar(nc, ax=axes, label="resistance distance to receiving bus 0 [ohm] (default config)", shrink=0.7)
    fig.suptitle("33-bus benchmark feeder as a graph (SIMULATED). star = receiving bus 0, orange square = PV, red diamond = battery, dashed = open line")
    fig.savefig(FIG / "grid_graph.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_siting(sit, nodes):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (cname, g) in zip(axes, sit.groupby("condition")):
        ax.scatter(g["loss_moment"], g["loss_reduction_kw"], c=g["r_dist_ohm"], cmap="viridis")
        for _, r in g.iterrows():
            ax.annotate(int(r.bus), (r.loss_moment, r.loss_reduction_kw), fontsize=7)
        ax.set(xlabel="graph metric: downstream load x path resistance [MW.ohm]",
               ylabel="AC-PF loss reduction [kW]", title=f"1 MW battery {'discharge' if 'peak' in cname else 'charge'} - {cname}")
    fig.tight_layout()
    fig.savefig(FIG / "grid_battery_siting.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
