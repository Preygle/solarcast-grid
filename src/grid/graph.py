"""Graph representation and topology analysis of the feeder (NetworkX).

Nodes = buses (attrs: base load P/Q, PV/battery flags, voltage limits).
Edges = lines incl. normally-open tie lines (attrs: r, x, |z| [ohm], rating, tie flag, in_service).

Graph metrics are used to IDENTIFY candidates; every candidate is EVALUATED with AC power flow,
because shortest paths / spanning trees do not by themselves minimise electrical losses.
"""
from __future__ import annotations


import networkx as nx
import numpy as np
import pandas as pd

from src.grid.network import PV_BUSES, RECEIVING_BUS, TIE_LINES


def to_graph(net, include_open: bool = True) -> nx.Graph:
    g = nx.Graph()
    load_p = net.load.groupby("bus")["p_base_mw"].sum()
    load_q = net.load.groupby("bus")["q_base_mvar"].sum()
    for b in net.bus.index:
        g.add_node(int(b), load_mw=float(load_p.get(b, 0.0)), load_mvar=float(load_q.get(b, 0.0)),
                   pv=b in PV_BUSES, receiving=b == RECEIVING_BUS,
                   vmin=float(net.bus.at[b, "min_vm_pu"]), vmax=float(net.bus.at[b, "max_vm_pu"]))
    for i, ln in net.line.iterrows():
        if not include_open and not ln.in_service:
            continue
        r = ln.r_ohm_per_km * ln.length_km
        x = ln.x_ohm_per_km * ln.length_km
        g.add_edge(int(ln.from_bus), int(ln.to_bus), line=int(i), r_ohm=r, x_ohm=x, z_ohm=float(np.hypot(r, x)),
                   max_i_ka=float(ln.max_i_ka), tie=int(i) in TIE_LINES, in_service=bool(ln.in_service))
    return g


def radial_tree(net) -> nx.Graph:
    return to_graph(net, include_open=False)


def node_metrics(net) -> pd.DataFrame:
    """Per-bus topology metrics in the current (radial) configuration."""
    t = radial_tree(net)
    dist_z = nx.single_source_dijkstra_path_length(t, RECEIVING_BUS, weight="z_ohm")
    dist_r = nx.single_source_dijkstra_path_length(t, RECEIVING_BUS, weight="r_ohm")
    hops = nx.single_source_shortest_path_length(t, RECEIVING_BUS)
    btw = nx.betweenness_centrality(t, weight="z_ohm", normalized=True)
    # Downstream load: load fed through each bus (sum over its subtree away from bus 0).
    parent = nx.bfs_predecessors(t, RECEIVING_BUS)
    children: dict[int, list[int]] = {}
    for c, p in parent:
        children.setdefault(p, []).append(c)

    def subtree_load(b: int) -> float:
        return t.nodes[b]["load_mw"] + sum(subtree_load(c) for c in children.get(b, []))

    df = pd.DataFrame({
        "bus": list(t.nodes),
        "hops_to_receiving": [hops[b] for b in t.nodes],
        "z_dist_ohm": [dist_z[b] for b in t.nodes],
        "r_dist_ohm": [dist_r[b] for b in t.nodes],
        "betweenness_z": [btw[b] for b in t.nodes],
        "degree": [t.degree(b) for b in t.nodes],
        "load_mw": [t.nodes[b]["load_mw"] for b in t.nodes],
        "downstream_load_mw": [subtree_load(b) for b in t.nodes],
    }).set_index("bus").sort_index()
    # Loss-weighted moment: downstream load x path resistance (proxy for I^2R reduction potential
    # of injecting power at the bus).
    df["loss_moment"] = df["downstream_load_mw"] * df["r_dist_ohm"]
    return df


def edge_metrics(net) -> pd.DataFrame:
    full = to_graph(net, include_open=True)
    t = radial_tree(net)
    ebt = nx.edge_betweenness_centrality(t, weight="z_ohm", normalized=True)
    bridges_full = {tuple(sorted(e)) for e in nx.bridges(full)}
    rows = []
    for u, v, d in full.edges(data=True):
        key = tuple(sorted((u, v)))
        rows.append({"line": d["line"], "from": u, "to": v, "r_ohm": d["r_ohm"], "x_ohm": d["x_ohm"],
                     "tie": d["tie"], "in_service": d["in_service"],
                     "edge_betweenness": ebt.get((u, v), ebt.get((v, u), np.nan)),
                     # A bridge of the meshed graph has no alternative path even if ties close.
                     "no_alternative_path": key in bridges_full})
    return pd.DataFrame(rows).set_index("line").sort_index()


def loops(net) -> list[list[int]]:
    """Fundamental loops created by closing each tie line, as lists of line indices."""
    t = radial_tree(net)
    full = to_graph(net, include_open=True)
    out = []
    for u, v, d in full.edges(data=True):
        if d["in_service"]:
            continue
        path = nx.shortest_path(t, u, v)
        out.append([d["line"]] + [t.edges[a, b]["line"] for a, b in zip(path[:-1], path[1:])])
    return out


def is_radial(net, open_ids: list[int]) -> bool:
    full = to_graph(net, include_open=True)
    g = nx.Graph()
    g.add_nodes_from(full.nodes)
    g.add_edges_from((u, v) for u, v, d in full.edges(data=True) if d["line"] not in set(open_ids))
    return nx.is_tree(g)


def count_radial_configs(net) -> int:
    """Number of spanning trees of the meshed graph (Kirchhoff's matrix-tree theorem)."""
    full = to_graph(net, include_open=True)
    lap = nx.laplacian_matrix(full).toarray().astype(float)
    return int(round(np.linalg.det(lap[1:, 1:])))


def one_switch_neighbours(net, open_ids: list[int]) -> list[list[int]]:
    """Branch-exchange neighbours: close one open line and open another line in its loop."""
    import copy
    from src.grid.network import set_configuration
    out = []
    for o in open_ids:
        test = copy.deepcopy(net)
        set_configuration(test, [i for i in open_ids if i != o])
        t = to_graph(test, include_open=False)  # closing line o creates exactly one loop
        loop_lines = [t.edges[a, b]["line"] for a, b in nx.find_cycle(t)]
        for cand in loop_lines:
            if cand == o:
                continue
            cfg = sorted([i for i in open_ids if i != o] + [cand])
            out.append(cfg)
    uniq = []
    for c in out:
        if c not in uniq:
            uniq.append(c)
    return uniq
