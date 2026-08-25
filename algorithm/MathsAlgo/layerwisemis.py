
# ============================================================
# 8. CANDIDATE POOL: per-layer greedy MIS approximation 
# ============================================================
def get_per_layer_pools(G, max_candidates=250):
    layers = sorted({d["layer"] for _, _, d in G.edges(data=True) if "layer" in d})
    combined = set()
    per_layer_sizes = {}

    for lay in layers:
        edges_lay = [(u, v) for u, v, d in G.edges(data=True) if d.get("layer") == lay]
        if len(edges_lay) < 10:
            continue
        H = nx.Graph(edges_lay)  # undirected for MIS
        H.add_nodes_from(G.nodes())

        # Greedy MIS approximation (fast): sort by degree, add if independent
        mis = set()
        nodes_sorted = sorted(H.nodes(), key=lambda v: H.degree(v), reverse=True)
        for v in nodes_sorted:
            if all(nb not in mis for nb in H.neighbors(v)):
                mis.add(v)

        per_layer_sizes[lay] = len(mis)
        combined.update(mis)

    # Final selection: top nodes by total degree in original directed graph (Stage 3)
    if combined:
        deg_sum = {v: G.in_degree(v) + G.out_degree(v) for v in combined}
        ranked = sorted(deg_sum, key=deg_sum.get, reverse=True)
        keep = min(max_candidates, max(80, int(0.45 * G.number_of_nodes())))
        combined = set(ranked[:keep])

    return combined, per_layer_sizes
