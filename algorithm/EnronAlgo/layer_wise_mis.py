# ============================================================
# 9. PER-LAYER MIS POOL 
# ============================================================
def get_per_layer_pools(G, max_iter=180):
    layers_present = sorted({d["layer"] for _, _, d in G.edges(data=True)})
    combined = set()
    per_layer_sizes = {}

    for lay in layers_present:
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("layer") == lay]
        if not edges:
            continue
        Gl = nx.Graph(edges)
        Gl.add_nodes_from(G.nodes())
        nodes = list(Gl.nodes())
        if len(nodes) < 10:
            continue

        las = {v: eDGPA(2) for v in nodes}
        best, best_size = set(), 0
        adj = defaultdict(set)
        for u, v in Gl.edges():
            adj[u].add(v)
            adj[v].add(u)

        for _ in range(max_iter):
            S = {v for v in nodes if las[v].choose() == 1}
            valid = all(not any(w in S for w in adj[u] if w > u) for u in S)
            size = len(S) if valid else 0
            if size > best_size:
                best_size = size
                best = S.copy()
            reward = 1.8 if valid and size >= best_size * 0.8 else -0.6
            for v in S:
                las[v].update(1, reward)

        # greedy fill
        remaining = sorted(nodes, key=lambda v: -Gl.degree(v))
        for v in remaining:
            if v not in best and all(v not in adj[u] for u in best):
                best.add(v)
                if len(best) >= 0.28 * len(nodes):
                    break

        per_layer_sizes[lay] = len(best)
        combined.update(best)

    if combined:
        ranked = sorted(combined, key=lambda v: G.out_degree(v) + G.in_degree(v), reverse=True)
        max_keep = max(60, int(0.45 * G.number_of_nodes()))
        combined = set(ranked[:max_keep])
    return combined, per_layer_sizes