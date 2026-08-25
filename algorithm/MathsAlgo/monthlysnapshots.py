# ============================================================
# 3. BUILD MULTIPLEX GRAPH
# ============================================================
G_multi = nx.MultiDiGraph()

for layer_name, df in [("A2Q", a2q), ("C2Q", c2q), ("C2A", c2a)]:
    for row in df.itertuples():
        G_multi.add_edge(
            row.src, row.dst,
            month=row.month,
            layer=layer_name
        )

# ============================================================
# 4. MONTHLY SNAPSHOTS 
# ============================================================
monthly_snapshots = {}
all_months = sorted(set(nx.get_edge_attributes(G_multi, "month").values()))

for month in all_months:
    G_month = nx.DiGraph()
    for u, v, key, data in G_multi.edges(keys=True, data=True):
        if data.get("month") == month:
            G_month.add_edge(u, v, layer=data.get("layer", "unknown"))
    # Preserve all nodes (important for consistent node set across snapshots)
    G_month.add_nodes_from(G_multi.nodes())
    monthly_snapshots[month] = G_month

print(f"Created {len(monthly_snapshots)} monthly snapshots")
