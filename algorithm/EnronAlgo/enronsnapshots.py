# ============================================================
# 4. INTER-LAYER EDGES
# ============================================================
inter_edges = []
for _, row in main.iterrows():
    u, v, t = row["src"], row["dst"], row["ts"]
    dept_u = DepartmentMapping[u]
    dept_v = DepartmentMapping[v]
    if dept_u.isdisjoint(dept_v):
        inter_edges.append((u, v, t, list(dept_u), list(dept_v)))

inter_edges_tab = pd.DataFrame(
    inter_edges, columns=["src", "dst", "ts", "src_dept", "dst_dept"]
)

# Remove Unknown
inter_edges_tab = inter_edges_tab[
    (inter_edges_tab["src_dept"].apply(lambda x: "Unknown" not in x))
    & (inter_edges_tab["dst_dept"].apply(lambda x: "Unknown" not in x))
]
print(f"Inter-layer edges (after removing Unknown): {len(inter_edges_tab)}")

# ============================================================
# 5. CONVERT TO DAYS / WEEKS
# ============================================================
def add_time_columns(df):
    df["day"] = (df["ts"] / (24 * 3600)).astype(int)
    df["week"] = (df["day"] // 7).astype(int)

for df in [main, dept1, dept2, dept3, dept4, inter_edges_tab]:
    add_time_columns(df)

# ============================================================
# 6. BUILD LAYER GRAPHS + COMBINED
# ============================================================
layers = {}
for name, df in [("Dept1", dept1), ("Dept2", dept2), ("Dept3", dept3), ("Dept4", dept4)]:
    G = nx.DiGraph()
    nodes = set(df["src"]).union(set(df["dst"]))
    G.add_nodes_from(nodes)
    for _, row in df.iterrows():
        G.add_edge(row["src"], row["dst"], week=row["week"])
    layers[name] = G

G_inter = nx.DiGraph()
for _, row in inter_edges_tab.iterrows():
    for sd in row["src_dept"]:
        for dd in row["dst_dept"]:
            G_inter.add_edge(
                row["src"], row["dst"],
                week=row["week"], src_dept=sd, dst_dept=dd
            )

G_combined = nx.MultiDiGraph()
for dept, G in layers.items():
    for u, v, data in G.edges(data=True):
        G_combined.add_edge(u, v, week=data["week"], layer=dept)
for u, v, data in G_inter.edges(data=True):
    G_combined.add_edge(u, v, week=data["week"], layer="inter")

all_nodes = set(G_combined.nodes())
all_weeks = sorted({d["week"] for _, _, d in G_combined.edges(data=True)})

# Weekly snapshots
weekly_snapshots = {}
for week in all_weeks:
    G_week = nx.DiGraph()
    for u, v, key, data in G_combined.edges(keys=True, data=True):
        if data["week"] == week:
            G_week.add_edge(u, v, layer=data["layer"])
    G_week.add_nodes_from(all_nodes)
    weekly_snapshots[week] = G_week

print(f"Built {len(weekly_snapshots)} weekly snapshots (weeks {min(all_weeks)}–{max(all_weeks)})")