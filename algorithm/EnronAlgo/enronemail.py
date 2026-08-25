
import time
from collections import defaultdict, deque
import random

import numpy as np
import pandas as pd
import networkx as nx

# ============================================================
# 1. LOAD DATA
# ============================================================

header = ["src", "dst", "ts"]

main = pd.read_csv("email-Eu-core-temporal (2).txt", sep=" ", names=header)

dept1 = pd.read_csv("email-Eu-core-temporal-Dept1 (2).txt", sep=" ", names=header)

dept2 = pd.read_csv("email-Eu-core-temporal-Dept2 (1).txt", sep=" ", names=header)

dept3 = pd.read_csv("email-Eu-core-temporal-Dept3 (1).txt", sep=" ", names=header)

dept4 = pd.read_csv("email-Eu-core-temporal-Dept4 (1).txt", sep=" ", names=header)

dept1["dept"] = "Dept1"
dept2["dept"] = "Dept2"
dept3["dept"] = "Dept3"
dept4["dept"] = "Dept4"


# ============================================================
# 2. DEPARTMENT MAPPING
# ============================================================

DepartmentMapping = defaultdict(set)


def dept_map(df):
    dept = df["dept"].iloc[0]

    for u, v in zip(df["src"], df["dst"]):
        DepartmentMapping[u].add(dept)
        DepartmentMapping[v].add(dept)


dept_map(dept1)
dept_map(dept2)
dept_map(dept3)
dept_map(dept4)


print("Department Mapping")

for node, dept in list(DepartmentMapping.items())[:20]:
    print("Node", node, "->", dept)

all_nodes = len(DepartmentMapping)

multi_dept_nodes = sum(
    1 for d in DepartmentMapping.values()
    if len(d) > 1
)

print("Total nodes:", all_nodes)
print("Nodes in multiple departments:", multi_dept_nodes)


# ============================================================
# 3. FIND MISSING NODES
# ============================================================

main_nodes = set(main["src"]).union(set(main["dst"]))
dept_nodes = set(DepartmentMapping.keys())

missing_nodes = main_nodes - dept_nodes

print("Missing nodes:", len(missing_nodes))

for node in missing_nodes:
    DepartmentMapping[node] = {"Unknown"}

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

# ============================================================
# 7. WEIGHTED CASCADE MODEL
# ============================================================
def simulate_spread(G: nx.DiGraph, seeds, mc=800, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    seeds = set(seeds)
    in_deg = dict(G.in_degree())
    total = 0.0
    for _ in range(mc):
        activated = set(seeds)
        q = deque(seeds)
        while q:
            u = q.popleft()
            for v in G.successors(u):
                if v in activated:
                    continue
                p = 1.0 / max(in_deg.get(v, 1), 1)
                if random.random() < p:
                    activated.add(v)
                    q.append(v)
        total += len(activated)
    return total / mc

# ============================================================
# 8. eDGPA + DyCLA
# ============================================================
class eDGPA:
    def __init__(self, n_actions, R=20):
        self.n = n_actions
        self.delta = 1.0 / (R * n_actions)
        self.p = np.ones(n_actions) / n_actions
        self.Z = np.ones(n_actions)
        self.R = np.zeros(n_actions)

    def choose(self, epsilon=0.15):
        if random.random() < epsilon:
            return random.randint(0, self.n - 1)
        return np.random.choice(self.n, p=self.p)

    def update(self, action, reward):
        self.Z[action] += 1
        self.R[action] = (self.R[action] * (self.Z[action] - 2) + reward) / self.Z[action]
        best = np.argmax(self.R)
        for i in range(self.n):
            if i == best:
                self.p[i] = min(self.p[i] + self.delta * 1.3, 1.0)
            else:
                self.p[i] = max(self.p[i] - self.delta / (self.n - 1), 0.0)
        self.p = np.clip(self.p, 0, 1)
        self.p /= self.p.sum()

class DyCLA:
    def __init__(self, K, pool_size, phi=0.25):
        self.K = K
        self.phi = phi
        self.las = [eDGPA(pool_size) for _ in range(K)]
        self.prev_sigma = 0.0

    def select_seeds(self, pool_list):
        seeds, used = [], set()
        for la in self.las:
            for _ in range(12):
                idx = la.choose(epsilon=0.18)
                v = pool_list[idx]
                if v not in used:
                    seeds.append(v)
                    used.add(v)
                    break
            else:
                avail = [i for i, n in enumerate(pool_list) if n not in used]
                if avail:
                    idx = random.choice(avail)
                    seeds.append(pool_list[idx])
                    used.add(pool_list[idx])
        return seeds

    def train_one_round(self, G, pool_list, mc=600):
        S = self.select_seeds(pool_list)
        sigma = simulate_spread(G, S, mc)
        baseline = simulate_spread(G, random.sample(pool_list, min(self.K, len(pool_list))), mc=200)
        marginal = sigma - baseline
        reward = marginal / (G.number_of_nodes() * 0.08 + 1e-6) * 3.5
        for k, la in enumerate(self.las):
            idx = pool_list.index(S[k % len(S)])
            la.update(idx, reward)
        return sigma

    def update_dynamic(self, current_sigma):
        delta = abs(current_sigma - self.prev_sigma)
        for la in self.las:
            m = np.argmax(la.p)
            psi = min(self.phi * delta / (current_sigma + 1e-6) * la.p[m], 0.85)
            la.p += (1.0 / (la.n - 1)) * psi
            la.p[m] -= psi
            la.p = np.clip(la.p, 0, 1)
            la.p /= la.p.sum()
            la.R += np.random.normal(0, 0.12 * delta, la.n)
        self.prev_sigma = current_sigma

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

# ============================================================
# 10. MAIN EXPERIMENT
# ============================================================
#Sample(not actual values)
K = 15
MC_FINAL = 1200          
TRAIN_ROUNDS = 45        
MC_TRAIN = 500

results = defaultdict(dict)
mis_sizes = {}

print("\nRunning comparison on all weeks...\n")
start_total = time.time()

for week in sorted(weekly_snapshots.keys()):
    t0 = time.time()
    G = weekly_snapshots[week]
    if G.number_of_edges() == 0:
        continue

    pool, layer_sizes = get_per_layer_pools(G)
    pool_list = list(pool) if len(pool) >= K else list(G.nodes())
    mis_size = len(pool_list)
    mis_sizes[week] = mis_size

    print(f"Week {week:3d} | Nodes: {G.number_of_nodes():4d} | "
          f"Edges: {G.number_of_edges():5d} | MIS pool: {mis_size:3d}")

    # ----- MISM-DyCLA -----
    dycla = DyCLA(K, len(pool_list))
    # warm-start
    ranked = sorted(range(len(pool_list)),
                    key=lambda i: G.out_degree(pool_list[i]) + G.in_degree(pool_list[i]),
                    reverse=True)
    for la in dycla.las:
        for i in ranked[:min(12, len(ranked))]:
            la.p[i] += 0.10
        la.p /= la.p.sum()

    for _ in range(TRAIN_ROUNDS):
        dycla.train_one_round(G, pool_list, mc=MC_TRAIN)

    # greedy refinement on pool
    current_seeds = []
    for _ in range(K):
        best_idx, best_gain = -1, -np.inf
        for idx, cand in enumerate(pool_list):
            if cand in current_seeds:
                continue
            trial = current_seeds + [cand]
            gain = (simulate_spread(G, trial, mc=250) -
                    simulate_spread(G, current_seeds, mc=250))
            if gain > best_gain:
                best_gain, best_idx = gain, idx
        if best_idx >= 0:
            current_seeds.append(pool_list[best_idx])

    spread_mism = simulate_spread(G, current_seeds, MC_FINAL)
    results[week]["MISM-DyCLA"] = spread_mism

    # ----- MIS-static -----
    static_seeds = sorted(pool_list,
                          key=lambda v: G.out_degree(v) + G.in_degree(v),
                          reverse=True)[:K]
    results[week]["MIS-static"] = simulate_spread(G, static_seeds, MC_FINAL)

    # ----- DyCLA-pure -----
    full_list = list(G.nodes())
    dycla_pure = DyCLA(K, len(full_list))
    for _ in range(TRAIN_ROUNDS // 2):
        dycla_pure.train_one_round(G, full_list, mc=MC_TRAIN // 2)
    seeds_pure = dycla_pure.select_seeds(full_list)
    results[week]["DyCLA-pure"] = simulate_spread(G, seeds_pure, MC_FINAL)

    # ----- Random -----
    rand_seeds = random.sample(pool_list, min(K, len(pool_list)))
    results[week]["Random"] = simulate_spread(G, rand_seeds, MC_FINAL)

    print(f"  MISM-DyCLA : {results[week]['MISM-DyCLA']:7.1f}")
    print(f"  MIS-static : {results[week]['MIS-static']:7.1f}")
    print(f"  DyCLA-pure : {results[week]['DyCLA-pure']:7.1f}")
    print(f"  Random     : {results[week]['Random']:7.1f}")
    print(f"  Time: {time.time()-t0:.1f}s\n")

# ============================================================
# 11. RESULTS
# ============================================================
df = pd.DataFrame(results).T.round(2)
df.index.name = "week"
df["MIS_pool_size"] = pd.Series(mis_sizes)

print("\n" + "=" * 100)
print("FINAL RESULTS (K=15)")
print("=" * 100)
print(df[["MIS_pool_size", "MISM-DyCLA", "MIS-static", "DyCLA-pure", "Random"]])

print("\nAverage spread across weeks:")
for col in ["MISM-DyCLA", "MIS-static", "DyCLA-pure", "Random"]:
    print(f"  {col:12s}: {df[col].mean():.1f}  (std={df[col].std():.1f})")

df.to_csv("full_graph_comparison.csv")
print(f"\nTotal runtime: {(time.time()-start_total)/60:.1f} minutes")
print("Results saved → full_graph_comparison.csv")
print("Done!")