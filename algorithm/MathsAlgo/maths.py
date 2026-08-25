
import time
from collections import defaultdict, deque
import random

import numpy as np
import pandas as pd
import networkx as nx

# For reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# 1. LOAD DATA 
# ============================================================
header = ["src", "dst", "ts"]

main = pd.read_csv("mathsoverflow(main).csv", sep=" ", names=header)
a2q = pd.read_csv("sx-mathoverflow-a2q.csv", sep=" ", names=header)
c2a = pd.read_csv("sx-mathoverflow-c2a.csv", sep=" ", names=header)
c2q = pd.read_csv("sx-mathoverflow-c2q.csv", sep=" ", names=header)

a2q["layer"] = "A2Q"
c2a["layer"] = "C2A"
c2q["layer"] = "C2Q"

# ============================================================
# 2. TIME BINNING 
# ============================================================
def convert_to_days(df):
    df["day"] = (df["ts"] / (24 * 60 * 60)).astype(int)

def convert_to_months(df):
    df["month"] = (df["day"] // 30).astype(int)

for df in [main, a2q, c2a, c2q]:
    convert_to_days(df)
    convert_to_months(df)

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

# ============================================================
# 5. INFLUENCE SPREAD SIMULATION 
# ============================================================
def simulate_spread(G: nx.DiGraph, seeds, mc=400, p=0.03):
    seeds = set(seeds)
    total = 0.0

    for _ in range(mc):
        activated = set(seeds)
        frontier = deque(seeds)

        while frontier:
            u = frontier.popleft()
            for v in G.successors(u):
                if v not in activated and random.random() < p:
                    activated.add(v)
                    frontier.append(v)

        total += len(activated)

    return total / mc

# ============================================================
# 6. eDGPA LEARNER
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
        self.p /= self.p.sum() + 1e-12

# ============================================================
# 7. DyCLA (Dynamic Conjugate Learning Automata)
# ============================================================
class DyCLA:
    def __init__(self, K, pool_size, phi=0.25):
        self.K = K
        self.phi = phi
        self.las = [eDGPA(pool_size) for _ in range(K)]
        self.prev_sigma = 0.0

    def select_seeds(self, pool_list):
        seeds = []
        used = set()
        for la in self.las:
            for _ in range(12):  # collision-avoidance retry mechanism
                idx = la.choose(epsilon=0.18)
                v = pool_list[idx]
                if v not in used:
                    seeds.append(v)
                    used.add(v)
                    break
            else:
                avail = [i for i in range(len(pool_list)) if pool_list[i] not in used]
                if avail:
                    idx = random.choice(avail)
                    seeds.append(pool_list[idx])
                    used.add(pool_list[idx])
        return seeds[:self.K]  # ensure exactly K or less

    def train_one_round(self, G, pool_list, mc=600):
        # (i) Seed selection
        S = self.select_seeds(pool_list)
        # (ii) Influence simulation
        sigma = simulate_spread(G, S, mc=mc)
        # (iii) Baseline computation
        baseline_seeds = random.sample(pool_list, min(self.K, len(pool_list)))
        baseline = simulate_spread(G, baseline_seeds, mc=mc // 3)
        # (iv) Reward computation: r = (sigma(S) - sigma(S_rand)) / (|V| * 0.08) * alpha
        marginal = sigma - baseline
        n_nodes = G.number_of_nodes()
        reward = (marginal / (n_nodes * 0.08 + 1e-6)) * 4.0  # alpha=4.0 for MathsOverflow
        # (v) Probability update for all K automata (shared team reward)
        for k, la in enumerate(self.las):
            if k < len(S):
                try:
                    idx = pool_list.index(S[k])
                    la.update(idx, reward)
                except ValueError:
                    pass 
        return sigma

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

# ============================================================
# 9. MAIN EXPERIMENT LOOP
# ============================================================
K = 15
MC = 400            
TRAIN_MC = 600       
GREEDY_MC = 180      
TRAIN_ROUNDS = 80    

results = defaultdict(dict)
mis_sizes = {}

print("Starting final comparison...\n")
start_time = time.time()

for month in sorted(monthly_snapshots.keys())[:]:   
    G = monthly_snapshots[month]
    print(f"Month {month:3d} | Nodes: {G.number_of_nodes():4d} | Edges: {G.number_of_edges():5d}", end=" ")

    pool, layer_sizes = get_per_layer_pools(G)
    pool_list = list(pool) if len(pool) >= K else list(G.nodes())
    mis_sizes[month] = len(pool_list)

    print(f"| Pool size: {len(pool_list):3d} | Layers: {len(layer_sizes)}")

    # ── MISM-DyCLA ───────────────────────────────────────────────
    dycla = DyCLA(K, len(pool_list))

    # Warm-start high-degree nodes 
    if pool_list:
        ranked_idx = sorted(range(len(pool_list)),
                            key=lambda i: G.out_degree(pool_list[i]) + G.in_degree(pool_list[i]),
                            reverse=True)
        for la in dycla.las:
            for i in ranked_idx[:min(12, len(ranked_idx))]:
                la.p[i] += 0.12
            la.p /= la.p.sum() + 1e-12

    for _ in range(TRAIN_ROUNDS):
        dycla.train_one_round(G, pool_list, mc=TRAIN_MC)

   
    greedy_candidates = sorted(pool_list,
                               key=lambda v: G.out_degree(v) + G.in_degree(v),
                               reverse=True)[:max(40, len(pool_list) // 3)]

    current_seeds = []
    for _ in range(K):
        best_idx, best_gain = -1, -np.inf
        for idx, v in enumerate(greedy_candidates):
            if v in current_seeds:
                continue
            trial = current_seeds + [v]
            gain = (simulate_spread(G, trial, mc=GREEDY_MC) -
                    simulate_spread(G, current_seeds, mc=GREEDY_MC))
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx >= 0:
            current_seeds.append(greedy_candidates[best_idx])

    spread_mism = simulate_spread(G, current_seeds, mc=MC)
    results[month]["MISM-DyCLA"] = round(spread_mism, 2)

    # ── Baselines ─────────────────────────────────────────────────
    static_seeds = sorted(pool_list,
                          key=lambda v: G.out_degree(v) + G.in_degree(v),
                          reverse=True)[:K]
    results[month]["MIS-static"] = round(simulate_spread(G, static_seeds, mc=MC), 2)

    # DyCLA-pure on full graph 
    full_list = list(G.nodes())
    dycla_pure = DyCLA(K, len(full_list))
    for _ in range(TRAIN_ROUNDS // 2):
        dycla_pure.train_one_round(G, full_list, mc=TRAIN_MC // 2)
    seeds_dycla = dycla_pure.select_seeds(full_list)
    results[month]["DyCLA (pure)"] = round(simulate_spread(G, seeds_dycla, mc=MC), 2)

    rand_seeds = random.sample(pool_list, min(K, len(pool_list)))
    results[month]["Random"] = round(simulate_spread(G, rand_seeds, mc=MC), 2)

    print(f"  MISM-DyCLA: {results[month]['MISM-DyCLA']:6.1f}  "
          f"  static: {results[month]['MIS-static']:6.1f}  "
          f"  pure: {results[month]['DyCLA (pure)']:6.1f}  "
          f"  rand: {results[month]['Random']:6.1f}")

# ============================================================
# 10. RESULTS TABLE
# ============================================================
#Sample(not actual values)
df = pd.DataFrame(results).T
df["MIS_pool_size"] = pd.Series(mis_sizes)
df = df.round(2)
df.index.name = "month"

print("\n" + "=" * 100)
print(f"FINAL RESULTS: MIS Pool Size + Spread per Month (K={K}, IC p=0.03)")
print("=" * 100)
print(df[["MIS_pool_size", "MISM-DyCLA", "MIS-static", "DyCLA (pure)", "Random"]])

print("\nAverage Spread Across Months:")
for col in ["MISM-DyCLA", "MIS-static", "DyCLA (pure)", "Random"]:
    print(f"  {col:16}: {df[col].mean():6.1f}  (std = {df[col].std():5.1f})")

df.to_csv("final_comparison_mathoverflow.csv", index=True)
print("\nResults saved to: final_comparison_mathoverflow.csv")
print(f"Total runtime: {(time.time() - start_time)/60:.1f} minutes")
print("Done!")