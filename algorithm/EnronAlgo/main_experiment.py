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