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