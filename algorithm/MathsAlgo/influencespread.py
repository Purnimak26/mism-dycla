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
