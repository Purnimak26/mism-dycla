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
