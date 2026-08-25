MISM-DyCLA
Maximum Independent Set integrated with Dynamic Conjugate Learning Automata for Influence Maximization

A hybrid framework for Influence Maximization (IM) in temporal multiplex social networks. MISM-DyCLA combines layer wise Maximum Independent Set (MIS) preprocessing, multi-agent adaptive learning automata (DyCLA), and greedy seed refinement to select high-quality seed sets under evolving network conditions-outperforming static and full graph baselines.

Overview
Influence Maximization is defined as the computational problem of selecting a small, optimal set
of initial nodes, known as seeds from a social network such that, when they adopt a product, idea or behaviour, the anticipated count of influenced(activated) nodes across the network is most optimal. Given a social network G (directed or undirected) and a positive integer k (k ≤ N), nodes in the network are added to the seed set with respect to the specified diffusion model with the aim of optimizing the influence spread. Most classical algorithms assume network to be static or single-layered but networks are multiplex (having multiple layers of interactions) and temporal (keep on changing with time).

## MISM-DyCLA Pipeline

MISM-DyCLA addresses these gaps with a six-stage pipeline applied to temporal snapshots of the network:

1. **Temporal Snapshot Construction**  
   Bin timestamped edges into monthly or weekly snapshots.

2. **Layer-wise MIS Extraction**  
   Compute the Maximum Independent Set for each interaction layer to identify structurally non-redundant candidates.

3. **Candidate Pool Construction**  
   Merge and rank per-layer MIS nodes by centrality to reduce the action space.

4. **DyCLA Adaptive Learning**  
   K conjugate learning automata (eDGPA) jointly learn which candidate combinations maximize influence spread. A shared team reward signal is used across snapshots.

5. **Greedy Seed Refinement**  
   Apply greedy marginal-gain refinement to the learned distributions to obtain a high-quality seed set.

6. **Influence Spread Evaluation**  
   Evaluate the final seed set using Monte Carlo simulations under the Independent Cascade model.

## Repository Structure

```text
mism-algorithm/
├── .gitignore
├── README.md
├── requirements.txt
├── parameters.txt
├── temporalresults.txt
│
├── algorithm/
│   ├── EnronAlgo/
│   │   ├── dataset_load.py
│   │   ├── department_mapping.py
│   │   ├── enronemail.py
│   │   ├── enronsnapshots.py
│   │   ├── layer_wise_mis.py
│   │   ├── main_experiment.py
│   │   └── weightedcascade.py
│   │
│   └── MathsAlgo/
│       ├── dataset_load_maths.py
│       ├── influencespread.py
│       ├── layerwisemis.py
│       ├── mainexp.py
│       ├── maths.py
│       ├── monthlysnapshots.py
│       └── timebinning.py
│
├── data/
│   ├── datasetlinks.py
│   ├── EnronEmail/
│   └── MathsOverFlow/
│
└── results/
    ├── enronresults/
    └── mathsoverflowresults/
```

Datasets
| Dataset       | Nodes | Temporal Edges | Layers                    | Snapshotting                 | Diffusion Model                          |
| ------------- | ------ | -------------- | ------------------------- | ---------------------------- | ---------------------------------------- |
| Enron Email   |    986 |        332,334 | Dept1–Dept4 + Inter-layer | Weekly (78 active snapshots) | Weighted Cascade, p(u,v) = 1 / in-deg(v) |
| MathsOverflow | 24,818 |        506,550 | A2Q, C2A, C2Q             | Monthly (80 snapshots)     | Independent Cascade, fixed p = 0.03      |

Both datasets are sourced from the SNAP temporal network collection:

email-Eu-core-temporal ( Dept1–Dept4 department files)
sx-mathoverflow (a2q, c2a, c2q interaction files)

Installation
git clone https://github.com/Purnimak26/mism-dycla.git
cd mism-dycla
pip install -r requirements.txt

Baseline Methods

| Method     | Description                                                              |
| ---------- | ------------------------------------------------------------------------ |
| MISM-DyCLA | Full pipeline: MIS pooling + DyCLA adaptive learning + greedy refinement |
| MIS-static | Same MIS candidate pool, seeds chosen by static degree ranking           |
| DyCLA-pure | DyCLA trained on the full node set without MIS pooling                   |
| Random     | K nodes drawn uniformly at random from the MIS pool                      |

Results Summary
| Dataset       | K |   MISM-DyCLA | MIS-static | DyCLA-pure | Random |
| ------------- | -- | ------------ | ----------| ---------- | -------|
| Enron Email   | 10 | 44.2 (±22.9) |       33.6|       14.5 |   16.1 |
| Enron Email   | 15 | 54.0 (±24.2) |       39.3|       21.0 |   23.4 |
| MathsOverflow | 10 | 24.5 (±6.7)  |       23.3|       10.0 |   13.0 |
| MathsOverflow | 15 | 33.4 (±7.7)  |       32.0|       15.1 |   19.8 |

##Key Findings
**MISM-DyCLA consistently performs better than other baselines.**
**MIS candidate pooling helps in significant action space reduction.**
**Adaptive learning depends on diffusion model**
**Advantage grows with seed budget.**

Conclusion
In summary, MISM-DyCLA is a novel algorithm that unifies three components: - structural
diversity through MIS, adaptive learning through DyCLA, approximation guarantees through
greedy refinement and experimental validation on real-world datasets. The path forward is clear
focusing on richer diffusion models, efficient deep leaning mechanisms, real-time adaptation and
fairness-aware objectives.

##Future Directions
**GNN + Deep RL integration**
**Behaviour-aware diffusion models**
**Broader benchmarking**
**Fairness-constrained IM**

