# Models

These two models are a deep dive into 
    Bounded Rationality and Strategic Evolution. 

I’m a big Game Theory nerd, so I built these to see if I could make complex ideas like QRE and Cognitive Hierarchies actually work in a living, breathing Mesa simulation.

## Model 1: Cognitive Hierarchy Market (`ch_market_model.py`)

### What it does

A double-auction market where agents differ in strategic sophistication. 
Based on Camerer, Ho & Chong's Cognitive Hierarchy theory: 
    level-0 agents bid randomly
    level-k agents best-respond to a mix of levels 0 through k-1. 
    
Each agent also holds a Bayesian prior (Gamma distributed) over the population's mean sophistication parameter τ, which they update after observing market-clearing prices.

### Mesa features used

- `mesa.Agent` and `mesa.Model` base classes
- `AgentSet` via `self.agents` — adding agents and calling `shuffle_do`
- `mesa.DataCollector` with model-level reporters (`AvgTau`, `Efficiency`)

### What I learned

- `shuffle_do` is clean for synchronous steps, but I had to be careful not to assume agent ordering in the market-clearing logic that follows.
- `DataCollector` reporters are lambdas evaluated at collect time, not at step time — this tripped me up when I first tried to reference `self.last_bid` before agents had acted.
- Separating cognitive-level assignment (at init) from belief updating (per step) felt natural with Mesa's agent lifecycle.

### What was hard

The market-clearing logic sits outside the agents entirely (in `Model.step`), which felt slightly unnatural — in a real DA market, the matching happens between agents. I ended up treating it as a centralised auctioneer, which is a simplification. A more realistic version would probably use Mesa's event scheduler to let agents post orders asynchronously.

The `calculate_efficiency` placeholder is honest: computing true allocative efficiency requires tracking each agent's private valuation against what they traded at, which I didn't wire up fully.

### What I'd do differently

- Replace the efficiency placeholder with real surplus tracking (store private values at role assignment, compare against final trade price)
- Use `mesa.time.SimultaneousActivation` style two-phase stepping more explicitly — the current version conflates bid generation and belief updating in a way that could cause subtle ordering bugs
- Add a `PropertyLayer` for spatial market structure (e.g. agents in different "regions" with limited order flow)

---

## Model 2: QRE Social Evolution (`qre_social_evol.py`)

### What it does

Agents play a three-strategy Hawk-Dove-Bourgeois game. Each agent has a rationality parameter λ (lambda): higher λ means sharper best-response, lower λ means more random play (the Quantal Response Equilibrium model). Agents update λ by imitating more successful neighbors — social learning under noise. The model explores how imitation pressure and mutation noise jointly shape whether populations evolve toward high or low rationality.

Includes a batch experiment (`mesa.batch_run`) crossing imitation probability and noise level across seeds, with a heatmap of final mean λ.

### Mesa features used

- Two-phase stepping via `shuffle_do("step")` + `shuffle_do("advance")` — decision phase separated from interaction/learning phase
- `mesa.DataCollector` with three model reporters (`MeanLambda`, `MeanPayoff`, `PopEntropy`)
- `mesa.batch_run` with parameter sweep and multi-seed replication
- `pd.DataFrame` post-processing of batch results

### What I learned

- The two-phase `step` / `advance` pattern is genuinely useful here 
it prevents agents from reacting to actions that haven't been decided yet in the same round. Mesa makes this easy; I just had to not mix the phases.

- `batch_run` is powerful but the returned DataFrame column naming for model reporters needs care — I had to filter by `Step == 100` and group manually to get the analysis I wanted.

- `PopEntropy` as a reporter (Shannon entropy over mean population strategy mix) gave me a much richer signal than `MeanLambda` alone — it reveals whether the population converges to a sharp strategy or stays diffuse.

### What was hard

The imitation logic samples a random agent from the full `AgentSet` — this means popular/lucky agents can propagate their λ widely in one step, which may be unrealistically fast diffusion. A network-space version where agents only observe neighbors would be more realistic and would let me explore topology effects on rationality evolution.

The `batch_run` filter on `lam_noise` uses `np.isclose` because floating-point parameter values don't survive the DataFrame round-trip cleanly. Took me a while to diagnose why exact equality was silently returning empty DataFrames.

### What I'd do differently

- Put agents on a `mesa.spaces.NetworkGrid` and restrict imitation to graph neighbors — this is the natural next step and directly tests whether local vs global information changes the evolutionary outcome
- Track the full λ distribution per step (not just mean) — the mean hides bimodal populations where hawks and doves split into high/low rationality clusters
- Separate the `run_experiment` batch logic from the single-run model so the model can be used interactively in Mesa's visualization without triggering a batch run

---

